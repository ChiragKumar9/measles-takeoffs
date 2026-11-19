"""
School-level MMR coverage from two sources, and its county-level summary.

Sources
    Chen & Bento (PMID 42613442): per-state school files, many school years,
    no school names, types or exemption rates (Zenodo 10.5281/zenodo.19430274
    and 10.5281/zenodo.20174071; CC-BY-4.0).
    Washington Post: one school year per state, with names, addresses, types and
    exemption rates (data/vaxrates_schools.csv).
    State files (STATE_FILE): school-level files published by the states (the
    "state files" of the school-data zip), read by school_state_files.py. They
    carry the county the file names and no coordinates. Optional: without
    them, the state-years only they cover are missing.

Rates are percentages (0-100). A school year is labelled "2023-2024"; its
`year_start` (2023) is how data/processed_vaccinations_data.csv labels it, and
its `year_end` (2024) is the year of the outbreak data that uses it.

Which source a state and school year uses (select_sources)
    One source for each state and school year, the first of: the Washington
    Post, Chen & Bento, a state file. The Washington Post is used only if it
    has at least wapo_min_share_of_chen_bento (params.yml) of the rated schools
    Chen & Bento has in that state and school year (it is then not a partial
    file). The other years of a state come from Chen & Bento, and a state-year
    that neither covers from the state files.

The steps, in this order (assemble_school_vaccination.py runs them)
    1. read both sources and put them in one layout
    2. place checks: a school is placed in a county of its own state
    3. fallback points (many records on one coordinate) are dropped
    4. value checks: rates outside 0-100 and a district row at exactly 0%
       (missing), unknown enrollment, a rate of 0 with no enrollment (Chen &
       Bento only)
    5. duplicates (same point and rate, enrollment the same or missing), unless another
       school year shows two or more schools at the point
    6. copied years are flagged
    7. source selection (per state-year), then borrowed enrollment and
       inherited school types
    8. flags, the school-level file, the county-level file
    9. the county MMR QC flag of the county file (add_county_mmr_qc)

Every dropped or changed row is written to the QC log with its reason.
"""

import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from pyproj import Transformer
from scipy.spatial import cKDTree  # type: ignore

EQUAL_AREA_CRS = "EPSG:5070"  # metres; used for distances only
SCHOOL_TYPES = ["PUBLIC", "PRIVATE"]
WAPO = "washington_post"
CHEN_BENTO = "chen_bento"
STATE_FILE = "state_file"  # the same string as school_state_files.STATE_FILE

COLUMN_TYPES = {
    "rid": pl.UInt32,
    "row_id": pl.String,
    "source": pl.String,
    "state": pl.String,
    "year": pl.String,
    "year_start": pl.Int64,
    "year_end": pl.Int64,
    "school_name": pl.String,
    "address": pl.String,
    "school_type": pl.String,
    "school_type_source": pl.String,
    "unit": pl.String,
    "enrollment": pl.Float64,
    "enrollment_source": pl.String,
    "rate": pl.Float64,
    "rate_source": pl.String,
    "exemption_medical": pl.Float64,
    "exemption_religious": pl.Float64,
    "exemption_personal": pl.Float64,
    "exemption_nonmedical": pl.Float64,
    "exemption_overall": pl.Float64,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "county_label": pl.String,
    "county_fips": pl.String,
    "county_name": pl.String,
    "county_join": pl.String,
    "county_label_differs": pl.Boolean,
    "n_at_point": pl.Int64,
    "copied_year": pl.Boolean,
    "wapo_mmr": pl.Float64,
    "wapo_overall": pl.Float64,
    "band": pl.String,  # a state file that gives a band ("≥95%") instead of a rate: its text (the rate is the midpoint)
}


class QCLog:
    """The rows dropped or changed by each step, and the row counts."""

    def __init__(self):
        self.rows = []
        self.counts = []

    def add(self, df, step, action, reason):
        if df.height == 0:
            return
        columns = [
            "row_id",
            "source",
            "state",
            "year",
            "school_name",
            "lon",
            "lat",
        ]
        self.rows.append(
            df.select(columns).with_columns(
                step=pl.lit(step),
                action=pl.lit(action),
                reason=pl.lit(reason),
            )
        )

    def count(self, source, step, before, after):
        self.counts.append(
            {
                "source": source,
                "step": step,
                "rows_before": before,
                "rows_after": after,
            }
        )

    def rows_frame(self):
        if not self.rows:
            return pl.DataFrame()
        return pl.concat(self.rows, how="diagonal_relaxed")

    def counts_frame(self):
        return pl.DataFrame(self.counts)


def standardize(df):
    """The common column layout; columns a source lacks are empty."""
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in COLUMN_TYPES.items()
        if name not in df.columns
    ]
    return df.with_columns(missing).select(list(COLUMN_TYPES))


def transform_points(transformer, x, y):
    """pyproj on arrays; a single point is padded, as pyproj handles one point differently."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n == 1:
        x, y = np.repeat(x, 2), np.repeat(y, 2)
    new_x, new_y = transformer.transform(x, y)
    return np.asarray(new_x)[:n], np.asarray(new_y)[:n]


def project(df):
    """The points of df (lon, lat) in metres."""
    transformer = Transformer.from_crs(
        "EPSG:4326", EQUAL_AREA_CRS, always_xy=True
    )
    x, y = transform_points(
        transformer, df["lon"].to_numpy(), df["lat"].to_numpy()
    )
    return np.column_stack([x, y])


def unambiguous_pairs(a, b, radius_m):
    """
    Row positions (in a, in b) of the pairs that are each other's only
    candidate: b has exactly one school within radius_m of the school of a, and
    a has exactly one school within radius_m of that school of b. Schools that
    share a point have several candidates and are never paired.
    """
    if a.height == 0 or b.height == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    points_a, points_b = project(a), project(b)
    near_a = cKDTree(points_b).query_ball_point(points_a, r=radius_m)
    near_b = cKDTree(points_a).query_ball_point(points_b, r=radius_m)
    pairs = [
        (i, found[0])
        for i, found in enumerate(near_a)
        if len(found) == 1 and len(near_b[found[0]]) == 1
    ]
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def normalize_county_name(name):
    """A county name without accents, 'Saint', and 'County', 'City', ..."""
    name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode()
        .upper()
    )
    for long, short in (("SAINT", "ST"), ("STE.", "ST"), ("ST.", "ST")):
        name = name.replace(long, short)
    name = "".join(ch for ch in name if ch.isalpha())
    for suffix in ("COUNTY", "CITY", "PARISH", "BOROUGH"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
    return name


# --- 1. reading ---------------------------------------------------------------


def nan_to_missing(df, log):
    """
    A NaN in a number column (the text "NaN" in a file) is a missing value. It
    must not stay NaN: NaN is larger than every number in comparisons and in
    clipping, so a missing rate would become 100%.
    """
    floats = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    found = pl.any_horizontal(
        [pl.col(c).is_nan().fill_null(False) for c in floats]
    )
    log.add(df.filter(found), "value", "changed", "NaN treated as missing")
    return df.with_columns([pl.col(c).fill_nan(None) for c in floats])


def read_chen_bento(raw, log):
    """
    The Chen & Bento schools from data/schools/school_points_raw.csv (all
    columns as text, see analysis/export_school_rds.R): numbers parsed
    (censored counts such as "<20" become missing), Kentucky's metres turned
    into longitude and latitude, and the rate in percent (a state-year whose
    median is above 1.5 is already in percent, the rule of the authors' code).
    A rate outside 0-100 is handled by missing_rates_outside_range.
    """
    known_systems = {"NAD83", "EPSG:4326", "EPSG:3857"}
    unexpected = set(raw["crs"].unique().to_list()) - known_systems
    if unexpected:
        raise ValueError(
            f"unexpected coordinate systems: {sorted(unexpected)}"
        )
    x = raw["x"].cast(pl.Float64).to_numpy().copy()
    y = raw["y"].cast(pl.Float64).to_numpy().copy()
    metres = (raw["crs"] == "EPSG:3857").to_numpy()
    if metres.any():
        web_mercator = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )
        x[metres], y[metres] = transform_points(
            web_mercator, x[metres], y[metres]
        )
    median = pl.col("mmr").median().over("state", "year")
    df = raw.with_row_index("rid").with_columns(
        lon=pl.Series(x),
        lat=pl.Series(y),
        enrollment=pl.col("enrollment").cast(pl.Float64, strict=False),
        mmr=pl.col("mmr").cast(pl.Float64, strict=False),
        row_id=pl.concat_str(pl.lit(f"{CHEN_BENTO}:"), pl.col("rid")),
        source=pl.lit(CHEN_BENTO),
        school_name=pl.lit(None, dtype=pl.String),
    )
    df = nan_to_missing(df, log).with_columns(
        year_start=pl.col("year").str.slice(0, 4).cast(pl.Int64),
        year_end=pl.col("year").str.slice(5, 4).cast(pl.Int64),
        rate=pl.when(median > 1.5)
        .then(pl.col("mmr"))
        .otherwise(pl.col("mmr") * 100),
        rate_source=pl.lit(CHEN_BENTO),
    )
    return standardize(df)


def missing_rates_outside_range(df, log):
    """A rate below 0 or above 100 is an error, not a rate near the edge: it becomes missing."""
    outside = ((pl.col("rate") < 0) | (pl.col("rate") > 100)).fill_null(False)
    log.add(
        df.filter(outside),
        "value",
        "changed",
        "rate outside 0-100 treated as missing",
    )
    return df.with_columns(
        rate=pl.when(outside).then(None).otherwise(pl.col("rate"))
    )


def read_washington_post(raw, log):
    """The Washington Post schools: rates in percent, MMR else overall rate."""
    df = raw.with_row_index("rid").with_columns(
        row_id=pl.concat_str(pl.lit(f"{WAPO}:"), pl.col("rid")),
        source=pl.lit(WAPO),
        lon=pl.col("LON"),
        lat=pl.col("LAT"),
    )
    df = nan_to_missing(df, log).with_columns(
        year_start=pl.col("year").str.slice(0, 4).cast(pl.Int64),
        year_end=pl.col("year").str.slice(5, 4).cast(pl.Int64),
        enrollment=pl.col("students_enrolled").cast(pl.Float64),
        rate=pl.coalesce("mmr_rate", "overall_rate") * 100,
        rate_source=pl.when(pl.col("mmr_rate").is_not_null())
        .then(pl.lit("mmr"))
        .when(pl.col("overall_rate").is_not_null())
        .then(pl.lit("overall")),
        exemption_medical=pl.col("medical_exemption_rate") * 100,
        exemption_religious=pl.col("religious_exemption_rate") * 100,
        exemption_personal=pl.col("personal_exemption_rate") * 100,
        exemption_nonmedical=pl.col("nonmedical_exemption_rate") * 100,
        exemption_overall=pl.col("overall_exemption_rate") * 100,
        county_label=pl.col("county"),
        school_type_source=pl.when(pl.col("school_type").is_not_null()).then(
            pl.lit("reported")
        ),
        copied_year=pl.lit(False),
        wapo_mmr=pl.col("mmr_rate"),
        wapo_overall=pl.col("overall_rate"),
    )
    return standardize(df)


# --- 2. place checks -----------------------------------------------------------


def read_counties(path):
    """The county polygons (Census cartographic boundary file)."""
    counties = gpd.read_file(path).rename(
        columns={
            "GEOID": "county_fips",
            "NAME": "county_name",
            "STUSPS": "county_state",
        }
    )
    return counties[["county_fips", "county_name", "county_state", "geometry"]]


def drop_invalid_coordinates(df, bounds, log):
    (lon_low, lon_high), (lat_low, lat_high) = bounds["lon"], bounds["lat"]
    inside = (
        pl.col("lon").is_between(lon_low, lon_high)
        & pl.col("lat").is_between(lat_low, lat_high)
    ).fill_null(False)
    log.add(
        df.filter(~inside),
        "place",
        "dropped",
        "missing or out-of-range coordinates",
    )
    return df.filter(inside)


def assign_counties(df, counties, max_km, log):
    """
    Each school's county, always one of its own state's. A school inside a
    polygon of its own state takes that county ("contained"); any other school
    (in another state's county, or in no polygon) takes the nearest county of
    its own state ("nearest_in_state"), and is dropped if that is further than
    max_km. The school files come from the state health departments, so the
    state of a school is known, and only its county has to be found.
    """
    points = (
        df.select("state", "lon", "lat")
        .unique()
        .sort("state", "lon", "lat")
        .with_row_index("pid")
    )
    geometry = gpd.points_from_xy(points["lon"], points["lat"])
    located = gpd.GeoDataFrame(
        points.to_pandas(), geometry=geometry, crs=counties.crs
    )
    inside = located.sjoin(counties, how="inner", predicate="within")
    inside = inside[inside["state"] == inside["county_state"]]
    inside = inside.sort_values(["pid", "county_fips"]).drop_duplicates("pid")  # type: ignore
    found = pl.from_pandas(
        inside[["pid", "county_fips", "county_name"]]
    ).with_columns(  # type: ignore
        county_join=pl.lit("contained"),
        distance_m=pl.lit(0.0),
    )
    rest = located[~located["pid"].isin(found["pid"].to_list())]
    counties_m = counties.to_crs(EQUAL_AREA_CRS)
    nearest = []
    for state, part in rest.groupby("state"):
        own = counties_m[counties_m["county_state"] == state]
        if own.empty:
            raise ValueError(f"no county polygons for state {state}")
        near = part.to_crs(EQUAL_AREA_CRS).sjoin_nearest(
            own[["county_fips", "county_name", "geometry"]],
            how="left",
            distance_col="distance_m",
        )
        near = near.sort_values(
            ["pid", "distance_m", "county_fips"]
        ).drop_duplicates("pid")
        nearest.append(
            near[["pid", "county_fips", "county_name", "distance_m"]]
        )
    if nearest:
        near_found = pl.from_pandas(pd.concat(nearest)).with_columns(
            county_join=pl.lit("nearest_in_state")
        )
        found = pl.concat([found, near_found], how="diagonal_relaxed")
    found = found.join(points, on="pid").select(
        "state",
        "lon",
        "lat",
        "county_fips",
        "county_name",
        "county_join",
        "distance_m",
    )
    df = df.drop("county_fips", "county_name", "county_join").join(
        found, on=["state", "lon", "lat"], how="left"
    )
    too_far = pl.col("distance_m") > max_km * 1000
    log.add(
        df.filter(too_far),
        "place",
        "dropped",
        f"more than {max_km} km from the counties of its own state",
    )
    moved = df.filter(~too_far & (pl.col("county_join") == "nearest_in_state"))
    log.add(
        moved,
        "place",
        "changed",
        "outside its state's county polygons: nearest county of its own state",
    )
    return df.filter(~too_far).drop("distance_m")


def check_county_labels(df, counties, max_km, log, skip_states=("CT", "DC")):
    """
    The Washington Post file names a county for each school. The label names
    are compared with the county found from the point (names normalised, and a
    name matching several counties, such as Richmond city and Richmond County,
    counts as all of them). `county_label_differs` is false when the point's
    county is among the labelled ones, true when it is not but lies within
    max_km of one of them, and missing when the label cannot be used
    (Connecticut's labels are planning regions, DC's label is "DC", or the name
    matches no county of the state). A school further than max_km from every
    county its label names is dropped: the point or the label is wrong, and
    nothing says which.
    """
    names = pl.from_pandas(
        counties.drop(columns="geometry").assign(
            label_key=counties["county_name"].map(normalize_county_name)
        )[["county_state", "county_fips", "label_key"]]
    ).rename({"county_state": "state", "county_fips": "label_fips"})
    keyed = df.with_columns(
        label_key=pl.col("county_label").map_elements(
            normalize_county_name, return_dtype=pl.String
        )
    )
    candidates = (
        keyed.select("rid", "state", "label_key")
        .join(names, on=["state", "label_key"], how="inner")
        .group_by("rid")
        .agg(candidates=pl.col("label_fips").sort())
    )
    keyed = keyed.join(candidates, on="rid", how="left")
    usable = pl.col("candidates").is_not_null() & ~pl.col("state").is_in(
        skip_states
    )
    is_member = pl.struct("county_fips", "candidates").map_elements(
        lambda r: r["county_fips"] in (r["candidates"] or []),
        return_dtype=pl.Boolean,
    )
    keyed = keyed.with_columns(member=is_member)
    differs = keyed.filter(usable & ~pl.col("member"))
    counties_m = counties.to_crs(EQUAL_AREA_CRS).set_index("county_fips")[
        "geometry"
    ]
    points = gpd.GeoDataFrame(
        differs.select("rid").to_pandas(),
        geometry=gpd.points_from_xy(differs["lon"], differs["lat"]),
        crs=counties.crs,
    ).to_crs(EQUAL_AREA_CRS)
    distance = {
        rid: min(geom.distance(counties_m[c]) for c in cands) / 1000
        for rid, geom, cands in zip(
            points["rid"], points["geometry"], differs["candidates"]
        )
    }
    keyed = keyed.with_columns(
        label_km=pl.col("rid").map_elements(
            lambda r: distance.get(r), return_dtype=pl.Float64
        )
    )
    far = usable & ~pl.col("member") & (pl.col("label_km") > max_km)
    log.add(
        keyed.filter(far),
        "place",
        "dropped",
        f"more than {max_km} km from the county its label names",
    )
    kept = keyed.filter(~far).with_columns(
        county_label_differs=pl.when(~usable)
        .then(None)
        .when(pl.col("member"))
        .then(False)
        .otherwise(True)
    )
    log.add(
        kept.filter(pl.col("county_label_differs") == True),  # noqa: E712
        "place",
        "flagged",
        "the county of the point differs from the county of the label",
    )
    return kept.drop("label_key", "candidates", "member", "label_km")


# --- 3. value checks -------------------------------------------------------------


def value_checks(df, log):
    """
    A school with an enrollment of 0 or less (an error or a "no data" code; the
    data release does not say which) has no students to be vaccinated, so its
    rate means nothing either: both enrollment and rate become missing, the same
    in every state, and the school leaves the school file, which keeps only
    schools with a rate. A rate of exactly 0 where the enrollment is missing
    (not zero) is a placeholder for missing data and becomes missing (in the
    Chen & Bento file 162 of the 165 rows below 10% with no enrollment are
    exactly 0.0, 144 of them in South Dakota); with known enrollment a 0 is a
    school and is kept. The placeholder rule is for Chen & Bento only: the
    Washington Post has no enrollment at all, so there a rate with "missing
    enrollment" says nothing, and the state files are not checked by it
    either (New York's has no enrollment column and 1,020 zeros).
    """
    zero = (pl.col("enrollment") <= 0).fill_null(False)
    missing = pl.col("enrollment").is_null()
    placeholder = (
        (pl.col("rate") == 0) & missing & (pl.col("source") == CHEN_BENTO)
    ).fill_null(False)
    log.add(
        df.filter(zero),
        "value",
        "changed",
        "enrollment of 0 or less: enrollment and rate set to missing",
    )
    log.add(
        df.filter(placeholder),
        "value",
        "changed",
        "Chen & Bento rate of exactly 0 with missing enrollment treated as missing",
    )
    return df.with_columns(
        enrollment=pl.when(zero).then(None).otherwise(pl.col("enrollment")),
        rate=pl.when(zero | placeholder).then(None).otherwise(pl.col("rate")),
    ).with_columns(
        enrollment_source=pl.when(pl.col("enrollment").is_not_null()).then(
            pl.col("source")
        )
    )


def unit_of_row(district_states):
    """
    A row says whether it is a school or a school district: Washington Post by
    name (ISD or school district), Chen & Bento by state (district_states,
    params.yml chen_bento_district_states). A state file row keeps the unit its
    reader gave it, or is a school. Empty for other sources.
    """
    return (
        pl.when(pl.col("source") == WAPO)
        .then(
            pl.when(
                pl.col("school_name").str.contains(
                    r"(?i)\bISD\b|school district"
                )
            )
            .then(pl.lit("district"))
            .otherwise(pl.lit("school"))
        )
        .when(pl.col("source") == CHEN_BENTO)
        .then(
            pl.when(pl.col("state").is_in(district_states))
            .then(pl.lit("district"))
            .otherwise(pl.lit("school"))
        )
        .when(pl.col("source") == STATE_FILE)
        .then(pl.col("unit").fill_null("school"))
    )


def missing_district_zero_rates(df, district_states, log):
    """
    A district row (not a school row) with a rate of exactly 0% is an error: a
    whole district without one vaccinated child. In the data this is the row
    "TEXAS TECH UNIVERSITY K-12" (Lubbock County, 2024-25), which the
    Washington Post leaves blank and which moved Lubbock's county SD from 8.5
    to 22.8 points, and one zip row of a charter academy. School rows at 0%
    are small schools and are kept.
    """
    zero = (
        (unit_of_row(district_states) == "district") & (pl.col("rate") == 0)
    ).fill_null(False)
    log.add(
        df.filter(zero),
        "value",
        "changed",
        "district rate of exactly 0% treated as missing",
    )
    return df.with_columns(
        rate=pl.when(zero).then(None).otherwise(pl.col("rate"))
    )


# --- 4. duplicates ------------------------------------------------------------------


def drop_duplicates_washington_post(df, log):
    keys = ["state", "year", "school_name", "lon", "lat", "rate"]
    return _drop_same_point_and_rate(
        df,
        keys,
        "identical Washington Post record (same name, point and rate",
        log,
    )


def _points_with_two_schools_in_another_year(df):
    """
    The (state, year, point) rows whose point holds records with two or more
    different rates in another school year of the same state: two or more real
    schools share the point, so records at it with one rate are not copies.
    """
    by_year = (
        df.filter(pl.col("rate").is_not_null())
        .group_by("state", "year", "lon", "lat")
        .agg(distinct_rates=pl.col("rate").n_unique())
    )
    several = by_year.filter(pl.col("distinct_rates") >= 2).select(
        "state", "lon", "lat", other_year="year"
    )
    return (
        by_year.select("state", "year", "lon", "lat")
        .join(several, on=["state", "lon", "lat"])
        .filter(pl.col("year") != pl.col("other_year"))
        .select("state", "year", "lon", "lat")
        .unique()
        .with_columns(confirmed_distinct=pl.lit(True))
    )


def _drop_same_point_and_rate(df, keys, reason, log, other_years_test=False):
    """
    Records that agree on `keys` (state, year, point, rate, and for the Washington
    Post the name) are copies of one school unless both enrollments are known
    and differ. In such a group one record is kept per distinct known
    enrollment (the first); records with a missing enrollment are dropped when
    the group has any known enrollment, and otherwise one of them is kept. A
    record without a rate is never checked. With `other_years_test`, a record
    at a point that holds two or more schools in another school year is kept
    too (the Washington Post has one school year and checks the name instead).
    """
    known = pl.col("enrollment").is_not_null()
    ordered = df.sort("rid").with_columns(
        first_of_enrollment=pl.int_range(pl.len()).over(keys + ["enrollment"])
        == 0,
        group_has_known=known.any().over(keys),
    )
    keep = pl.col("rate").is_null() | (
        pl.col("first_of_enrollment") & (known | ~pl.col("group_has_known"))
    )
    if other_years_test:
        ordered = (
            ordered.join(
                _points_with_two_schools_in_another_year(df),
                on=["state", "year", "lon", "lat"],
                how="left",
            )
            .sort("rid")
            .with_columns(
                confirmed_distinct=pl.col("confirmed_distinct").fill_null(
                    False
                )
            )
        )
        kept_back = ordered.filter(~keep & pl.col("confirmed_distinct"))
        log.add(
            kept_back,
            "duplicates",
            "flagged",
            reason
            + "), kept: another school year shows two or more schools at this point",
        )
        keep = keep | pl.col("confirmed_distinct")
    log.add(
        ordered.filter(~keep & known),
        "duplicates",
        "dropped",
        reason + ", same enrollment)",
    )
    log.add(
        ordered.filter(~keep & ~known),
        "duplicates",
        "dropped",
        reason + ", enrollment missing)",
    )
    drop = ["first_of_enrollment", "group_has_known"] + (
        ["confirmed_distinct"] if other_years_test else []
    )
    return ordered.filter(keep).drop(drop)


def drop_duplicates_chen_bento(df, log):
    keys = ["state", "year", "lon", "lat", "rate"]
    return _drop_same_point_and_rate(
        df,
        keys,
        "identical Chen & Bento record (same point and rate",
        log,
        other_years_test=True,
    )


def drop_fallback_points(df, min_schools, min_share, log):
    """
    A point with more than `min_schools` records that are at least `min_share`
    of the records of its county and school year is a fallback coordinate (a
    placeholder such as a county centre), not the place of those schools: every
    record on it is dropped. Every record counts, with or without a rate.
    """
    at_point = pl.len().over("state", "year", "county_fips", "lon", "lat")
    in_county_year = pl.len().over("state", "year", "county_fips")
    fallback = (at_point > min_schools) & (
        at_point >= min_share * in_county_year
    )
    log.add(
        df.filter(fallback),
        "fallback point",
        "dropped",
        f"more than {min_schools} records at one point, at least {min_share:.0%} of the county-year's records",
    )
    return df.filter(~fallback)


def count_schools_at_point(df):
    return df.with_columns(
        n_at_point=pl.len().over("state", "year", "lon", "lat")
    )


# --- 5. copied years -------------------------------------------------------------------


def flag_copied_years(df, unchanged_share, min_schools):
    """
    A school year is a copy of the year before when most of the schools that are
    below 100% in both years have exactly the same rate (schools at 100% stay
    there in an ordinary year). The same school is the one at the same point;
    schools sharing a point are left out. The rows stay; the later year of the
    pair gets copied_year = true. Returns the frame and the check of every
    pair of consecutive years.
    """
    alone = (
        df.filter(pl.col("rate").is_not_null())
        .with_columns(
            alone=pl.len().over("state", "year_start", "lon", "lat") == 1
        )
        .filter(pl.col("alone"))
    )
    rows = []
    for state in sorted(alone["state"].unique().to_list()):
        years = sorted(
            alone.filter(pl.col("state") == state)["year_start"]
            .unique()
            .to_list()
        )
        for first, second in zip(years[:-1], years[1:]):
            if second - first != 1:
                continue
            a = alone.filter(
                (pl.col("state") == state) & (pl.col("year_start") == first)
            )
            b = alone.filter(
                (pl.col("state") == state) & (pl.col("year_start") == second)
            )
            both = a.select("lon", "lat", before="rate").join(
                b.select("lon", "lat", after="rate"), on=["lon", "lat"]
            )
            below = both.filter(
                (pl.col("before") < 100) & (pl.col("after") < 100)
            )
            share = (
                (below["before"] == below["after"]).mean()
                if below.height
                else None
            )
            rows.append(
                {
                    "state": state,
                    "year_start_first": first,
                    "year_start_second": second,
                    "schools_matched": both.height,
                    "schools_below_100": below.height,
                    "share_unchanged_below_100": share,
                    "median_change": (both["after"] - both["before"]).median()
                    if both.height
                    else None,
                    "copied_year": below.height >= min_schools
                    and share is not None
                    and share >= unchanged_share,
                }
            )
    checks = pl.DataFrame(
        rows,
        schema_overrides={
            "share_unchanged_below_100": pl.Float64,
            "median_change": pl.Float64,
        },
    )
    copied = (
        checks.filter(pl.col("copied_year"))
        .select("state", year_start="year_start_second")
        .with_columns(copied_year=pl.lit(True))
    )
    df = (
        df.drop("copied_year")
        .join(
            copied,
            on=["state", "year_start"],
            how="left",
            maintain_order="left",
        )
        .with_columns(copied_year=pl.col("copied_year").fill_null(False))
    )
    return df, checks


# --- 6. source selection, borrowed enrollment, inherited types ------------------------


def wapo_years(wapo):
    """The one school year of each Washington Post state."""
    years = wapo.group_by("state").agg(years=pl.col("year_start").unique())
    if (years["years"].list.len() != 1).any():
        raise ValueError(
            "a Washington Post state has more than one school year"
        )
    return dict(zip(years["state"], years["years"].list.first()))


def choose_sources(cb, wapo, state_files, min_share):
    """
    The source of each state and school year (see the module docstring): the
    Washington Post if it has rated schools there and, where Chen & Bento has
    some too, at least `min_share` of Chen & Bento's number (a file with fewer
    schools is partial and does not replace the complete one); otherwise
    Chen & Bento; otherwise the state files. Counts are the rated
    schools. A state file is used only if it is not partial either: it needs at
    least `min_share` of the rated schools of the same state in the nearest
    school year that Chen & Bento or the Washington Post has (a state with no
    such year has nothing to compare with, and its state files are used);
    otherwise the state-year has no source (source empty, with the reason).
    One row per state-year with at least one rated school:
    state, year_start, source, n_washington_post, n_chen_bento, n_state_file,
    and why (empty when the first choice is used).
    """
    counts = {}
    for name, df in [
        (WAPO, wapo),
        (CHEN_BENTO, cb),
        (STATE_FILE, state_files),
    ]:
        for state, year, n in (
            df.filter(pl.col("rate").is_not_null())
            .group_by("state", "year_start")
            .agg(n=pl.len())
            .iter_rows()
        ):
            counts.setdefault((state, year), {})[name] = n
    other_years = {}  # state: {school year: the larger of the two other sources' counts}
    for (state, year), n in counts.items():
        size = max(n.get(WAPO, 0), n.get(CHEN_BENTO, 0))
        if size:
            other_years.setdefault(state, {})[year] = size
    rows = []
    for (state, year), n in sorted(counts.items()):
        n_wapo, n_cb, n_sf = (
            n.get(k, 0) for k in (WAPO, CHEN_BENTO, STATE_FILE)
        )
        why = None
        if n_wapo and n_cb and n_wapo < min_share * n_cb:
            source, why = (
                CHEN_BENTO,
                f"the Washington Post has {n_wapo} schools, under {min_share:.0%} of Chen & Bento's {n_cb}",
            )
        elif n_wapo:
            source = WAPO
        elif n_cb:
            source = CHEN_BENTO
        else:
            source = STATE_FILE
            known = other_years.get(state)
            if known:
                nearest = min(known, key=lambda y: (abs(y - year), y))
                if n_sf < min_share * known[nearest]:
                    source = None
                    why = (
                        f"the state file has {n_sf} schools, under {min_share:.0%} "
                        f"of the {known[nearest]} of {nearest} in another source"
                    )
        rows.append(
            {
                "state": state,
                "year_start": year,
                "source": source,
                "n_washington_post": n_wapo,
                "n_chen_bento": n_cb,
                "n_state_file": n_sf,
                "why": why,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "year_start": pl.Int64,
            "source": pl.String,
            "n_washington_post": pl.Int64,
            "n_chen_bento": pl.Int64,
            "n_state_file": pl.Int64,
            "why": pl.String,
        },
    ).sort("state", "year_start")


def select_sources(cb, wapo, state_files, cfg):
    """
    The schools of each state and school year from its one source
    (choose_sources). Returns the schools and the choice table.
    """
    choice = choose_sources(
        cb,
        wapo,
        state_files,
        cfg["wapo_min_share_of_chen_bento"],
    )
    keys = ["state", "year_start"]
    parts = [
        df.join(
            choice.filter(pl.col("source") == name).select(keys),
            on=keys,
            how="semi",
        )
        for name, df in [
            (WAPO, wapo),
            (CHEN_BENTO, cb),
            (STATE_FILE, state_files),
        ]
    ]
    selected = pl.concat(parts, how="diagonal_relaxed")
    # every county and school year has the schools of ONE source: the files
    # and the county means below are built on it
    two_sources = (
        selected.filter(pl.col("county_fips").is_not_null())
        .group_by("state", "county_fips", "year_start")
        .agg(n_sources=pl.col("source").n_unique())
        .filter(pl.col("n_sources") > 1)
    )
    assert two_sources.height == 0, (
        f"{two_sources.height} county-years have schools of two sources, "
        f"e.g. {two_sources.head(3).rows()}"
    )
    return selected, choice


def borrow_enrollment(selected, cb, cfg):
    """
    A Washington Post school without enrollment takes the enrollment of the
    same school in Chen & Bento (same state and school year): the only
    candidate within match_radius_m in both directions, with a rate within
    rate_tolerance_points (a rate that differs means another school, or another
    measure). Nothing is borrowed where Chen & Bento has no school of that state
    and school year. Returns the schools and, for each state and school year of the
    Washington Post file, how many schools lacked enrollment, how many took it,
    and the counties that can be weighted before and after.
    """
    radius, tolerance = cfg["match_radius_m"], cfg["rate_tolerance_points"]
    selected = selected.with_columns(rid_sel=pl.int_range(pl.len()))
    fills = []
    report = []
    wapo_rows = selected.filter(
        (pl.col("source") == WAPO) & pl.col("rate").is_not_null()
    )
    for (state, year), part in wapo_rows.group_by(
        ["state", "year_start"], maintain_order=True
    ):
        part = part.sort("rid")
        donors = cb.filter(
            (pl.col("state") == state)
            & (pl.col("year_start") == year)
            & (pl.col("enrollment") > 0)
            & pl.col("rate").is_not_null()
        ).sort("rid")
        i, j = unambiguous_pairs(part, donors, radius)
        matched = 0
        if len(i):
            pairs = pl.DataFrame(
                {
                    "rid_sel": part["rid_sel"].to_numpy()[i],
                    "borrowed": donors["enrollment"].to_numpy()[j],
                    "gap": np.abs(
                        part["rate"].to_numpy()[i]
                        - donors["rate"].to_numpy()[j]
                    ),
                }
            ).filter(pl.col("gap") <= tolerance + 1e-9)
            fills.append(pairs.select("rid_sel", "borrowed"))
            matched = pairs.height
        report.append(
            {
                "state": state,
                "year_start": year,
                "schools": part.height,
                "without_enrollment": part.filter(
                    pl.col("enrollment").is_null()
                ).height,
                "matched_same_rate": matched,
            }
        )
    before = _weightable_counties(wapo_rows)
    if fills:
        selected = selected.join(pl.concat(fills), on="rid_sel", how="left")
        take = (
            pl.col("enrollment").is_null() & pl.col("borrowed").is_not_null()
        )
        selected = selected.with_columns(
            enrollment_source=pl.when(take)
            .then(pl.lit(CHEN_BENTO))
            .otherwise(pl.col("enrollment_source")),
            enrollment=pl.when(take)
            .then(pl.col("borrowed"))
            .otherwise(pl.col("enrollment")),
        ).drop("borrowed")
    after = _weightable_counties(
        selected.filter(
            (pl.col("source") == WAPO) & pl.col("rate").is_not_null()
        )
    )
    report = (
        pl.DataFrame(
            report,
            schema={
                "state": pl.String,
                "year_start": pl.Int64,
                "schools": pl.Int64,
                "without_enrollment": pl.Int64,
                "matched_same_rate": pl.Int64,
            },
        )
        .join(
            before.select(
                "state",
                "year_start",
                "counties",
                counties_weightable_before="counties_weightable",
            ),
            on=["state", "year_start"],
            how="left",
        )
        .join(
            after.select(
                "state",
                "year_start",
                counties_weightable_after="counties_weightable",
            ),
            on=["state", "year_start"],
            how="left",
        )
    )
    return selected.drop("rid_sel"), report.sort("state", "year_start")


def _weightable_counties(df):
    """Per state and school year: the counties, and those whose schools all have enrollment."""
    return (
        df.group_by("state", "year_start", "county_fips")
        .agg(all_known=(pl.col("enrollment") > 0).fill_null(False).all())
        .group_by("state", "year_start")
        .agg(counties=pl.len(), counties_weightable=pl.col("all_known").sum())
    )


def inherit_school_types(selected, wapo, cfg):
    """
    A Chen & Bento school takes the reported type of the Washington Post school
    at its point (the only candidate within match_radius_m in both
    directions). Chen & Bento has no types of its own. A point with more than
    one Chen & Bento school in any school year is left out of the matching: the
    Washington Post record cannot be told to belong to one of them.
    """
    radius = cfg["match_radius_m"]
    schools = (
        selected.filter(pl.col("source") == CHEN_BENTO)
        .group_by("state", "lon", "lat")
        .agg(shared=(pl.col("n_at_point") > 1).fill_null(False).any())
        .filter(~pl.col("shared"))
        .select("state", "lon", "lat")
        .sort("state", "lon", "lat")
    )
    inherited = []
    for state in sorted(schools["state"].unique().to_list()):
        points = schools.filter(pl.col("state") == state)
        donors = wapo.filter(pl.col("state") == state).sort("rid")
        i, j = unambiguous_pairs(points, donors, radius)
        if len(i) == 0:
            continue
        found = pl.DataFrame(
            {
                "state": state,
                "lon": points["lon"].to_numpy()[i],
                "lat": points["lat"].to_numpy()[i],
                "inherited_type": donors["school_type"][j.tolist()],
            }
        ).filter(pl.col("inherited_type").is_not_null())
        inherited.append(found)
    if not inherited:
        return selected
    found = pl.concat(inherited)
    selected = selected.join(found, on=["state", "lon", "lat"], how="left")
    take = (
        (pl.col("source") == CHEN_BENTO)
        & pl.col("school_type").is_null()
        & pl.col("inherited_type").is_not_null()
    )
    return selected.with_columns(
        school_type=pl.when(take)
        .then(pl.col("inherited_type"))
        .otherwise(pl.col("school_type")),
        school_type_source=pl.when(take)
        .then(pl.lit("inherited_washington_post"))
        .otherwise(pl.col("school_type_source")),
    ).drop("inherited_type")


# --- 7. flags and the two files -------------------------------------------------------


def add_flags(df, cfg):
    """
    unit, grade_scope, enrollment_basis, and the low-rate flags. The enrollment
    basis is that of the source the enrollment came from (a Washington Post
    school with borrowed enrollment takes Chen & Bento's).
    """
    basis = []
    for source, groups in cfg["enrollment_basis"].items():
        for name, states in groups.items():
            basis += [
                {
                    "enrollment_source": source,
                    "state": s,
                    "enrollment_basis": name,
                }
                for s in states
            ]
    basis = pl.DataFrame(
        basis,
        schema={
            "enrollment_source": pl.String,
            "state": pl.String,
            "enrollment_basis": pl.String,
        },
    )
    low = cfg["low_rate"]
    known = (pl.col("enrollment") > 0).fill_null(False)
    median = pl.col("rate").median().over("source", "state", "year_start")
    flags = [
        pl.when((pl.col("rate") < cfg["placeholder_rate_below"]) & known).then(
            pl.lit("below_placeholder_rate")
        ),
        pl.when(
            (pl.col("rate") < low["below"])
            & (pl.col("enrollment") >= low["below_min_enrollment"])
            & (pl.col("enrollment_basis") == "denominator")
        ).then(pl.lit("low_in_large_school")),
        pl.when(
            (pl.col("rate") < median - low["far_below_median_points"])
            & (pl.col("enrollment") >= low["far_below_median_min_enrollment"])
        ).then(pl.lit("far_below_state_median")),
    ]
    scope = [
        pl.when(pl.col("state").is_in(cfg["secondary_included_states"])).then(
            pl.lit("secondary_included")
        ),
        pl.when(
            pl.col("state").is_in(cfg["all_grade_enrollment_states"])
        ).then(pl.lit("all_grade_enrollment")),
    ]
    df = df.join(
        basis, on=["enrollment_source", "state"], how="left"
    ).with_columns(
        enrollment_basis=pl.col("enrollment_basis").fill_null("unknown"),
        unit=unit_of_row(cfg["chen_bento_district_states"]),
        grade_scope=pl.concat_str(scope, separator=";", ignore_nulls=True),
    )
    return df.with_columns(
        low_rate_flag=pl.concat_str(flags, separator=";", ignore_nulls=True),
        grade_scope=pl.when(pl.col("grade_scope") == "")
        .then(None)
        .otherwise(pl.col("grade_scope")),
    ).with_columns(
        low_rate_flag=pl.when(pl.col("low_rate_flag") == "")
        .then(None)
        .otherwise(pl.col("low_rate_flag"))
    )


SCHOOL_FILE_COLUMNS = [
    "state", "year", "year_start", "year_end", "source", "school_name", "address",
    "school_type", "school_type_source", "unit", "enrollment", "enrollment_source",
    "enrollment_basis", "grade_scope", "mmr_rate", "rate_source",
    "exemption_medical", "exemption_religious", "exemption_personal",
    "exemption_nonmedical", "exemption_overall", "lon", "lat", "county_fips",
    "county_name", "county_join", "county_label_differs", "copied_year",
    "low_rate_flag", "n_at_point",
]  # fmt: skip


def school_file(df):
    """The school-level file: the schools that have a rate, in a fixed order."""
    # the order is total only if a school (rid) is in the data once per source
    assert not df.select("source", "rid").is_duplicated().any(), (
        "a school appears twice in a source"
    )
    return (
        df.filter(pl.col("rate").is_not_null())
        .rename({"rate": "mmr_rate"})
        .sort(
            "state",
            "year_start",
            "county_fips",
            "school_name",
            "lon",
            "lat",
            "enrollment",
            "mmr_rate",
            "source",
            "rid",
            nulls_last=True,
        )
        .select(SCHOOL_FILE_COLUMNS)
    )


def add_county_mmr_qc(county, county_mmr, max_gap_pp):
    """
    Adds, to the county file, the model's county MMR of the same school year
    (county_mmr: fips, year, estimated_mmr_cov and is_imputed, as in
    data/processed_vaccinations_data.csv, which labels a school year by its
    start year, year_start), county_mmr_imputed (whether that value is
    imputed), mean_gap_pp (the county MMR minus the unweighted mean school
    rate, percentage points) and qc_gap_ok (the gap is at most max_gap_pp in
    absolute value). An imputed county MMR is not used: county_mmr, mean_gap_pp
    and qc_gap_ok are then empty, and so they are for a county-year that has
    no county MMR; an empty qc_gap_ok is inconclusive (neither ok nor not ok).
    The rows of the county file keep their order.
    """
    assert county_mmr.select("fips", "year").is_duplicated().sum() == 0
    return (
        county.join(
            county_mmr.select(
                pl.col("fips").alias("county_fips"),
                pl.col("year").alias("year_start"),
                pl.col("estimated_mmr_cov").alias("reported_county_mmr"),
                pl.col("is_imputed").alias("county_mmr_imputed"),
            ),
            on=["county_fips", "year_start"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            county_mmr=pl.when(~pl.col("county_mmr_imputed")).then(
                pl.col("reported_county_mmr")
            )
        )
        .drop("reported_county_mmr")
        .with_columns(
            mean_gap_pp=pl.col("county_mmr") - pl.col("mean_unweighted")
        )
        .with_columns(qc_gap_ok=pl.col("mean_gap_pp").abs() <= max_gap_pp)
    )


def county_file(df):
    """
    One row per county and school year, from the schools with a rate.
    Weighted values (weights are enrollments) need every school of the county
    to have enrollment above 0; the weighted SD is the SD of the rate over
    students, sqrt(sum(w (x - mean)^2) / sum(w)); the unweighted SD is that of
    the rates of the schools, sqrt(sum((x - mean)^2) / n). Both are population
    SDs (the schools of a county are all of its schools, not a sample of them),
    so a county of one school has an SD of 0. The accumulated values are the
    weighted ones when they exist and the unweighted ones otherwise.
    fraction_public is the public share of the schools, fraction_kids_public
    the public share of the enrollment; both need every school to be public or
    private (and, for the second, to have enrollment). Shares are 0-1.
    enrollment_basis is the schools' common basis, or "mixed" when the schools
    of the county differ (e.g. reported and borrowed enrollment).
    unit_type is "school", "district" or "mixed" when the rows of the county
    say which they are (Washington Post: by name, ISD or school district;
    Chen & Bento: by state, params.yml chen_bento_district_states), else
    empty; n_districts counts the district rows. share_overall_rate is the share of the rows whose
    rate is the overall rate, not the MMR rate (Washington Post and state files; Chen &
    Bento's rows do not say, so the share is empty for them).
    """
    keys = ["state", "county_fips", "year", "year_start", "year_end", "source"]
    assert not df.select("source", "rid").is_duplicated().any(), (
        "a school appears twice in a source"
    )
    # sorted, so that the sums of a county are taken in the same order every time
    df = df.sort(
        keys + ["school_name", "lon", "lat", "enrollment", "rate", "rid"],
        nulls_last=True,
    )
    rated = df.filter(pl.col("rate").is_not_null())
    known = (pl.col("enrollment") > 0).fill_null(False)
    typed = pl.col("school_type").is_in(SCHOOL_TYPES).fill_null(False)
    public = (pl.col("school_type") == "PUBLIC").fill_null(False)
    rate, weight = pl.col("rate"), pl.col("enrollment")
    mean_w = (rate * weight).sum() / weight.sum()
    n = pl.len()
    table = rated.group_by(
        keys
    ).agg(
        county_name=pl.col("county_name").first(),
        n_schools=n,
        n_schools_with_enrollment=known.sum(),
        total_enrollment=weight.filter(known).sum(),
        n_schools_with_type=typed.sum(),
        fraction_public=pl.when(typed.all()).then(public.sum() / n),
        fraction_kids_public=pl.when(typed.all() & known.all()).then(
            weight.filter(public).sum() / weight.sum()
        ),
        mean_unweighted=rate.mean(),
        sd_unweighted=rate.std(ddof=0),
        mean_weighted=pl.when(known.all()).then(mean_w),
        sd_weighted=pl.when(known.all()).then(
            pl.when(n == 1)
            .then(0.0)
            .otherwise(
                (((rate - mean_w) ** 2 * weight).sum() / weight.sum()).sqrt()
            )
        ),  # one school: exactly 0 (not the rounding error of the weighted mean)
        enrollment_basis=pl.when(pl.col("enrollment_basis").n_unique() == 1)
        .then(pl.col("enrollment_basis").first())
        .otherwise(pl.lit("mixed")),
        grade_scope=pl.col("grade_scope").first(),
        copied_year=pl.col("copied_year").any(),
        unit_type=pl.when(pl.col("unit").is_not_null().all()).then(
            pl.when((pl.col("unit") == "district").all())
            .then(pl.lit("district"))
            .when((pl.col("unit") == "school").all())
            .then(pl.lit("school"))
            .otherwise(pl.lit("mixed"))
        ),
        n_districts=pl.when(pl.col("unit").is_not_null().all()).then(
            (pl.col("unit") == "district").sum()
        ),
        share_overall_rate=pl.when(
            pl.col("rate_source").is_in(["mmr", "overall"]).all()
        ).then((pl.col("rate_source") == "overall").mean()),
    )
    table = table.with_columns(
        total_enrollment=pl.when(pl.col("n_schools_with_enrollment") > 0).then(
            pl.col("total_enrollment")
        ),
        accumulated_source=pl.when(pl.col("mean_weighted").is_not_null())
        .then(pl.lit("weighted"))
        .otherwise(pl.lit("unweighted")),
        mean_accumulated=pl.coalesce("mean_weighted", "mean_unweighted"),
        sd_accumulated=pl.when(pl.col("mean_weighted").is_not_null())
        .then(pl.col("sd_weighted"))
        .otherwise(pl.col("sd_unweighted")),
    )
    totals = df.group_by(keys).agg(n_schools_all=pl.len())
    return (
        table.join(totals, on=keys, how="left")
        .select(
            "state",
            "county_fips",
            "county_name",
            "year",
            "year_start",
            "year_end",
            "source",
            "n_schools",
            "n_schools_all",
            "n_schools_with_enrollment",
            "total_enrollment",
            "n_schools_with_type",
            "fraction_public",
            "fraction_kids_public",
            "mean_unweighted",
            "sd_unweighted",
            "mean_weighted",
            "sd_weighted",
            "mean_accumulated",
            "sd_accumulated",
            "accumulated_source",
            "enrollment_basis",
            "grade_scope",
            "copied_year",
            "unit_type",
            "n_districts",
            "share_overall_rate",
        )  # fmt: skip
        .sort("state", "county_fips", "year_start")
    )


# --- diagnostics -------------------------------------------------------------------------


def enrollment_denominator_test(df):
    """
    Is the enrollment the denominator of the rate? For schools of 1-200
    students whose rate is not a whole percent, rate x enrollment should be a
    whole number of students. A state is 'denominator' when 85% or more of its
    schools pass, 'not_denominator' below 50%, 'ambiguous' between; states with
    fewer than 30 testable schools are left out (their rates are whole
    percents, or they lack enrollment).
    """
    tested = df.filter(
        (pl.col("enrollment") >= 1)
        & (pl.col("enrollment") <= 200)
        & pl.col("rate").is_not_null()
        & ((pl.col("rate") - pl.col("rate").round(0)).abs() > 1e-6)
    )
    students = tested["rate"] / 100 * tested["enrollment"]
    tested = tested.with_columns(
        passes=(
            (students - students.round(0)).abs()
            < 0.02 + 0.0005 * tested["enrollment"]
        )
    )
    return (
        tested.group_by("source", "state")
        .agg(
            schools_tested=pl.len(),
            share_whole_students=pl.col("passes").mean(),
        )
        .filter(pl.col("schools_tested") >= 30)
        .with_columns(
            enrollment_basis=pl.when(pl.col("share_whole_students") >= 0.85)
            .then(pl.lit("denominator"))
            .when(pl.col("share_whole_students") < 0.5)
            .then(pl.lit("not_denominator"))
            .otherwise(pl.lit("ambiguous"))
        )
        .sort("source", "state")
    )


def seam_diagnostics(cb, wapo, choice):
    """
    How the Washington Post year joins Chen & Bento's series, county means (all
    schools, unweighted), for the states that are stitched: the change from
    Chen & Bento's year before (and, where Chen & Bento continues, the change
    to its year after), against the typical change between two Chen &
    Bento years of the state (the mean absolute change and the correlation
    across counties; years that copy the year before are left out), and the
    schools and counties gained and lost.
    """
    # only the states whose Washington Post year is used join its series to Chen & Bento's
    used = set(choice.filter(pl.col("source") == WAPO)["state"])
    year_of = wapo_years(wapo)
    means = lambda d: (
        d.filter(pl.col("rate").is_not_null())
        .group_by("state", "county_fips", "year_start")
        .agg(mean=pl.col("rate").mean(), schools=pl.len())
    )
    cb_means, wapo_means = means(cb), means(wapo)
    copied = set(
        cb.filter(pl.col("copied_year"))
        .select("state", "year_start")
        .unique()
        .iter_rows()
    )

    def change(a, b):
        joined = a.join(b, on="county_fips", suffix="_b").sort("county_fips")
        if joined.height < 8:
            return None
        diff = joined["mean_b"] - joined["mean"]
        return {
            "counties": joined.height,
            "mean_change": diff.mean(),
            "mean_abs_change": diff.abs().mean(),
            "r": joined.select(pl.corr("mean", "mean_b")).item(),
        }

    rows = []
    for state in sorted(set(cb["state"]) & set(wapo["state"]) & used):
        year = year_of[state]
        years = sorted(
            cb_means.filter(pl.col("state") == state)["year_start"]
            .unique()
            .to_list()
        )
        reference = (
            year - 1
            if (year - 1) in years
            else (year if year in years else None)
        )
        if reference is None:
            continue
        cb_ref = cb_means.filter(
            (pl.col("state") == state) & (pl.col("year_start") == reference)
        )
        wapo_year = wapo_means.filter(pl.col("state") == state)
        seam = change(cb_ref, wapo_year)
        following = None
        if (year + 1) in years:
            following = change(
                wapo_year,
                cb_means.filter(
                    (pl.col("state") == state)
                    & (pl.col("year_start") == year + 1)
                ),
            )
        typical = []
        for first, second in zip(years[:-1], years[1:]):
            if (
                second - first == 1
                and (state, second) not in copied
                and (second <= year - 1 or first >= year + 1)
            ):
                result = change(
                    cb_means.filter(
                        (pl.col("state") == state)
                        & (pl.col("year_start") == first)
                    ),
                    cb_means.filter(
                        (pl.col("state") == state)
                        & (pl.col("year_start") == second)
                    ),
                )
                if result and result["mean_abs_change"] >= 0.05:
                    typical.append(result)
        counties_cb, counties_wapo = (
            set(cb_ref["county_fips"]),
            set(wapo_year["county_fips"]),
        )
        rows.append(
            {
                "state": state,
                "washington_post_year_start": year,
                "compared_with_year_start": reference,
                "schools_washington_post": wapo_year["schools"].sum(),
                "schools_chen_bento": cb_ref["schools"].sum(),
                "counties_only_in_chen_bento": len(
                    counties_cb - counties_wapo
                ),
                "counties_only_in_washington_post": len(
                    counties_wapo - counties_cb
                ),
                "seam_counties": seam["counties"] if seam else None,
                "seam_mean_change": seam["mean_change"] if seam else None,
                "seam_mean_abs_change": seam["mean_abs_change"]
                if seam
                else None,
                "seam_r": seam["r"] if seam else None,
                "next_mean_change": following["mean_change"]
                if following
                else None,
                "next_mean_abs_change": following["mean_abs_change"]
                if following
                else None,
                "next_r": following["r"] if following else None,
                "typical_mean_abs_change": float(
                    np.median([t["mean_abs_change"] for t in typical])
                )
                if typical
                else None,
                "typical_r": float(np.median([t["r"] for t in typical]))
                if typical
                else None,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)
