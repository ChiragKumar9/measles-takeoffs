"""
Reading the case-assignment files (data/measles_2026_case_assignments.csv and
data/measles_2025_case_assignments_SC.csv) into the layout of the case sheets
that analysis/00_explore_data.py reads: year, state, county, cases, indigenous,
imported and fips.

The files hold rows of three types. Only `assignment` rows are cases:
`state_control_total` rows are the state's own count (a check, printed in the
report) and `alternate_marginal` rows count the same cases on another date
basis, so they are never added.
"""

import os

import geopandas as gpd
import polars as pl

# The 2026 file names a multi-county unit its own way; these are the keys of
# outbreak_size.regions.higher_level_regions() (state, unit in the file) ->
# the place's name in the case data
UNIT_KEYS = {
    ("Utah", "Southwest Utah"): "Southwest Health District, Utah",
    ("Utah", "Southeast Utah"): "Southeast Health District, Utah",
    ("Utah", "Bear River"): "Bear River, Utah",
    ("Utah", "Central Utah"): "Central Utah, Utah",
    ("Utah", "TriCounty"): "TriCounty, Utah",
    ("Utah", "Weber-Morgan"): "Weber-Morgan, Utah",
    (
        "Minnesota",
        "Twin Cities metro area",
    ): "Twin Cities Metro Area, Minnesota",
    ("Louisiana", "LDH Region 1"): "Region 1, Louisiana",
    ("Tennessee", "Mid-Cumberland Region"): "Mid-Cumberland Region, Tennessee",
    ("Tennessee", "East Tennessee Region"): "East tennessee?",
    ("Iowa", "HHS District 3"): "HHS District 3, Iowa",
    ("Iowa", "HHS District 4"): "HHS District 4, Iowa",
    ("Iowa", "HHS District 5"): "HHS District 5, Iowa",
    ("Iowa", "HHS District 6"): "HHS District 6, Iowa",
    ("Georgia", "Coastal Health District"): "Coastal Health District, Georgia",
    ("Kansas", "Northeast HCC Region"): "Northeast HCC Region, Kansas",
}

# Units whose counties no health department defines (the department reports
# the unit under a name that it never defines), so they cannot be placed;
# their cases are left out and counted in the report
DROPPED_UNITS = {
    ("Georgia", "Metro Atlanta"),
    ("Massachusetts", "Greater Boston"),
    ("Missouri", "Kansas City"),
    # three non-outbreak 2025 cases whose residence the department withheld
    ("South Carolina", "Upstate"),
}

# The county names of the vaccination data, where the case file spells them
# otherwise: (state, name in the case file) -> name in the vaccination data
SPELLING = {
    ("Illinois", "DeKalb"): "Dekalb",
    ("Indiana", "DeKalb"): "Dekalb",
    ("Indiana", "LaGrange"): "Lagrange",
    ("Pennsylvania", "McKean"): "Mckean",
    ("New Mexico", "Doña Ana"): "Dona Ana",
    ("Missouri", "St. Louis County"): "St. Louis",
}

# Arizona 2026: the file assigns 30 of the state's 123 cases (Maricopa, Pima
# and Yavapai). The control-total row's note gives the state's county table:
# "County 2026 values sum to 123: Coconino 4, Maricopa 28, Mohave 69, Pima 3,
# Pinal 18, Yavapai 1" (AZDHS, page last updated 09/22/2026). Mohave, Pinal
# and Coconino have no assignment row at all, so their counts are added here;
# Maricopa and Pima keep the file's rows.
ARIZONA_2026_COUNTIES_WITHOUT_ROWS = (
    ("Arizona", "Mohave", 69),
    ("Arizona", "Pinal", 18),
    ("Arizona", "Coconino", 4),
)


def shapefile_counties(data_dir: str) -> pl.DataFrame:
    """Every county of the 2020 shapefile: fips, county and state name."""
    counties = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    states = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_state_20m.shp")
    ).rename(columns={"NAME": "state"})
    counties = counties[
        ~counties["STATEFP"].isin(["60", "66", "69", "72", "78"])
    ].merge(states[["STATEFP", "state"]], on="STATEFP")
    return pl.DataFrame(
        {
            "fips": counties["GEOID"].tolist(),
            "county": counties["NAME"].tolist(),
            "state": counties["state"].tolist(),
        }
    ).sort("fips")


def read_case_assignments(
    path: str,
    year: int,
    counties: pl.DataFrame,
    extra_counties=(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    The cases of one assignment file, in the layout of the case sheets.

    `counties` has the columns fips, county and state of every county (see
    shapefile_counties); a county gets its fips only if its name is the
    only one of that name in its state (Fairfax and Baltimore are a county
    and an independent city), otherwise its fips is null and the merge with
    the vaccination data gives it one. `extra_counties` is a sequence of
    (state, county, cases) for counties that the file's own rows lack; a
    county that has an assignment row cannot be added (it would be counted
    twice).

    Returns the cases (year, state, county, cases, indigenous, imported,
    fips: one row per place, cases net of the negative revisions) and the
    report by state (control_total, assigned, state_only, dropped_unit,
    extra, cases and unplaced: the cases of the rows the file gives, the
    state's own count, and what the returned cases are made of).
    """
    raw = pl.read_csv(path, infer_schema_length=None)
    assigned = raw.filter(pl.col("row_type") == "assignment")
    control = raw.filter(pl.col("row_type") == "state_control_total")

    if not (assigned["marginal"] == "primary").all():
        raise ValueError(
            f"{path}: an assignment row is not on the primary basis: "
            f"{assigned.filter(pl.col('marginal') != 'primary').height} rows"
        )

    # the file lists New York City as a state of its own
    assigned = assigned.with_columns(
        state=pl.when(pl.col("state") == "New York City")
        .then(pl.lit("New York"))
        .otherwise(pl.col("state")),
        original_state=pl.col("state"),
    )

    is_state_only = (pl.col("geo_unit") == "state") | pl.col(
        "county"
    ).is_null()
    state_only = assigned.filter(is_state_only)
    units = assigned.filter(~is_state_only)

    is_dropped = pl.struct("state", "county").map_elements(
        lambda r: (r["state"], r["county"]) in DROPPED_UNITS,
        return_dtype=pl.Boolean,
    )
    dropped = units.filter(is_dropped)
    # (a filter, not an anti join: it keeps the order of the units, and the
    # order of the units decides which row of a place is its first)
    placed = units.filter(~is_dropped)

    # the multi-county units: each must be known (a unit the code does not
    # know would otherwise be silently lost to the model)
    def place_name(row):
        state, county, geo = row["state"], row["county"], row["geo_unit"]
        if geo in ("health_district", "region"):
            if (state, county) in UNIT_KEYS:
                return UNIT_KEYS[(state, county)]
            # the citywide rows of New York City: "New York City" becomes
            # Kings in process_outbreaks_df
            if (state, county) == ("New York", "New York City"):
                return county
            raise ValueError(
                f"{path}: the {geo} {county!r} of {state} is not a known "
                "unit (outbreak_size.case_assignments.UNIT_KEYS) or a "
                "dropped one (DROPPED_UNITS)"
            )
        return SPELLING.get((state, county), county)

    placed = placed.with_columns(
        original_county=pl.col("county"),
        place=pl.struct("state", "county", "geo_unit").map_elements(
            place_name, return_dtype=pl.String
        ),
    )

    # one row per place; negative revisions are netted, not clamped
    by_place = (
        placed.group_by("state", "place", maintain_order=True)
        .agg(
            cases=pl.col("case_count").sum().round(0).cast(pl.Int64),
            original_county=pl.col("original_county").first(),
            geo_unit=pl.col("geo_unit").first(),
        )
        .rename({"place": "county"})
    )
    negative = by_place.filter(pl.col("cases") < 0)
    if negative.height:
        raise ValueError(
            f"{path}: places whose cases net below zero: "
            f"{negative.select('state', 'county', 'cases').rows()}"
        )

    # counties the file's own rows lack
    extra = []
    for state, county, n in extra_counties:
        if by_place.filter(
            (pl.col("state") == state) & (pl.col("county") == county)
        ).height:
            raise ValueError(
                f"{state} {county} has an assignment row: adding {n} cases "
                "would count them twice"
            )
        extra.append((state, county, int(n), county, "county"))
    if extra:
        by_place = pl.concat(
            [
                by_place,
                pl.DataFrame(
                    extra,
                    schema=by_place.schema,
                    orient="row",
                ),
            ]
        )

    # the fips of a county whose name is the only one of its state
    unique_names = (
        counties.group_by("state", "county")
        .agg(fips=pl.col("fips"), n=pl.len())
        .filter(pl.col("n") == 1)
        .select("state", "county", fips=pl.col("fips").list.first())
    )
    out = (
        by_place.join(
            unique_names.rename({"county": "original_county"}),
            on=["state", "original_county"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            fips=pl.when(pl.col("geo_unit") == "county")
            .then(pl.col("fips"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .cast(pl.String)
        )
        .select(
            year=pl.lit(year, dtype=pl.Int64),
            state=pl.col("state"),
            county=pl.col("county"),
            cases=pl.col("cases"),
            indigenous=pl.lit(None, dtype=pl.Int64),
            imported=pl.lit(None, dtype=pl.Int64),
            fips=pl.col("fips"),
        )
    )

    report = _report(control, assigned, state_only, dropped, extra, out)
    # every case of the file is in the output or in a counted drop
    total_in = assigned["case_count"].sum()
    accounted = (
        out["cases"].sum()
        - sum(n for _, _, n, _, _ in extra)
        + state_only["case_count"].sum()
        + dropped["case_count"].sum()
    )
    if abs(total_in - accounted) > 1e-9:
        raise ValueError(
            f"{path}: {total_in} assigned cases, but the output and the "
            f"counted drops make {accounted}"
        )
    return out, report


def _report(
    control, assigned, state_only, dropped, extra, out
) -> pl.DataFrame:
    """The cases of each state: the state's own count, the file's rows, and
    what became of them."""
    state_total = lambda frame, column="case_count": (  # noqa: E731
        frame.group_by("state").agg(pl.col(column).sum())
    )
    parts = {
        "control_total": state_total(control).rename(
            {"case_count": "control_total"}
        ),
        "assigned": state_total(assigned).rename({"case_count": "assigned"}),
        "state_only": state_total(state_only).rename(
            {"case_count": "state_only"}
        ),
        "dropped_unit": state_total(dropped).rename(
            {"case_count": "dropped_unit"}
        ),
        "extra": pl.DataFrame(
            [(s, float(n)) for s, _, n, _, _ in extra],
            schema={"state": pl.String, "extra": pl.Float64},
            orient="row",
        )
        .group_by("state")
        .agg(pl.col("extra").sum()),
        "cases": out.group_by("state").agg(
            pl.col("cases").sum().cast(pl.Float64)
        ),
        "unplaced": out.filter(pl.col("fips").is_null())
        .group_by("state")
        .agg(pl.col("cases").sum().cast(pl.Float64).alias("unplaced")),
    }
    report = None
    for frame in parts.values():
        frame = frame.with_columns(pl.col(frame.columns[1]).cast(pl.Float64))
        report = (
            frame
            if report is None
            else report.join(frame, on="state", how="full", coalesce=True)
        )
    assert report is not None
    return report.fill_null(0.0).sort("state")
