"""
Enrollment-weighted SD of the school MMR rates of a county, with the
kindergarten counts of the Common Core of Data (CCD, analysis/
read_ccd_enrollment.py) as the weights.

A school row of our school file is matched to a CCD school (the same
normalized name in the same county, or the nearest CCD school within
`radius_m`, the rule of params.yml match_radius_m) or, for a district row (a
Texas ISD), to the CCD district of the same name that has schools in the
county. Its CCD weight is:
  - the kindergarten count of the matched school or district;
  - 0 for a matched school that does not offer kindergarten (its rate is not
    a kindergarten rate: it leaves the kindergarten SD);
  - missing for a matched school that offers kindergarten but has no count,
    and for a row with no match.

Two county SDs, both the population SD of the rates weighted by the weights
(an SD of exactly 0 for one school):
  sd_ccd        CCD weights only. The county is weighted when the share of its
                rows with no CCD match is at most `max_unmatched_share` and no
                matched row has a missing weight; the rows with no match leave
                the SD. Otherwise None.
  sd_hierarchy  school by school: the CCD weight of a row, else its own
                enrollment (the school file's, from the Washington Post, Chen &
                Bento or a state file) when that is known. The county is
                weighted when every row has a weight; otherwise None.
"""

import re

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

MATCHED = ["name", "name+xy", "xy", "district_name"]
_STOP = {"school", "schools", "the", "of", "and", "at", "sch"}
_EARTH_RADIUS_M = 6371000.0
_PROJECTION_LATITUDE = 38.0  # equirectangular: good to ~15% at +-12 degrees


def normalize_name(name):
    """Lower case words of a name without punctuation and filler words."""
    if name is None:
        return None
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return " ".join(word for word in words if word not in _STOP)


def _project(lon, lat):
    """Longitude and latitude (degrees) as meters on a plane."""
    scale = np.cos(np.radians(_PROJECTION_LATITUDE))
    return np.column_stack(
        [
            np.radians(lon) * scale * _EARTH_RADIUS_M,
            np.radians(lat) * _EARTH_RADIUS_M,
        ]
    )


def weighted_sd(rate, weight):
    """
    The population SD of `rate` weighted by `weight` over the rows with a
    weight above 0; None when there is none, and exactly 0 for one row.
    """
    rate = np.asarray(rate, dtype=float)
    weight = np.asarray(weight, dtype=float)
    keep = weight > 0
    if not keep.any():
        return None
    rate, weight = rate[keep], weight[keep]
    if len(rate) == 1:
        return 0.0
    mean = (weight * rate).sum() / weight.sum()
    return float(np.sqrt((weight * (rate - mean) ** 2).sum() / weight.sum()))


def match_to_ccd(rows, schools, districts, radius_m=100.0):
    """
    `rows`: school rows with state, county_fips, school_name, unit ("school"
    or "district"), lon and lat. `schools` and `districts` are the files of
    analysis/read_ccd_enrollment.py. Returns the rows with match ("name",
    "name+xy", "xy", "district_name", "name_ambiguous" or "none"), ncessch,
    leaid and ccd_weight (see the module docstring).
    """
    rows = rows.with_row_index("_row")
    schools = schools.with_columns(
        name_n=pl.col("school_name").map_elements(
            normalize_name, return_dtype=pl.String
        )
    )
    district_names = districts.select(
        "leaid",
        name_n=pl.col("lea_name").map_elements(
            normalize_name, return_dtype=pl.String
        ),
    )
    leas_in_county = (
        schools.select("county_fips", "leaid")
        .unique()
        .join(district_names, on="leaid")
    )
    found = []
    for (state,), part in rows.group_by("state"):
        in_state = schools.filter(pl.col("state") == state)
        tree = cKDTree(
            _project(in_state["lon"].to_numpy(), in_state["lat"].to_numpy())
        )
        by_county = in_state.partition_by("county_fips", as_dict=True)
        for row in part.iter_rows(named=True):
            result = dict(
                _row=row["_row"], match="none", ncessch=None, leaid=None
            )
            name_n = normalize_name(row["school_name"])
            if row["unit"] == "district":
                hit = leas_in_county.filter(
                    (pl.col("county_fips") == row["county_fips"])
                    & (pl.col("name_n") == name_n)
                )
                if name_n and hit.height == 1:
                    result.update(match="district_name", leaid=hit["leaid"][0])
            else:
                county = by_county.get((row["county_fips"],))
                by_name = (
                    county.filter(pl.col("name_n") == name_n)
                    if (county is not None and name_n)
                    else None
                )
                near = None
                if row["lon"] is not None and row["lat"] is not None:
                    distance, index = tree.query(
                        _project(
                            np.array([row["lon"]]), np.array([row["lat"]])
                        ),
                        k=1,
                    )
                    if distance[0] <= radius_m:
                        near = in_state["ncessch"][int(index[0])]
                if by_name is not None and by_name.height == 1:
                    same = near == by_name["ncessch"][0]
                    result.update(
                        match="name+xy" if same else "name",
                        ncessch=by_name["ncessch"][0],
                    )
                elif near is not None:
                    result.update(match="xy", ncessch=near)
                elif by_name is not None and by_name.height > 1:
                    result.update(match="name_ambiguous")
            found.append(result)
    matched = rows.join(
        pl.DataFrame(
            found,
            infer_schema_length=None,
            schema_overrides={
                "_row": pl.UInt32,
                "ncessch": pl.String,
                "leaid": pl.String,
            },
        ),
        on="_row",
    )
    matched = matched.join(
        schools.select(
            "ncessch",
            "kg_offered",
            school_count=pl.col("kg_count").cast(pl.Float64),
        ),
        on="ncessch",
        how="left",
    ).join(
        districts.select(
            "leaid", district_count=pl.col("kg_count").cast(pl.Float64)
        ),
        on="leaid",
        how="left",
    )
    return matched.with_columns(
        ccd_weight=pl.when(pl.col("match") == "district_name")
        .then(pl.col("district_count"))
        .when(pl.col("match").is_in(MATCHED))
        .then(
            pl.when(pl.col("kg_offered") == "No")
            .then(pl.lit(0.0))
            .otherwise(pl.col("school_count"))
        )
    ).sort("_row")


def county_sds(matched, max_unmatched_share=0.0):
    """
    The CCD-only and the hierarchy SD (module docstring) of each county of
    `matched`, the output of match_to_ccd with the columns county_fips,
    mmr_rate and enrollment (our own, null when unknown): county_fips,
    sd_ccd, sd_hierarchy (None where not available) and n_rows.
    """
    table = []
    for (fips,), rows in matched.group_by("county_fips"):
        rate = rows["mmr_rate"].to_numpy()
        ccd = rows["ccd_weight"]
        found = rows["match"].is_in(MATCHED)
        unmatched_share = float((~found).mean())
        missing_matched = bool((found & ccd.is_null()).any())
        sd_ccd = None
        if unmatched_share <= max_unmatched_share and not missing_matched:
            sd_ccd = weighted_sd(rate, ccd.fill_null(0.0).to_numpy())
        # the CCD weight, else our own enrollment, else missing (nan)
        ccd_weight = ccd.cast(pl.Float64).to_numpy()
        own = rows["enrollment"].cast(pl.Float64).fill_null(0.0).to_numpy()
        weight = np.where(
            np.isnan(ccd_weight),
            np.where(own > 0, own, np.nan),
            ccd_weight,
        )
        sd_hierarchy = None
        if not np.isnan(weight).any():
            sd_hierarchy = weighted_sd(rate, weight)
        table.append(
            dict(
                county_fips=fips,
                sd_ccd=sd_ccd,
                sd_hierarchy=sd_hierarchy,
                n_rows=rows.height,
            )
        )
    return pl.DataFrame(
        table,
        schema_overrides={"sd_ccd": pl.Float64, "sd_hierarchy": pl.Float64},
    ).sort("county_fips")
