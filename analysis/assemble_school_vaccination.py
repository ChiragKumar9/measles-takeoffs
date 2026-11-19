"""
Build the school-level and county-level vaccination files from the school
sources: Chen & Bento, the Washington Post, and the state files (data/schools/
state_files_schools.csv, school-level files published by the states; optional).
The rules are in src/outbreak_size/school_vaccination.py; the constants are in
params.yml, school_vaccination.

Run from the repo root, after the R converter and after
analysis/00_explore_data.py (the county files carry a QC flag that compares
the school mean with the county MMR of data/processed_vaccinations_data.csv,
imputed values excluded):
    Rscript --vanilla analysis/export_school_rds.R
    uv run analysis/assemble_school_vaccination.py

Writes, in data/:
    school_vaccination_schools.csv, school_vaccination_counties.csv
and, in outputs/school_vaccination/: the source of each state and school year
(source_by_state_year.csv), the QC log (every dropped or changed row
and why), the row counts of each step, and the checks (copied years, how the
Washington Post year joins Chen & Bento's series, borrowed enrollment, whether
enrollment is the denominator of the rate).
"""

import os

import polars as pl
import yaml

from outbreak_size import school_vaccination as sv

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "school_vaccination"
)
os.makedirs(output_dir, exist_ok=True)

with open("params.yml", "r") as f:
    cfg = yaml.safe_load(f)["school_vaccination"]

log = sv.QCLog()


def step(df, source, name, function):
    """Run one step and record the row counts before and after it."""
    before = df.height
    df = function(df)
    log.count(source, name, before, df.height)
    return df


# 1. read both sources
raw_chen_bento = pl.read_csv(
    os.path.join(data_dir, "schools", "school_points_raw.csv"),
    infer_schema=False,
)
raw_wapo = pl.read_csv(
    os.path.join(data_dir, "vaxrates_schools.csv"),
    null_values="NA",
    encoding="utf8-lossy",
    infer_schema_length=10000,
)
STATE_FILES = os.path.join(data_dir, "schools", "state_files_schools.csv")
if os.path.exists(STATE_FILES):
    state_files = pl.read_csv(
        STATE_FILES,
        schema_overrides={
            k: v for k, v in sv.COLUMN_TYPES.items() if k != "rid"
        },
    ).with_columns(pl.col("rid").cast(pl.UInt32))
else:
    print(
        f"no {STATE_FILES}: the state-years only the state files cover are missing"
    )
    state_files = pl.DataFrame(schema=sv.COLUMN_TYPES)
counties = sv.read_counties(
    os.path.join(
        data_dir, "cb_2020_us_county_500k", "cb_2020_us_county_500k.shp"
    )
)
chen_bento = sv.read_chen_bento(raw_chen_bento, log)
wapo = sv.read_washington_post(raw_wapo, log)
log.count(sv.CHEN_BENTO, "read", chen_bento.height, chen_bento.height)
log.count(sv.WAPO, "read", wapo.height, wapo.height)

# 2-4. place checks, value checks, duplicates (each source on its own)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "rates outside 0-100 to missing",
    lambda d: sv.missing_rates_outside_range(d, log),
)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "coordinates",
    lambda d: sv.drop_invalid_coordinates(d, cfg["us_bounds"], log),
)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "county of the point",
    lambda d: sv.assign_counties(d, counties, cfg["max_km_from_state"], log),
)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "fallback points",
    lambda d: sv.drop_fallback_points(
        d,
        cfg["fallback_point_min_schools"],
        cfg["fallback_point_min_share"],
        log,
    ),
)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "value checks",
    lambda d: sv.value_checks(d, log),
)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "district rows at 0%",
    lambda d: sv.missing_district_zero_rates(
        d, cfg["chen_bento_district_states"], log
    ),
)
chen_bento = step(
    chen_bento,
    sv.CHEN_BENTO,
    "duplicates",
    lambda d: sv.drop_duplicates_chen_bento(d, log),
)

wapo = step(
    wapo,
    sv.WAPO,
    "rates outside 0-100 to missing",
    lambda d: sv.missing_rates_outside_range(d, log),
)
wapo = step(
    wapo,
    sv.WAPO,
    "coordinates",
    lambda d: sv.drop_invalid_coordinates(d, cfg["us_bounds"], log),
)
wapo = step(
    wapo,
    sv.WAPO,
    "county of the point",
    lambda d: sv.assign_counties(d, counties, cfg["max_km_from_state"], log),
)
wapo = step(
    wapo,
    sv.WAPO,
    "county of the label",
    lambda d: sv.check_county_labels(
        d, counties, cfg["max_km_from_label_county"], log
    ),
)
wapo = step(
    wapo,
    sv.WAPO,
    "fallback points",
    lambda d: sv.drop_fallback_points(
        d,
        cfg["fallback_point_min_schools"],
        cfg["fallback_point_min_share"],
        log,
    ),
)
wapo = step(
    wapo,
    sv.WAPO,
    "value checks",
    lambda d: sv.value_checks(d, log),
)
wapo = step(
    wapo,
    sv.WAPO,
    "district rows at 0%",
    lambda d: sv.missing_district_zero_rates(
        d, cfg["chen_bento_district_states"], log
    ),
)
wapo = step(
    wapo,
    sv.WAPO,
    "duplicates",
    lambda d: sv.drop_duplicates_washington_post(d, log),
)

# the state files come with their county and are checked on their own: rates outside 0-100 and district rows at 0%
state_files = step(
    state_files,
    sv.STATE_FILE,
    "rates outside 0-100 to missing",
    lambda d: sv.missing_rates_outside_range(d, log),
)
state_files = step(
    state_files,
    sv.STATE_FILE,
    "district rows at 0%",
    lambda d: sv.missing_district_zero_rates(
        d, cfg["chen_bento_district_states"], log
    ),
)

chen_bento = sv.count_schools_at_point(chen_bento)
wapo = sv.count_schools_at_point(wapo)

# 5. copied years (flagged, nothing removed)
chen_bento, copied_checks = sv.flag_copied_years(
    chen_bento,
    cfg["copied_year_unchanged_share"],
    cfg["copied_year_min_schools"],
)


def assemble():
    """Source selection, borrowed enrollment, inherited types, flags, the two files."""
    selected, choice = sv.select_sources(chen_bento, wapo, state_files, cfg)
    selected, borrowed = sv.borrow_enrollment(selected, chen_bento, cfg)
    selected = sv.inherit_school_types(selected, wapo, cfg)
    selected = sv.add_flags(selected, cfg)
    return (
        sv.school_file(selected),
        sv.county_file(selected),
        borrowed,
        selected,
        choice,
    )


schools, county_table, borrowed, selected, choice = assemble()
# the QC flag of panel d: the mean school rate within school_mean_qc_max_gap_pp of the county MMR of the same school year
# (processed_vaccinations_data.csv labels a school year by its start year); an imputed county MMR makes the flag empty
county_mmr = pl.read_csv(
    os.path.join(data_dir, "processed_vaccinations_data.csv"),
    schema_overrides={"fips": pl.String},
    infer_schema_length=50000,
).select(
    "fips",
    "year",
    "estimated_mmr_cov",
    pl.col("_is_imputed").alias("is_imputed"),
)
county_mmr = county_mmr.filter(
    pl.col("fips").is_not_null() & (pl.col("fips").str.len_chars() == 5)
)
county_table = sv.add_county_mmr_qc(
    county_table, county_mmr, cfg["school_mean_qc_max_gap_pp"]
)
# a county-year whose schools all lack a rate has no mean and is not in the county file
no_rate = (
    selected.group_by("state", "county_fips", "year_start")
    .agg(
        county_name=pl.col("county_name").first(),
        source=pl.col("source").first(),
        rated=pl.col("rate").is_not_null().sum(),
        schools=pl.len(),
    )
    .filter(pl.col("rated") == 0)
    .drop("rated")
    .rename({"schools": "n_schools_all"})
    .sort("state", "county_fips", "year_start")
)
log.count(
    "both",
    "county-years in which no school has a rate (not in the county file)",
    no_rate.height,
    0,
)

for name, table in [
    ("school_vaccination_schools.csv", schools),
    ("school_vaccination_counties.csv", county_table),
]:
    table.write_csv(os.path.join(data_dir, name))
    print(f"wrote data/{name}: {table.height} rows")

# the QC log and the checks
# (a row can be logged twice in a step: the ties keep the order logged)
log.rows_frame().sort(
    "source", "step", "row_id", maintain_order=True
).write_csv(os.path.join(output_dir, "qc_log.csv"))
log.counts_frame().write_csv(os.path.join(output_dir, "step_counts.csv"))
copied_checks.write_csv(os.path.join(output_dir, "continuity_diagnostics.csv"))
no_rate.write_csv(os.path.join(output_dir, "county_years_without_a_rate.csv"))
choice.write_csv(os.path.join(output_dir, "source_by_state_year.csv"))
sv.seam_diagnostics(chen_bento, wapo, choice).write_csv(
    os.path.join(output_dir, "seam_diagnostics.csv")
)
borrowed.write_csv(os.path.join(output_dir, "borrow_diagnostics.csv"))
sv.enrollment_denominator_test(
    pl.concat([chen_bento, wapo], how="diagonal_relaxed")
).write_csv(os.path.join(output_dir, "enrollment_basis_diagnostic.csv"))
print(log.counts_frame())
print(
    f"{county_table.height} county-years in {county_table['state'].n_unique()} states: "
    + ", ".join(
        f"{s}: {n}"
        for s, n in county_table.group_by("source")
        .len()
        .sort("source")
        .iter_rows()
    )
)
