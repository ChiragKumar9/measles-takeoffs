"""
Check the school vaccination files (assemble_school_vaccination.py) against
what the plan expected, and against our county coverage.

1. Invariants of the files (no NaN, rates in 0-100, enrollment missing exactly
   when its source is, one source per county-year, unique county-years, ...).
2. The rows each step dropped or changed, against the expected counts. Each
   expected count says where it comes from: "plan" is the estimate made before
   the build, "re-baselined" is set from the build itself (with the reason), so
   a match on a re-baselined count only shows that nothing changed since.
   A count is a problem if it differs by more than 5% or 5 rows.
3. The size of the county file.
4. The copied years that were flagged (AZ, LA, MA, OK, the regression of the plan).
5. The joins of the Washington Post year with Chen & Bento's series, into it and
   (where Chen & Bento continues) out of it: a state whose change is more than
   twice a typical year, or whose county pattern (r) falls more than 0.25 below
   a typical year, is listed.
6. The county file against data/processed_vaccinations_data.csv (state, source:
   counties compared, mean absolute gap, correlation, mean gap, in points; the
   accumulated mean against the coverage of the school year's start year). It is
   a yardstick, not the truth; its imputed rows (state values) are left out.

Run from the repo root, after assemble_school_vaccination.py:
    uv run analysis/check_school_vaccination.py
Exits with status 1 if anything differs.
"""

import os
import sys

import polars as pl
import yaml

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "school_vaccination"
)
with open("params.yml", "r") as f:
    cfg = yaml.safe_load(f)["school_vaccination"]
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(60)
problems = []


def read(name):
    return pl.read_csv(
        os.path.join(data_dir, name),
        schema_overrides={"county_fips": pl.String},
        infer_schema_length=100000,
    )


schools = read("school_vaccination_schools.csv")
counties = read("school_vaccination_counties.csv")
choice = pl.read_csv(
    os.path.join(output_dir, "source_by_state_year.csv"),
    infer_schema_length=100000,
)

# 1. invariants
print("1. invariants")
checks = []
for name, table in [
    ("school file", schools),
    ("county file", counties),
]:
    floats = [c for c, t in table.schema.items() if t == pl.Float64]
    nan = sum(int(table[c].is_nan().sum()) for c in floats)
    checks.append((f"{name}: no NaN", nan == 0))
checks += [
    (
        "school file: every school has a rate",
        schools["mmr_rate"].null_count() == 0,
    ),
    (
        "school file: rates within 0-100",
        schools["mmr_rate"].min() >= 0 and schools["mmr_rate"].max() <= 100,  # type: ignore
    ),
    (
        "school file: enrollment is missing exactly when enrollment_source is",
        schools.filter(
            pl.col("enrollment").is_null()
            != pl.col("enrollment_source").is_null()
        ).height
        == 0,
    ),
    (
        "school file: one source per county-year",
        schools.group_by("state", "county_fips", "year_start")
        .agg(n=pl.col("source").n_unique())["n"]
        .max()
        == 1,
    ),
    (
        "county file: county-years are unique",
        counties.select("state", "county_fips", "year_start")
        .is_duplicated()
        .sum()
        == 0,
    ),
    (
        "county file: weighted values only where every school has enrollment",
        counties.filter(
            pl.col("mean_weighted").is_not_null()
            & (pl.col("n_schools_with_enrollment") < pl.col("n_schools"))
        ).height
        == 0,
    ),
    (
        "county file: n_schools_all is at least n_schools",
        (counties["n_schools_all"] >= counties["n_schools"]).all(),
    ),
    (
        "county file: shares within 0-1",
        counties["fraction_public"].drop_nulls().is_between(0, 1).all()
        and counties["fraction_kids_public"]
        .drop_nulls()
        .is_between(0, 1)
        .all(),
    ),
    (
        "a Washington Post state-year has at least the share of Chen & Bento's schools that params.yml requires",
        choice.filter(
            (pl.col("source") == "washington_post")
            & (pl.col("n_chen_bento") > 0)
            & (
                pl.col("n_washington_post")
                < cfg["wapo_min_share_of_chen_bento"] * pl.col("n_chen_bento")
            )
        ).height
        == 0,
    ),
    (
        "every state-year of the school file has the schools of one source",
        schools.group_by("state", "year_start")
        .agg(n=pl.col("source").n_unique())
        .filter(pl.col("n") > 1)
        .height
        == 0,
    ),
]
# the percent rule of the Chen & Bento reader decides per state-year; no raw value may look like
# a proportion (0-1) inside a state-year it read as percent, nor like a percent inside a proportion one
raw = pl.read_csv(
    os.path.join(data_dir, "schools", "school_points_raw.csv"),
    infer_schema=False,
)
raw = raw.with_columns(
    mmr=pl.col("mmr").cast(pl.Float64, strict=False).fill_nan(None)
)
raw = raw.with_columns(median=pl.col("mmr").median().over("state", "year"))
as_percent = raw.filter(pl.col("median") > 1.5)
wapo_raw = pl.read_csv(
    os.path.join(data_dir, "vaxrates_schools.csv"),
    null_values="NA",
    encoding="utf8-lossy",
    infer_schema_length=10000,
)
checks += [
    (
        f"Chen & Bento: no rate in (0, 1] inside a state-year read as percent ({as_percent.select('state', 'year').unique().height} such state-years)",
        as_percent.filter((pl.col("mmr") > 0) & (pl.col("mmr") <= 1)).height
        == 0,
    ),
    (
        "Chen & Bento: at most 2 rates outside 0-1 inside the state-years read as proportions (SD 1.03, HI -0.0001)",
        raw.filter(
            (pl.col("median") <= 1.5)
            & ((pl.col("mmr") > 1) | (pl.col("mmr") < 0))
        ).height
        <= 2,
    ),
    (
        "Washington Post: no rate outside 0-1, no NaN",
        wapo_raw.filter(
            (pl.col("mmr_rate") > 1)
            | (pl.col("overall_rate") > 1)
            | (pl.col("mmr_rate") < 0)
            | (pl.col("overall_rate") < 0)
            | pl.col("mmr_rate").is_nan()
            | pl.col("overall_rate").is_nan()
        ).height
        == 0,
    ),
]
for name, passed in checks:
    print("   " + ("pass  " if passed else "FAIL  ") + name)
    if not passed:
        problems.append(name)

# 2. rows dropped or changed by each step:
# (source, action, reason starts with, expected, states left out, where the expected count comes from)
log = pl.read_csv(
    os.path.join(output_dir, "qc_log.csv"), infer_schema_length=100000
)
EXPECTED = [
    ("washington_post", "dropped", "missing or out-of-range", 6, [], "plan"),
    (
        "washington_post",
        "dropped",
        "more than 20 km from the counties",
        26,
        [],
        "plan",
    ),
    (
        "washington_post",
        "dropped",
        "more than 20 km from the county its label",
        128,
        [],
        "plan",
    ),
    (
        "washington_post",
        "dropped",
        "identical Washington Post record",
        0,
        [],
        "dry run (Oct 31): the rule needs the same name, point and rate; the old rule dropped 2 identical rows whose points differ by metres",
    ),
    (
        "washington_post",
        "changed",
        "enrollment of 0 or less",
        126,
        [],
        "dry run (Oct 31); none of them has a rate",
    ),
    (
        "washington_post",
        "flagged",
        "the county of the point differs",
        537,
        [],
        "plan",
    ),
    # (the low-rate rule is for Chen & Bento only: the Washington Post has no enrollment)
    (
        "chen_bento",
        "dropped",
        "more than 20 km from the counties",
        14,
        [],
        "plan said 13; re-baselined: the raw file also has the Garrard school without MMR",
    ),
    # (ND and TN: Chen & Bento is not left out there, so their duplicates count)
    (
        "chen_bento",
        "dropped",
        "identical Chen & Bento record",
        471,
        [],
        "re-baselined: 291 without ND and TN (an earlier dry run: same point and rate unless another school year shows two or more schools there; the old rule dropped 495 here)",
    ),
    (
        "chen_bento",
        "dropped",
        "more than 15 records at one point",
        2895,
        [],
        "dry run (Oct 31): fallback points (VA, SD, KY, MN, IN, IA)",
    ),
    (
        "chen_bento",
        "changed",
        "enrollment of 0 or less",
        1793,
        [],
        "re-baselined after the regeneration: 1,803 before the fallback step, which removes 10 of them first; 1,363 before it had a rate (MD 1,122 at exactly 100%)",
    ),
    (
        "chen_bento",
        "changed",
        "Chen & Bento rate of exactly 0",
        138,
        [],
        "plan said 8 (a rate below 10%); changed to exactly 0: the raw file holds the placeholders (144 in SD, all 0.0) that the old panel dropped; 162 of the 165 rows below 10% with no enrollment are exactly 0.0; 138 after the fallback step, which removes 24 of them first",
    ),
    (
        "chen_bento",
        "changed",
        "rate outside 0-100",
        2,
        [],
        "re-baselined: not in the plan (SD 103, HI -0.01), now missing, not clipped; an earlier 195 included 193 NaN rates, see the next line",
    ),
    (
        "chen_bento",
        "changed",
        "district rate of exactly 0%",
        1,
        [],
        "new rule: Texas Tech University K-12 (Lubbock County, 2024-25)",
    ),
    (
        "chen_bento",
        "changed",
        "NaN treated as missing",
        193,
        [],
        "re-baselined: found in review (MI 142, ID 33, DC 18); these had become 100%",
    ),
]
rows = []
for source, action, reason, expected, left_out, origin in EXPECTED:
    found = log.filter(
        (pl.col("source") == source)
        & (pl.col("action") == action)
        & pl.col("reason").str.starts_with(reason)
        & ~pl.col("state").is_in(left_out)
    ).height
    off = abs(found - expected) > max(5, 0.05 * expected)
    rows.append(
        (
            source,
            action,
            reason,
            expected,
            found,
            origin,
            "DIFFERENT" if off else "",
        )
    )
    if off:
        problems.append(
            f"{source} {reason}: expected about {expected}, found {found}"
        )
print("\n2. rows dropped, flagged or changed")
print(
    pl.DataFrame(
        rows,
        schema=[
            "source",
            "action",
            "reason",
            "expected",
            "found",
            "where the expected count comes from",
            "flag",
        ],
        orient="row",
    )
)

# 3. the county file
years = counties.group_by("state").agg(years=pl.col("year_start").n_unique())
sizes = [
    (
        "county-years",
        counties.height,
        15091,
        "was 11,379 (plan said 11,378); re-baselined: one source per state-year, Chen & Bento also for ND and TN, and the readers of the state files (OH, IA, ME, WI, CO, IN, NC, MA, MN, PA, TX, MD, NY, CA, SC)",
    ),
    (
        "county-years from the Washington Post file",
        counties.filter(pl.col("source") == "washington_post").height,
        1831,
        "was 1,991; re-baselined: the Post year of a partial file (IA, IL, LA, MN) goes to Chen & Bento, and the Post no longer fills single counties",
    ),
    (
        "county-years from Chen & Bento",
        counties.filter(pl.col("source") == "chen_bento").height,
        9741,
        "was 9,388; re-baselined (see the line above)",
    ),
    (
        "states with a single school year",
        years.filter(pl.col("years") == 1).height,
        13,
        "was 18; re-baselined (the state files and Chen & Bento years of ND and TN give states more years)",
    ),
    (
        "states with five or more school years",
        years.filter(pl.col("years") >= 5).height,
        20,
        "was 16 (plan); re-baselined: the state files give states more school years",
    ),
]
print("\n3. the county file")
for name, found, expected, origin in sizes:
    off = abs(found - expected) > max(3, 0.01 * expected)
    print(
        f"   {name}: {found} (expected {expected}; {origin})",
        "  <- DIFFERENT" if off else "",
    )
    if off:
        problems.append(f"{name}: expected about {expected}, found {found}")

# 4. copied years
continuity = pl.read_csv(
    os.path.join(output_dir, "continuity_diagnostics.csv")
)
flagged_years = sorted(
    continuity.filter(pl.col("copied_year"))
    .select("state", "year_start_second")
    .rows()
)
expected_years = sorted(
    [
        ("AZ", 2016),
        ("LA", 2022),
        ("LA", 2023),
        ("LA", 2024),
        ("MA", 2018),
        ("OK", 2019),
    ]
)
print("\n4. copied years flagged:", flagged_years, "| plan:", expected_years)
if flagged_years != expected_years:
    problems.append(
        f"copied years changed: now {flagged_years}, plan {expected_years}"
    )

# 5. the joins
seam = pl.read_csv(
    os.path.join(output_dir, "seam_diagnostics.csv")
).with_columns(
    into_ratio=pl.col("seam_mean_abs_change")
    / pl.col("typical_mean_abs_change"),
    into_r_drop=pl.col("typical_r") - pl.col("seam_r"),
    out_ratio=pl.col("next_mean_abs_change")
    / pl.col("typical_mean_abs_change"),
    out_r_drop=pl.col("typical_r") - pl.col("next_r"),
)
into = sorted(
    seam.filter((pl.col("into_ratio") > 2) | (pl.col("into_r_drop") > 0.25))[
        "state"
    ].to_list()
)
out = sorted(
    seam.filter((pl.col("out_ratio") > 2) | (pl.col("out_r_drop") > 0.25))[
        "state"
    ].to_list()
)
print(
    "\n5. unusual joins into the Washington Post year:",
    into,
    "| plan: ['CO'] (the baseline leaves out copied years: re-baselined)",
)
print(
    "   unusual joins out of it (AZ, CO, NC, WA have a Chen & Bento year after):",
    out,
    "| plan: none were expected to be unusual; this is the first measurement",
)
if into != ["CO"]:
    problems.append(
        f"joins into the Washington Post year: now {into}, plan ['CO']"
    )
if out:
    problems.append(
        f"joins out of the Washington Post year are unusual in {out}"
    )
print(
    seam.select(
        "state",
        "washington_post_year_start",
        "seam_mean_change",
        "seam_mean_abs_change",
        "seam_r",
        "next_mean_change",
        "next_mean_abs_change",
        "next_r",
        "typical_mean_abs_change",
        "typical_r",
        "schools_washington_post",
        "schools_chen_bento",
    )  # fmt: skip
    .sort("state")
    .with_columns(pl.col(pl.Float64).round(2))
)

# 6. against our county coverage
reference = (
    pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        infer_schema_length=20000,
        null_values=["NA"],
        schema_overrides={"fips": pl.String},
    )
    .filter(~pl.col("_is_imputed"))
    .select(
        county_fips=pl.col("fips").str.zfill(5),
        year_start=pl.col("year"),
        reference=pl.col("estimated_mmr_cov"),
    )
)  # imputed rows are state values, not county data
joined = counties.join(
    reference,
    on=["county_fips", "year_start"],
    maintain_order="left",
    # one reference value per county and year
    validate="m:1",
).filter(
    pl.col("reference").is_not_null()
    & pl.col("mean_accumulated").is_not_null()
)
gap = pl.col("mean_accumulated") - pl.col("reference")
benchmark = (
    joined.group_by("state", "source")
    .agg(
        counties=pl.len(),
        mean_abs_gap=gap.abs().mean(),
        r=pl.corr("mean_accumulated", "reference"),
        mean_gap=gap.mean(),
    )
    .sort("state", "source")
)
benchmark.write_csv(os.path.join(output_dir, "benchmark_comparison.csv"))
print(
    "\n6. against our county coverage; states with a mean absolute gap above 6 points:"
)
print(
    benchmark.filter(pl.col("mean_abs_gap") > 6).with_columns(
        pl.col(pl.Float64).round(2)
    )
)

if problems:
    print("\nPROBLEMS:")
    for problem in problems:
        print("  -", problem)
    sys.exit(1)
print("\nNothing differs from what was expected.")
