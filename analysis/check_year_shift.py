"""
Checks of the year shift (cases of 2026, projections of 2027-2030), run after
the pipeline. Read-only: nothing is written.

Each check prints PASS, FAIL or SKIP. A check is skipped when an input is
missing and when a file it reads is older than the file it is built from (for
example a projection dataset written before the analysis dataset: rerun the
script that writes it). The exit status is 1 if any check FAILs. Only
invariants within one run are checked: a difference
from the files of an earlier run (the region corrections move the 2024 and
2025 rows; SC 2025 is 210 cases) is expected and not checked here.

Run it as `uv run analysis/check_year_shift.py`.
"""

import glob
import os
import re
import sys

import numpy as np
import polars as pl
import yaml

from outbreak_size.flag_discordance import COVARIATES
from outbreak_size.importation import IMPORT_CONDITIONING_VALIDATION_YEAR
from outbreak_size.projection_dataset import require_consistent_places
from outbreak_size.samples_io import load_samples

with open("params.yml") as f:
    GLOBALS_PRIOR_SD = yaml.safe_load(f)["validation_globals_prior_sd"]

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

VALIDATION_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
FIRST_PROJECTION_YEAR = 2027
FUTURE_YEARS = [2027, 2028, 2029, 2030]
# the model inputs of a county that two files must agree on
MODEL_INPUTS = list(
    dict.fromkeys(
        ["pop", "estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
        + list(COVARIATES)
    )
)

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(
        f"{'PASS' if ok else 'FAIL'}  {name}"
        + (f"  [{detail}]" if detail else "")
    )


def skip(name, path):
    print(f"SKIP  {name}  [missing: {path}]")


def stale_skip(name, path, *sources):
    """Print SKIP and return True if `path` is older than a file it is built
    from (a file of an earlier run says nothing about this run). Sources that
    do not exist are ignored."""
    newer = [s for s in sources if os.path.exists(s)]
    if newer and os.path.getmtime(path) < max(map(os.path.getmtime, newer)):
        print(
            f"SKIP  {name}  [stale: {path} is older than "
            f"{', '.join(newer)}; rerun the script that writes it]"
        )
        return True
    return False


def read(path, **kwargs):
    # every row is read to infer the types: a column that is empty in its
    # first rows (new_nebraska) is not read as another type in each file
    kwargs.setdefault("infer_schema_length", None)
    return pl.read_csv(path, schema_overrides={"fips": pl.String}, **kwargs)


def same_values(a: pl.Series, b: pl.Series) -> bool:
    if a.dtype.is_float() or b.dtype.is_float():
        return bool(
            np.allclose(
                a.cast(pl.Float64).to_numpy(),
                b.cast(pl.Float64).to_numpy(),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-9,
            )
        )
    return a.to_list() == b.to_list()


def columns_that_differ(
    a: pl.DataFrame, b: pl.DataFrame, columns
) -> list[str]:
    """The `columns` (those both frames have) whose values differ, rows in
    the order given."""
    return [
        c
        for c in columns
        if c in a.columns and c in b.columns and not same_values(a[c], b[c])
    ]


def exists(name, path):
    if os.path.exists(path):
        return True
    skip(name, path)
    return False


# 1. the rows up to 2025 of the file with 2026 are the rows of the file without
name = "analysis_dataset_with_2026 up to 2025 = analysis_dataset"
old_path = os.path.join(data_dir, "analysis_dataset.csv")
new_path = os.path.join(data_dir, "analysis_dataset_with_2026.csv")
if exists(name, old_path) and exists(name, new_path):
    old, new = read(old_path), read(new_path)
    keys = ["year", "state", "county"]
    new_to_2025 = new.filter(pl.col("year") <= 2025)
    unique = (
        not old.select(keys).is_duplicated().any()
        and not new_to_2025.select(keys).is_duplicated().any()
    )
    old, new_to_2025 = old.sort(keys), new_to_2025.sort(keys)
    shared = [c for c in old.columns if c in new.columns]
    differ = (
        columns_that_differ(old, new_to_2025, shared)
        if old.height == new_to_2025.height
        else ["row count"]
    )
    report(
        name,
        unique and not differ,
        f"{old.height} rows; differing columns: {differ}; keys unique: {unique}",
    )
    n_2026 = new.filter(pl.col("year") == 2026)
    report(
        "analysis_dataset_with_2026 has 2026 rows and no later year",
        n_2026.height > 0 and new["year"].max() == 2026,
        f"{n_2026.height} rows, {int(n_2026['cases'].sum())} cases",
    )
    if "vaccine_from_earlier_year" in new.columns:
        report(
            "vaccine_from_earlier_year is set on no row up to 2025",
            new_to_2025["vaccine_from_earlier_year"].fill_null(False).sum()
            == 0,
        )

    # 2. the train rows of each projection dataset are the observed rows
    for year in VALIDATION_YEARS:
        name = f"projection_dataset_{year}: train rows = the {year} rows of the analysis dataset"
        path = os.path.join(output_dir, f"projection_dataset_{year}.csv")
        if not exists(name, path) or stale_skip(name, path, new_path):
            continue
        train = read(path).filter(pl.col("type") == "train")["fips"].sort()
        observed = new.filter(pl.col("year") == year)["fips"].sort()
        report(
            name,
            train.to_list() == observed.to_list(),
            f"{train.len()} train rows, {observed.len()} observed",
        )

# 3. no observed rows in the base of the projections
name = f"projection_dataset_{FIRST_PROJECTION_YEAR} has no train rows"
base_path = os.path.join(
    output_dir, f"projection_dataset_{FIRST_PROJECTION_YEAR}.csv"
)
base = read(base_path) if exists(name, base_path) else None
if base is not None and stale_skip(name, base_path, new_path):
    base = None  # the checks that start from the base are skipped as well
if base is not None:
    report(name, base.filter(pl.col("type") == "train").height == 0)

# 4. the baseline scenario of 2027 and the projection rows of 2027 have the
# same model inputs: every value before a scenario is trended to the school
# year the outbreak year uses (Y - 1), so the year file of 2027 and the
# baseline scenario of 2027 are the same numbers
name = f"baseline {FIRST_PROJECTION_YEAR} = projection_dataset_{FIRST_PROJECTION_YEAR}"
bs_path = os.path.join(
    output_dir, f"projection_dataset_baseline_{FIRST_PROJECTION_YEAR}.csv"
)
if (
    base is not None
    and exists(name, bs_path)
    and not stale_skip(name, bs_path, new_path)
):
    baseline = read(bs_path).filter(pl.col("type") == "projection")
    projection = base.filter(pl.col("type") == "projection")
    same_places = baseline["fips"].to_list() == projection["fips"].to_list()
    differ = (
        columns_that_differ(baseline, projection, MODEL_INPUTS)
        if same_places
        else ["fips order"]
    )
    report(name, same_places and not differ, f"differing: {differ}")

# 5. the 2027 projection rows are the 2026 projection rows with the three
# vaccine values moved one school year along their slopes: the census and the
# other inputs are the same (the same census cap), only the vaccine columns
# may differ
VACCINE_INPUTS = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
name = (
    f"{FIRST_PROJECTION_YEAR} projection rows = 2026 projection rows, "
    "except the vaccine values"
)
p26_path = os.path.join(output_dir, "projection_dataset_2026.csv")
if (
    base is not None
    and exists(name, p26_path)
    and not stale_skip(name, p26_path, new_path)
):
    p26 = read(p26_path).filter(pl.col("type") == "projection").sort("fips")
    p27 = base.filter(pl.col("type") == "projection").sort("fips")
    report(
        f"projection rows are one per fips (2026 and {FIRST_PROJECTION_YEAR})",
        not p26["fips"].is_duplicated().any()
        and not p27["fips"].is_duplicated().any(),
    )
    same_places = p26["fips"].to_list() == p27["fips"].to_list()
    differ = (
        columns_that_differ(p26, p27, MODEL_INPUTS)
        if same_places
        else ["fips set"]
    )
    other = [c for c in differ if c not in VACCINE_INPUTS]
    report(name, same_places and not other, f"differing: {differ}")

# 5a. no_change of 2027 holds the last known school year's value (the year
# file of 2026): the same model inputs as the 2026 projection rows
name = (
    f"no_change {FIRST_PROJECTION_YEAR} = 2026 projection rows "
    "(the last known school year, trended)"
)
nc_path = os.path.join(
    output_dir, f"projection_dataset_no_change_{FIRST_PROJECTION_YEAR}.csv"
)
if (
    base is not None
    and exists(name, p26_path)
    and exists(name, nc_path)
    and not stale_skip(name, p26_path, new_path)
    and not stale_skip(name, nc_path, new_path)
):
    no_change = read(nc_path).sort("fips")
    p26 = read(p26_path).filter(pl.col("type") == "projection").sort("fips")
    same_places = no_change["fips"].to_list() == p26["fips"].to_list()
    differ = (
        columns_that_differ(no_change, p26, VACCINE_INPUTS)
        if same_places
        else ["fips order"]
    )
    report(name, same_places and not differ, f"differing: {differ}")

# 5b. the census of the year before the outbreak year (county_data_year_of_
# outbreak): the 2025 rows have the ACS of 2024 and the 2024 rows that of 2023,
# so their populations differ (they are the same where the census of the same
# year as the outbreak year is asked: a stale file); and the rows of 2025 to
# 2027 all have the latest ACS year, so their populations are the same
name = "census of the year before: 2024 and 2025 rows differ, 2025-2027 same"
paths = {
    y: os.path.join(output_dir, f"projection_dataset_{y}.csv")
    for y in (2024, 2025, 2026, FIRST_PROJECTION_YEAR)
}
if all(exists(name, p) for p in paths.values()) and not any(
    stale_skip(name, p, new_path) for p in paths.values()
):
    pops = {
        y: read(p)
        .filter(pl.col("type") == "projection")
        .select("fips", pl.col("pop").alias(f"pop_{y}"))
        for y, p in paths.items()
    }
    joined = pops[2024]
    for y in (2025, 2026, FIRST_PROJECTION_YEAR):
        joined = joined.join(pops[y], on="fips")
    share_2024_2025 = float(
        (joined["pop_2024"] == joined["pop_2025"]).cast(pl.Float64).mean()  # type: ignore
    )
    same_2025_on = (joined["pop_2025"] == joined["pop_2026"]).all() and (
        joined["pop_2026"] == joined[f"pop_{FIRST_PROJECTION_YEAR}"]
    ).all()
    report(
        name,
        bool(share_2024_2025 < 0.5 and same_2025_on),
        f"share of counties with the same population in 2024 and 2025: "
        f"{share_2024_2025:.3f}",
    )

# 6. every row of 2027 joins exactly one import row of the conditioning year
imp_path = os.path.join(output_dir, "empirical_importation_probability.csv")
name = "2027 rows join one 2026 projection import row each"
if exists(name, imp_path):
    imports = read(imp_path)
    report(
        "no import rows of 2027",
        imports.filter(
            pl.col("validation_year") >= FIRST_PROJECTION_YEAR
        ).height
        == 0,
    )
    conditioning = imports.filter(
        (pl.col("validation_year") == IMPORT_CONDITIONING_VALIDATION_YEAR)
        & (pl.col("type") == "projection")
    )
    for label, path in (
        ("projection_dataset_2027", base_path),
        (
            "projection_dataset_baseline_2027",
            os.path.join(
                output_dir,
                f"projection_dataset_baseline_{FIRST_PROJECTION_YEAR}.csv",
            ),
        ),
    ):
        if not exists(f"{label} joins import rows", path) or stale_skip(
            f"{label} joins import rows", path, new_path
        ):
            continue
        rows = read(path).filter(pl.col("type") == "projection")
        counts = rows.join(
            conditioning.group_by("fips").len(), on="fips", how="left"
        )["len"]
        report(
            f"{label}: each row joins one {IMPORT_CONDITIONING_VALIDATION_YEAR} projection import row",
            bool((counts == 1).all()),
            f"{rows.height} rows; without a row: {int(counts.is_null().sum())}, "
            f"with several: {int((counts > 1).sum())}",
        )

# 7. the draws match the rows of their projection file
pattern = re.compile(
    r"risk_projections_(?:(?P<scenario>[a-z_]+?)_(?P<year>\d{4})_)?globals_prior_sd="
)
# (the draws of the prior of the validation and projections; other priors are
# the files of other runs)
for path in sorted(
    glob.glob(
        os.path.join(
            output_dir,
            f"risk_projections_*globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
        )
    )
):
    match = pattern.match(os.path.basename(path))
    if match is None:
        continue
    if match["scenario"] is None:
        csv = base_path
        label = os.path.basename(path)
    else:
        csv = os.path.join(
            output_dir,
            f"projection_dataset_{match['scenario']}_{match['year']}.csv",
        )
        label = os.path.basename(path)
    # the files of 2026 scenarios are of an earlier run (listed below)
    if int(match["year"] or FIRST_PROJECTION_YEAR) < FIRST_PROJECTION_YEAR:
        continue
    if (
        not exists(f"width of {label}", csv)
        or stale_skip(f"width of {label}", csv, new_path)
        or stale_skip(f"width of {label}", path, csv)
    ):
        continue
    n_rows = read(csv).filter(pl.col("type") == "projection").height
    width = load_samples(path).shape[1]
    report(
        f"width of {label}",
        width == n_rows,
        f"{width} draws columns, {n_rows} rows",
    )

# 8. no 2026 in the inferential data and the validation of 2021-2025
name = "inferential data end in 2025"
if exists(name, old_path):
    report(name, read(old_path)["year"].max() == 2025)
for path in sorted(
    glob.glob(os.path.join(fig_dir, "supplemental_data_s[12]_*.csv"))
):
    table = pl.read_csv(path)
    column = next((c for c in table.columns if "year" in c.lower()), None)
    if column is not None:
        years = set(table[column].cast(pl.String).to_list())
        report(f"{os.path.basename(path)}: no 2026", "2026" not in years)

# 9. the scenario files are those of 2027-2030, and none of 2026 is read
scenarios = sorted(
    {
        os.path.basename(p)[len("projection_dataset_") : -len("_2030.csv")]
        for p in glob.glob(
            os.path.join(output_dir, "projection_dataset_*_20*.csv")
        )
    }
)
stale = sorted(
    glob.glob(os.path.join(output_dir, "projection_dataset_*_2026.csv"))
    + glob.glob(os.path.join(output_dir, "risk_projections_*_2026_*.npz"))
)
stale = [p for p in stale if not p.endswith("projection_dataset_2026.csv")]
if stale:
    print("INFO  files of 2026 scenarios of an earlier run (stale, not read):")
    for path in stale:
        print(f"        {path}")
scenario_pattern = re.compile(
    r"projection_dataset_[a-z_{}]*2026|risk_projections_[a-z_{}]*2026_"
)
readers = []
# (the sources next to this script, whatever the directory it is run from)
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for script in sorted(
    glob.glob(os.path.join(repo, "analysis", "*.py"))
    + glob.glob(os.path.join(repo, "src", "outbreak_size", "*.py"))
):
    if script.endswith("check_year_shift.py"):
        continue
    with open(script) as f:
        for number, line in enumerate(f, 1):
            if (
                scenario_pattern.search(line)
                and "projection_dataset_2026" not in line
            ):
                readers.append(f"{script}:{number}")
report("no script reads a 2026 scenario file", not readers, str(readers))
for scenario in scenarios:
    missing = [
        y
        for y in FUTURE_YEARS
        if not os.path.exists(
            os.path.join(output_dir, f"projection_dataset_{scenario}_{y}.csv")
        )
    ]
    report(
        f"scenario {scenario}: files for {FUTURE_YEARS}",
        not missing,
        f"missing {missing}",
    )

# 5. places: in each projection dataset a county has one geoid and a geoid one
# county, and the geoid is the analysis dataset's (the models number their
# spatial units by geoid and their adjacency matrix by fips; the year files
# label some counties with a region's name, "Panhandle" for Sheridan, Nebraska)
analysis_path = os.path.join(data_dir, "analysis_dataset_with_2026.csv")
if exists("projection datasets: one geoid per county", analysis_path):
    analysis_places = read(analysis_path).select("fips", "geoid")
    for path in sorted(
        glob.glob(os.path.join(output_dir, "projection_dataset_*.csv"))
    ):
        name = f"{os.path.basename(path)}: one geoid per county, the analysis dataset's"
        if stale_skip(name, path, analysis_path):
            continue
        try:
            require_consistent_places(read(path), analysis_places, name)
            report(name, True)
        except ValueError as error:
            report(name, False, str(error))

n_fail = results.count(False)
print(f"\n{results.count(True)} passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)
