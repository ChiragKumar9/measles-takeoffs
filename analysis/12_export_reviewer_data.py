"""
Export the unhashed reviewer-facing data bundle to outputs/reviewer_data_unhashed/.

Main file: outbreak_records_master.csv -- ONE ROW PER RAW OUTBREAK RECORD from the
case sources (nothing silently disappears), each tagged INCLUDED_DIRECT /
INCLUDED_IMPUTED / DROPPED, with the reason and, for imputed rows, which vaccine
fields were imputed and how.

Reads only ./data (no network / Census calls) and re-uses the pipeline's own
readers from 00_explore_data.py, so the case cleaning is mirrored, not re-invented.
The mirrored cleaning is asserted to reproduce the pipeline's own output.
Values in the master come from the pipeline's files (analysis_dataset*.csv,
raw_merged_outbreaks_vaccinations.csv); only labels/flags are derived here.
"""

import hashlib
import importlib.util
import os
import shutil
import sys

import geopandas as gpd
import polars as pl
from addfips import AddFIPS

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo, "data")
out_dir = os.path.join(
    repo, "outputs", "reviewer_data_unhashed"
)  # outputs/reviewer_data/ holds only the hashed files for the reviewer
os.makedirs(out_dir, exist_ok=True)

# import 00_explore_data.py as a module (its work is under __main__, but it parses
# argv at import, so hide ours)
argv, sys.argv = sys.argv, ["x"]
os.chdir(repo)  # the module opens params.yml relative to the repo root
spec = importlib.util.spec_from_file_location(
    "explore", os.path.join(repo, "analysis", "00_explore_data.py")
)
if spec is None or spec.loader is None:
    raise ImportError("cannot load 00_explore_data.py")
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)
sys.argv = argv

from outbreak_size.outbreaks import process_outbreaks_df  # noqa: E402
from outbreak_size.regions import higher_level_regions  # noqa: E402

KEY = ["year", "state", "county"]
REGIONS = higher_level_regions()
# multi-county places whose vaccine value is a population-weighted average of counties
AGGREGATED_REGIONS = [r for r in REGIONS if r != "Gulf Coast"]
# regions that appear in the case data but are never given vaccine data
UNMAPPED_REGIONS = [
    "District 1",
    "Fulton, Gwinett",
    "Forsyth, Fulton, Paulding",
    "Blue Ridge",
]


def sy(v):
    """vaccine school year label: 2023 -> '2023-24'"""
    return f"{v}-{str(v + 1)[2:]}"


# ---------------------------------------------------------------------------
# 1. Case records, with every drop/recode tracked (mirrors read_outbreaks +
#    process_outbreaks_df, in the same order). A record keeps its row; a drop
#    only sets drop_stage/drop_reason (first reason wins = pipeline order).
# ---------------------------------------------------------------------------
SRC_COMBINED = "Combined Measles Dataset QCed.xlsx"
SRC_2025 = "2025CasesImports QCed clean.csv"
SRC_SC = "SC2025Cases.xlsx"

combined = pl.read_excel(os.path.join(data_dir, SRC_COMBINED))
combined = combined.rename({c: c.lower() for c in combined.columns})
combined = combined.with_columns(
    source_file=pl.lit(SRC_COMBINED), source_row=pl.int_range(pl.len()) + 2
)
csv25 = pl.read_csv(
    os.path.join(data_dir, SRC_2025), schema_overrides={"fips": pl.String}
).with_columns(
    source_file=pl.lit(SRC_2025), source_row=pl.int_range(pl.len()) + 2
)
sc = pl.read_excel(os.path.join(data_dir, SRC_SC))
sc = sc.rename({c: c.lower() for c in sc.columns})
sc = sc.with_columns(
    source_file=pl.lit(SRC_SC),
    source_row=pl.int_range(pl.len()) + 2,
    indigenous=pl.lit(None),
    imported=pl.lit(None),
    fips=pl.lit(None),
    year=pl.lit(2025),
    state=pl.lit("South Carolina"),
)
rec = pl.concat([combined, csv25, sc], how="diagonal_relaxed")
rec = rec.with_columns(
    record_id=pl.col("source_file").replace(
        {SRC_COMBINED: "COMBINED", SRC_2025: "QC2025", SRC_SC: "SC2025"}
    )
    + pl.lit(":")
    + pl.col("source_row").cast(pl.String),
    year_source=pl.col("year").cast(pl.Int64),
    county_source=pl.col("county"),
    cases_source=pl.col("cases").cast(pl.Float64),
    indigenous=pl.col("indigenous").cast(pl.Float64),
    imported=pl.col("imported").cast(pl.Float64),
    drop_stage=pl.lit(None, dtype=pl.String),
    drop_reason=pl.lit(None, dtype=pl.String),
    recode_notes=pl.lit(None, dtype=pl.String),
)
rec = rec.with_columns(
    year=pl.col("year_source"), cases=pl.col("cases_source")
).drop("__unnamed__6", strict=False)
assert rec["record_id"].is_unique().all()


def mark(rec, mask, stage, reason):
    """Drop (first reason wins) the records matching `mask`."""
    hit = mask & pl.col("drop_stage").is_null()
    return rec.with_columns(
        drop_stage=pl.when(hit)
        .then(pl.lit(stage))
        .otherwise(pl.col("drop_stage")),
        drop_reason=pl.when(hit)
        .then(pl.lit(reason))
        .otherwise(pl.col("drop_reason")),
    )


def note(rec, mask, text):
    """Append a recode note to surviving records matching `mask`."""
    hit = mask & pl.col("drop_stage").is_null()
    return rec.with_columns(
        recode_notes=pl.when(hit)
        .then(
            pl.concat_str(
                [pl.col("recode_notes"), text],
                separator="; ",
                ignore_nulls=True,
            )
            if isinstance(text, pl.Expr)
            else pl.concat_str(
                [pl.col("recode_notes"), pl.lit(text)],
                separator="; ",
                ignore_nulls=True,
            )
        )
        .otherwise(pl.col("recode_notes"))
    )


is_csv25 = pl.col("source_file") == SRC_2025
# --- read_outbreaks ---------------------------------------------------------
bad_year = is_csv25 & (pl.col("year") == 2925)
rec = note(rec, bad_year, "year 2925 corrected to 2025 (typo in source)")
rec = rec.with_columns(
    year=pl.when(bad_year).then(2025).otherwise(pl.col("year"))
)
rec = mark(
    rec,
    is_csv25 & (pl.col("county") == "Central") & (pl.col("state") == "Utah"),
    "case_location_unusable",
    "'Central' Utah is a multi-county region that cannot be located to a county",
)
rec = mark(
    rec,
    is_csv25 & pl.col("Lauren Case").is_null(),
    "case_report_not_verified",
    "could not verify case report",
)
override = (
    is_csv25
    & pl.col("Lauren Case").is_not_null()
    & (pl.col("Lauren Case").cast(pl.Float64) != pl.col("cases_source"))
)
rec = note(
    rec,
    override,
    pl.concat_str(
        [
            pl.lit("cases overridden by QC'd 2025 count: "),
            pl.col("cases_source").cast(pl.Int64).cast(pl.String),
            pl.lit(" -> "),
            pl.col("Lauren Case")
            .cast(pl.Float64)
            .cast(pl.Int64)
            .cast(pl.String),
        ]
    ),
)
rec = rec.with_columns(
    cases=pl.when(is_csv25 & pl.col("Lauren Case").is_not_null())
    .then(pl.col("Lauren Case").cast(pl.Float64))
    .otherwise(pl.col("cases"))
)
rec = mark(
    rec,
    is_csv25 & (pl.col("state") == "South Carolina"),
    "case_source_superseded",
    "South Carolina 2025 taken from SC2025Cases.xlsx instead of the 2025 QC sheet",
)
# --- process_outbreaks_df ---------------------------------------------------
rec = mark(
    rec,
    pl.col("year").is_null(),
    "case_not_a_record",
    "summary/total row in the source sheet (no year)",
)
rec = mark(
    rec,
    pl.col("year") < 2011,
    "case_outside_study_period",
    "before the 2011-2025 study period",
)
rec = mark(
    rec,
    (pl.col("cases") == 0) & (pl.col("cases_source") != 0),
    "case_zero_or_missing",
    "QC'd 2025 case count is 0 (overrides the non-zero count in the source sheet)",
)
rec = mark(
    rec,
    pl.col("cases") == 0,
    "case_zero_or_missing",
    "zero cases reported (no outbreak)",
)
rec = mark(
    rec,
    pl.col("cases").is_null(),
    "case_zero_or_missing",
    "case count missing",
)
rec = rec.with_columns(county=pl.col("county"))
de = pl.col("state") == "Delaware"
rec = note(
    rec,
    de & (pl.col("county") != "State"),
    pl.concat_str(
        [
            pl.lit("Delaware county '"),
            pl.col("county"),
            pl.lit(
                "' pooled to state level (Delaware reports vaccine data by state)"
            ),
        ]
    ),
)
rec = rec.with_columns(
    county=pl.when(de).then(pl.lit("State")).otherwise(pl.col("county"))
)
dc = (pl.col("state") == "District of Columbia") & (
    pl.col("county") != "District of Columbia"
)
rec = note(rec, dc, "county set to 'District of Columbia' (none listed)")
rec = rec.with_columns(
    county=pl.when(pl.col("state") == "District of Columbia")
    .then(pl.lit("District of Columbia"))
    .otherwise(pl.col("county"))
)
rec = mark(
    rec,
    pl.col("county").is_null(),
    "case_location_unusable",
    "no county listed (state-level count)",
)
rec = mark(
    rec,
    pl.col("county").is_in(["Unknown", "TOTAL", "Out of state resident"]),
    "case_location_unusable",
    "county is 'Unknown'/'TOTAL'/'Out of state resident'; cannot be located to a county",
)
SPELLING = [
    ("San Bernadino", "San Bernardino"),
    ("Los Angleles", "Los Angeles"),
    ("shasta", "Shasta"),
    ("Alamakee", "Allamakee"),
    ("Detroit City", "Detroit"),
    ("St Clair", "St. Clair"),
    ("Washentaw", "Washtenaw"),
    ("LaSueur", "Le Sueur"),
    ("LeSueur", "Le Sueur"),
    ("St Louis City", "St. Louis City"),
    ("St Charles", "St. Charles"),
    ("New York City", "Kings"),
    ("Salt lake", "Salt Lake"),
    ("Albermarle", "Albemarle"),
    ("Spokane\xa0", "Spokane"),
]
for old, new in SPELLING:
    hit = pl.col("county").str.contains(old, literal=True)
    text = (
        "New York City records assigned to Kings County (assumption: NYC outbreak was in Brooklyn)"
        if old == "New York City"
        else f"county name '{old}' -> '{new}' (spelling/standardisation)"
    )
    rec = note(rec, hit, text)
    rec = rec.with_columns(county=pl.col("county").str.replace_all(old, new))
det = pl.col("county").str.contains("Detroit", literal=True)
rec = note(
    rec, det, "Detroit aggregated up to Wayne County (Detroit has no FIPS)"
)
rec = rec.with_columns(
    county=pl.col("county").str.replace_all("Detroit", "Wayne")
)
rec = mark(
    rec,
    (pl.col("year") == 2024)
    & (pl.col("state") == "Illinois")
    & (pl.col("county") == "Cook"),
    "case_outbreak_removed",
    "outbreak among a non-US population (Chicago migrant shelter, 2024)",
)
rec = mark(
    rec,
    (pl.col("year") == 2021)
    & (pl.col("state") == "Wisconsin")
    & (pl.col("county") == "Monroe"),
    "case_outbreak_removed",
    "outbreak among a non-US population (Afghan evacuees, Operation Allies Welcome, 2021)",
)
rec = mark(
    rec,
    (pl.col("year") == 2016)
    & (pl.col("state") == "Arizona")
    & (pl.col("county") == "Pinal"),
    "case_outbreak_removed",
    "outbreak in an ICE detention facility (2016)",
)

# --- check the mirrored cleaning against the pipeline's own output ---------
alive = rec.filter(pl.col("drop_stage").is_null())
cy = (
    alive.group_by(KEY)
    .agg(
        n_records=pl.len(),
        cases=pl.sum("cases"),
        imported_raw=pl.when(pl.col("imported").is_null().all())
        .then(None)
        .otherwise(pl.col("imported").sum()),
        indigenous_raw=pl.when(pl.col("indigenous").is_null().all())
        .then(None)
        .otherwise(pl.col("indigenous").sum()),
        case_sources=pl.col("source_file").unique().sort().str.join(" + "),
    )
    .sort(KEY)
)
pipe = process_outbreaks_df(explore.read_outbreaks(data_dir)).sort(KEY)
assert cy.select(KEY).equals(pipe.select(KEY)), (
    "case-county-year keys differ from pipeline"
)
chk = cy.join(pipe, on=KEY, suffix="_p")
assert (chk["cases"] == chk["cases_p"]).all(), "cases differ"
assert (chk["imported_raw"].fill_null(0) == chk["imported"]).all(), (
    "imported differ"
)
assert (chk["indigenous_raw"].fill_null(0) == chk["indigenous"]).all(), (
    "indigenous differ"
)
print(
    f"[ok] mirrored case cleaning reproduces pipeline: {cy.height} county-years"
)

# ---------------------------------------------------------------------------
# 2. Vaccine side: merge result + analysis datasets (values come from the pipeline)
# ---------------------------------------------------------------------------
S = {"fips": pl.String}
rm = pl.read_csv(
    os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
    schema_overrides={**S, "new_nebraska": pl.String},
)
prim = pl.read_csv(
    os.path.join(data_dir, "analysis_dataset.csv"), schema_overrides=S
)
imp = pl.read_csv(
    os.path.join(data_dir, "analysis_dataset_imputed.csv"), schema_overrides=S
)
assert rm.select(KEY).is_unique().all() and rm.height == cy.height
assert cy.select(KEY).equals(rm.select(KEY).sort(KEY)), (
    "merge keys differ from case keys"
)
assert prim.select(KEY).is_unique().all() and imp.select(KEY).is_unique().all()
assert prim.join(imp.select(KEY), on=KEY, how="anti").height == 0, (
    "primary not subset of imputed"
)
MEAS = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
both = prim.join(imp, on=KEY, suffix="_i")
for c in MEAS + [
    "cases_outbreak",
    "imported",
    "neighbors_coverage",
    "oddities_2025",
    "mega_region_2025",
]:
    same = both[c].eq_missing(both[c + "_i"])
    assert same.all(), f"primary vs imputed differ on {c}"
assert prim["imputation_source"].is_null().all()
print(
    f"[ok] primary {prim.height} rows ⊂ imputed {imp.height} rows, identical on model columns"
)

# vaccine QC edits made when reading the compiled vaccine sheet (mirrors read_vaccinations)
VACC_QC_DROPS = [
    ("Butts", "Georgia", 2024),
    ("Jasper", "Georgia", 2024),
    ("Lumpkin", "Georgia", 2024),
    ("Richmond", "Georgia", 2024),
    ("Schley", "Georgia", 2024),
    ("Baltimore City", "Maryland", 2019),
    ("Slope", "North Dakota", 2021),
]
VACC_QC_DROP_REASON = "vaccine row removed after manual QC of the compiled vaccine sheet (specific reason not recorded in the repo)"
VACC_COMPILED = (
    "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied_AK_CA_Updated.xlsx"
)
xl = pl.read_excel(os.path.join(data_dir, VACC_COMPILED))
xl = xl.rename({c: c.lower() for c in xl.columns})
xl = xl.with_columns(
    county=pl.col("county")
    .str.replace_all("Mcdonough", "McDonough")
    .str.replace_all("Dupage", "DuPage")
    .str.replace_all("Mcculloch", "McCulloch"),
    year=pl.col("year").cast(pl.Int64),
    me_raw_null=pl.col("med_exempt_pct").is_null(),
    nme_f=pl.col("nonmed_exempt_pct").cast(pl.Float64, strict=False),
    mmr_f=pl.col("mmr_cov").cast(pl.Float64, strict=False),
    all_f=pl.col("all_vacc_cov").cast(pl.Float64, strict=False),
).select("state", "county", "year", "me_raw_null", "nme_f", "mmr_f", "all_f")
xl_key = ["state", "county", "year"]
n_dup_xl = xl.height - xl.select(xl_key).n_unique()
print(
    f"[info] {n_dup_xl} duplicate (state, county, year) rows in the compiled sheet; the first is used"
)
xl = xl.group_by(xl_key).agg(pl.all().first())

# ---------------------------------------------------------------------------
# 3. Per-county-year status, origins, flags
# ---------------------------------------------------------------------------
m = cy.join(
    rm.drop("cases", "indigenous", "imported", strict=False),
    on=KEY,
    how="left",
)
m = m.with_columns(vyear=pl.col("year") - 1)
m = m.join(
    xl.rename({"year": "vyear"}), on=["state", "county", "vyear"], how="left"
)
m = m.join(
    imp.select(
        KEY
        + [
            "fips",
            "imputation_source",
            "cases_outbreak",
            "imported",
            "neighbors_coverage",
        ]
        + ["oddities_2025", "mega_region_2025"]
        + [
            "estimated_mmr_cov_5yr",
            "nonmed_exempt_pct_5yr",
            "med_exempt_pct_5yr",
        ]
        + ["pop", "fraction_under_5", "fraction_school_aged", "avg_house_size"]
        + ["std_house_size", "fraction_public", "fraction_white"]
        + ["fraction_graduate_degree", "log_population_density", "log_income"]
        + [c for c in MEAS]
    ).rename({c: c + "_final" for c in MEAS}),
    on=KEY,
    how="left",
    suffix="_imp",
)
in_prim = m.join(
    prim.select(KEY).with_columns(p=pl.lit(True)), on=KEY, how="left"
)["p"].fill_null(False)
m = m.with_columns(
    in_primary_models=in_prim,
    in_imputed_sensitivity=pl.col("imputation_source").is_not_null() | in_prim,
)
assert m["in_primary_models"].sum() == prim.height
assert m["in_imputed_sensitivity"].sum() == imp.height
# the pipeline's measure columns at merge time (pre-imputation) tell us what was missing
miss = [
    f
    for f, c in [
        ("MMR", "estimated_mmr_cov"),
        ("NME", "nonmed_exempt_pct"),
        ("ME", "med_exempt_pct"),
    ]
]
m = m.with_columns(
    fields_missing_at_aligned_year=pl.concat_str(
        [
            pl.when(pl.col("estimated_mmr_cov").is_null()).then(pl.lit("MMR")),
            pl.when(pl.col("nonmed_exempt_pct").is_null()).then(pl.lit("NME")),
            pl.when(pl.col("med_exempt_pct").is_null()).then(pl.lit("ME")),
        ],
        separator="+",
        ignore_nulls=True,
    )
)

# --- origin labels ---------------------------------------------------------
ORIGIN_IMPUTE = {
    "year+0": "imputed_same_county_FOLLOWING_school_year",
    "year-2": "imputed_same_county_EARLIER_school_year",
    "state_cdc": "imputed_state_cdc_value",
    "state_cdc_zero": "imputed_set_to_zero_no_state_ME",
}
ZERO_ME_STATES = ["Wyoming", "Louisiana", "Vermont", "Mississippi"]


def field_token(src, field):
    """step used to impute `field` ('mmr'/'nonmed'/'med'), or None"""
    if src is None:
        return None
    for part in src.split("; "):
        f, step = part.split(":", 1)
        if f == field:
            return step
    return None


def origin(row, field, col):
    """field in {'mmr','nonmed','med'}; col = pipeline column"""
    step = field_token(row["imputation_source"], field)
    if step is not None:
        return ORIGIN_IMPUTE[step]
    if row[col] is None:
        return "MISSING_not_imputed"  # dropped county-years only
    county = row["county"]
    if county in AGGREGATED_REGIONS:
        return "observed_popweighted_average_of_member_counties"
    if county in UNMAPPED_REGIONS:
        return "MISSING_not_imputed"
    if row["county_type"] == "Economic Region":
        return "observed_reporting_region_applied_to_members"
    if row["state"] == "Delaware":
        return "observed_state_level_applied_to_county"
    if row["state"] == "Kansas":
        return "observed_county_kansas_separate_query"
    if field == "mmr" and (row["state"], county, row["vyear"]) == (
        "Tennessee",
        "Cocke",
        2024,
    ):
        return "observed_manually_corrected_to_94.5"
    if (
        field == "med"
        and row["state"] in ZERO_ME_STATES
        and row["me_raw_null"]
    ):
        return "observed_blank_set_to_zero_state_rule"
    if row["state"] == "Oklahoma":
        xlv = {"mmr": row["mmr_f"], "nonmed": row["nme_f"]}.get(field)
        if field != "med" and xlv is None:
            return "observed_county_filled_from_Oklahoma_xlsx"
    return "observed_county_aligned_year"


def mmr_measure(row):
    src = field_token(row["imputation_source"], "mmr")
    if src is not None:
        return {"state_cdc": "CDC_state_MMR"}.get(
            src, "inherited_from_donor_row"
        )
    if row["estimated_mmr_cov"] is None:
        return None
    if row["county"] in AGGREGATED_REGIONS:
        return "regional_average_of_MMR_or_UTD"
    if row["mmr_cov"] is not None and row["mmr_cov"] != 0:
        return "MMR"
    if row["all_vacc_cov"] is not None:
        return "all_vaccine_UTD_fallback"
    return "MMR (regional/other)"


def detail(row):
    """human-readable imputation description"""
    src = row["imputation_source"]
    if src is None:
        return None
    y, v = row["year"], row["vyear"]
    names = {
        "mmr": "MMR coverage",
        "nonmed": "nonmedical exemption",
        "med": "medical exemption",
    }
    out = []
    for part in src.split("; "):
        f, step = part.split(":", 1)
        if step == "year+0":
            how = f"same county, school year {sy(y)} (ONE YEAR AFTER the aligned school year {sy(v)})"
        elif step == "year-2":
            how = f"same county, school year {sy(y - 2)} (ONE YEAR BEFORE the aligned school year {sy(v)})"
        elif step == "state_cdc" and f == "nonmed":
            how = f"CDC state value, school year {sy(v)}, 'Non-Medical Exemption' (Any Exemption minus Medical Exemption where CDC has no Non-Medical figure)"
        elif step == "state_cdc":
            how = f"CDC state value, school year {sy(v)}"
        else:
            how = "set to 0 (CDC state value has no medical-exemption figure)"
        out.append(f"{names[f]}: {how}")
    return " | ".join(out)


m_rows = m.to_dicts()
for r in m_rows:
    r["mmr_origin"] = origin(r, "mmr", "estimated_mmr_cov")
    r["nme_origin"] = origin(r, "nonmed", "nonmed_exempt_pct")
    r["me_origin"] = origin(r, "med", "med_exempt_pct")
    r["mmr_measure"] = mmr_measure(r)
    r["imputation_detail"] = detail(r)
    r["imputed_fields"] = (
        None
        if r["imputation_source"] is None
        else "+".join(
            {"mmr": "MMR", "nonmed": "NME", "med": "ME"}[p.split(":")[0]]
            for p in r["imputation_source"].split("; ")
        )
    )
m = pl.DataFrame(m_rows, infer_schema_length=None)

# --- county-year level columns (what the models saw) -----------------------
fy = pl.col("vyear")
m = m.with_columns(
    vaccine_school_year=pl.format("{}-{}", fy, (fy + 1) % 100).str.replace(
        r"-(\d)$", r"-0$1"
    ),
    fips_5digit=pl.when(pl.col("fips").str.len_chars() == 5)
    .then(pl.col("fips"))
    .when(pl.col("fips_imp").str.len_chars() == 5)
    .then(pl.col("fips_imp")),
    model_fips_field=pl.col("fips_imp"),
    geography_type=pl.when(
        pl.col("county").is_in(
            AGGREGATED_REGIONS + UNMAPPED_REGIONS + ["Gulf Coast"]
        )
        | (pl.col("county_type") == "Economic Region")
        | (
            (pl.col("fips_imp").str.len_chars() == 2)
            & (pl.col("county") != "State")
        )
        | (pl.col("county") == "State")  # Delaware: whole state pooled
    )
    .then(pl.lit("multi_county_region"))
    .otherwise(pl.lit("county_or_equivalent")),
    region_member_fips=pl.col("county").replace_strict(
        {
            **{k: ";".join(v) for k, v in REGIONS.items()},
            "State": "10001;10003;10005",
        },
        default=None,
    ),
    model_imported=pl.col("imported_raw").fill_null(0),
    model_indigenous=pl.col("indigenous_raw").fill_null(0),
    mmr_cov_used=pl.coalesce("estimated_mmr_cov_final", "estimated_mmr_cov"),
    nme_pct_used=pl.coalesce("nonmed_exempt_pct_final", "nonmed_exempt_pct"),
    me_pct_used=pl.coalesce("med_exempt_pct_final", "med_exempt_pct"),
    import_reporting_status=pl.when(
        pl.col("imported_raw").is_not_null()
        & pl.col("indigenous_raw").is_not_null()
    )
    .then(
        pl.when(
            pl.col("imported_raw") + pl.col("indigenous_raw")
            == pl.col("cases")
        )
        .then(pl.lit("both_reported_sum_to_total"))
        .otherwise(pl.lit("both_reported_do_not_sum_to_total"))
    )
    .when(
        pl.col("imported_raw").is_not_null()
        | pl.col("indigenous_raw").is_not_null()
    )
    .then(pl.lit("only_one_reported"))
    .otherwise(pl.lit("not_reported")),
)
m = m.with_columns(
    model_cases_outbreak=pl.col("cases") - pl.col("model_imported")
)
inc = m.filter(pl.col("in_imputed_sensitivity"))
assert (inc["model_cases_outbreak"] == inc["cases_outbreak"]).all()
assert (inc["model_imported"] == inc["imported"]).all()
assert (inc["mmr_cov_used"] == inc["estimated_mmr_cov_final"]).all()
print(
    "[ok] cases_outbreak = cases - imported (blank imported = 0) matches the pipeline"
)

# imputed state-CDC nonmedical-exemption values: CDC's Non-Medical Exemption or,
# for state-years that have only Any Exemption, Any minus Medical (floored at
# 0; Any itself when the state has no Medical figure), as in 00_explore_data.py
cdc = pl.read_csv(
    os.path.join(
        data_dir,
        "Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260526.csv",
    ),
    infer_schema_length=0,
)


def cdc_state_series(dose, name):
    return (
        cdc.filter(
            (pl.col("Geography Type") == "States") & (pl.col("Dose") == dose)
        )
        .select(
            state=pl.col("Geography"),
            vyear=pl.col("School Year").str.slice(0, 4).cast(pl.Int64),
            **{name: pl.col("Estimate (%)").cast(pl.Float64, strict=False)},
        )
        .unique(["state", "vyear"], keep="first", maintain_order=True)
    )


nme_state = m.filter(pl.col("nme_origin") == "imputed_state_cdc_value")
n_nme_state = int(nme_state.height)
nme_state = (
    nme_state.join(
        cdc_state_series("Non-Medical Exemption", "cdc_nme"),
        on=["state", "vyear"],
        how="left",
    )
    .join(
        cdc_state_series("Any Exemption", "cdc_any"),
        on=["state", "vyear"],
        how="left",
    )
    .join(
        cdc_state_series("Medical Exemption", "cdc_med"),
        on=["state", "vyear"],
        how="left",
    )
    .with_columns(
        expected=pl.coalesce(
            "cdc_nme",
            pl.when(pl.col("cdc_med").is_not_null())
            .then(
                pl.max_horizontal(
                    pl.col("cdc_any") - pl.col("cdc_med"), pl.lit(0.0)
                )
            )
            .otherwise(pl.col("cdc_any")),
        )
    )
)
assert nme_state["expected"].null_count() == 0
assert (
    (nme_state["nonmed_exempt_pct_final"] - nme_state["expected"]).abs() < 1e-9
).all()
print(
    f"[ok] {n_nme_state} imputed state NME values equal CDC Non-Medical Exemption (Any minus Medical where only Any exists)"
)

# --- model-subset flags (eligibility, mirroring the filters in 02/03) ------
prev = (
    prim.sort(["state", "county", "year"])
    .with_columns(
        prev=pl.col("cases_outbreak").cum_sum().over(["county", "state"])
        - pl.col("cases_outbreak")
    )
    .select(KEY + [(pl.col("prev") == 0).alias("no_prior_outbreak")])
)
MN_2017 = ["Hennepin", "Crow Wing", "Ramsey", "Le Sueur"]
m = m.join(prev, on=KEY, how="left")
m = m.with_columns(
    period_bucket=pl.when(pl.col("year") <= 2020)
    .then(pl.lit("2011-2020"))
    .otherwise(pl.lit("2021-2025")),
    model_me_le_5=pl.col("in_primary_models") & (pl.col("me_pct_used") <= 5.0),
    model_me_le_10=pl.col("in_primary_models")
    & (pl.col("me_pct_used") <= 10.0),
    model_has_neighbors_coverage=pl.col("in_primary_models")
    & pl.col("neighbors_coverage").is_not_null(),
    model_no_prior_outbreak_2021_2025=pl.col("in_primary_models")
    & pl.col("no_prior_outbreak")
    & (pl.col("year") >= 2021),
    flag_oddities_2025=pl.col("oddities_2025").fill_null(False),
    flag_mega_region_2025=pl.col("mega_region_2025").fill_null(False),
    flag_random_walk_excludes_mn_2017=pl.col("in_primary_models")
    & (pl.col("year") == 2017)
    & (pl.col("state") == "Minnesota")
    & pl.col("county").is_in(MN_2017),
)

# --- status and reason per county-year -------------------------------------
rows = m.to_dicts()
for r in rows:
    sy_aligned = r["vaccine_school_year"]
    if r["in_primary_models"]:
        r["record_status"], r["status_stage"], r["status_reason"] = (
            "INCLUDED_DIRECT",
            None,
            None,
        )
    elif r["in_imputed_sensitivity"]:
        r["record_status"], r["status_stage"], r["status_reason"] = (
            "INCLUDED_IMPUTED",
            None,
            None,
        )
    else:
        r["record_status"] = "DROPPED"
        r["status_stage"] = "vaccine_data_missing_not_imputable"
        if r["geography_type"] == "multi_county_region":
            r["status_reason"] = (
                f"multi-county reporting region with incomplete vaccine data at school year "
                f"{sy_aligned} (missing: {r['fields_missing_at_aligned_year']}); no region-level imputation was possible"
            )
        else:
            r["status_reason"] = (
                f"vaccine data incomplete at aligned school year {sy_aligned} (missing: "
                f"{r['fields_missing_at_aligned_year']}); imputation (same county in school years "
                f"{sy(r['year'])} / {sy(r['year'] - 2)}, then CDC state value) did not complete all three fields"
            )
m = pl.DataFrame(rows, infer_schema_length=None)

QC_NOTES = pl.DataFrame(
    [
        {
            "state": s,
            "county": c,
            "vyear": y - 0,
            "vaccine_qc_note": VACC_QC_DROP_REASON,
        }
        for c, s, y in VACC_QC_DROPS
    ]
    + [
        {
            "state": "Tennessee",
            "county": "Cocke",
            "vyear": 2024,
            "vaccine_qc_note": "MMR coverage for this county/school year manually set to 94.5",
        }
    ]
)
m = m.drop("vaccine_qc_note", strict=False).join(
    QC_NOTES, on=["state", "county", "vyear"], how="left"
)

# ---------------------------------------------------------------------------
# 4. Assemble the record-level master
# ---------------------------------------------------------------------------
COLS = [
    # --- which record is this
    (
        "record_id",
        "Unique id of the raw case record: SOURCE:line, where line is the line number in that file (header = line 1).",
    ),
    ("source_file", "Case file the record came from."),
    ("year_source", "Year exactly as in the source file."),
    ("county_source", "County/place exactly as in the source file."),
    ("year", "Outbreak year after cleaning."),
    ("state", "State."),
    (
        "county",
        "County/place after cleaning (name fixes; Delaware pooled to 'State'; Detroit->Wayne; New York City->Kings).",
    ),
    (
        "geography_type",
        "county_or_equivalent | multi_county_region (one value spans several counties; includes Delaware pooled to the whole state).",
    ),
    (
        "fips_5digit",
        "5-digit county FIPS; blank for regions and state-level rows.",
    ),
    (
        "model_fips_field",
        "The 'fips' value in analysis_dataset*.csv (a place name for regions; 2-digit for some regions/Delaware). Only for rows in the imputed analysis file.",
    ),
    (
        "region_member_fips",
        "For multi-county regions, the member county FIPS (semicolon list) the analysis expands the region to.",
    ),
    (
        "recode_notes",
        "Every change applied to this record before analysis (blank = none).",
    ),
    # --- what happened to it
    (
        "record_status",
        "INCLUDED_DIRECT = in the primary models; vaccine data observed at the aligned school year. INCLUDED_IMPUTED = only in the imputed sensitivity analysis; >=1 vaccine field imputed. DROPPED = in neither.",
    ),
    (
        "status_stage",
        "For DROPPED records: the step that removed it (see README).",
    ),
    ("status_reason", "For DROPPED records: why."),
    (
        "county_year_n_records",
        "How many raw records were summed into this county-year after cleaning (>1 only where cleaning merged names, e.g. Detroit->Wayne).",
    ),
    (
        "county_year_representative",
        "TRUE on exactly one record per county-year. Filter on it when counting county-years or summing model_* columns (which repeat on merged records).",
    ),
    # --- cases
    (
        "cases_source",
        "This record's case count exactly as in its source file.",
    ),
    (
        "cases_used",
        "This record's case count after QC overrides (the 2025 QC sheet's verified count replaces the source count).",
    ),
    (
        "imported_reported",
        "This record's imported count as reported (blank = not reported). Definition of 'imported' is not documented in the repo.",
    ),
    (
        "indigenous_reported",
        "This record's indigenous (locally acquired) count as reported (blank = not reported).",
    ),
    ("model_cases", "County-year total cases (what the models saw)."),
    (
        "model_imported",
        "County-year imported cases as the pipeline stored them (blank treated as 0).",
    ),
    (
        "model_indigenous",
        "County-year indigenous cases as the pipeline stored them (blank treated as 0).",
    ),
    (
        "model_cases_outbreak",
        "MODEL RESPONSE: model_cases - model_imported. Counts cases not reported as imported (locally acquired plus unclassified).",
    ),
    (
        "import_reporting_status",
        "both_reported_sum_to_total | both_reported_do_not_sum_to_total | only_one_reported | not_reported. 'not_reported' means model_imported=0 reflects a blank, not a reported zero.",
    ),
    ("case_sources", "Case file(s) behind the county-year."),
    # --- vaccine/exemption values the models used
    (
        "vaccine_school_year",
        "School year of the vaccine/exemption data matched to this outbreak year (outbreak year Y uses school year Y-1; e.g. outbreak 2024 -> '2023-24').",
    ),
    ("mmr_cov_used", "MMR coverage (%) used by the models (kindergarten)."),
    (
        "nme_pct_used",
        "Nonmedical exemption prevalence (%) used by the models (kindergarten).",
    ),
    (
        "me_pct_used",
        "Medical exemption prevalence (%) used by the models (kindergarten).",
    ),
    (
        "mmr_measure",
        "What mmr_cov_used is: MMR | all_vaccine_UTD_fallback (MMR blank/0, all-vaccine up-to-date used) | CDC_state_MMR | inherited_from_donor_row (imputed from another year; measure not tracked) | regional_average_of_MMR_or_UTD.",
    ),
    ("mmr_origin", "How mmr_cov_used was obtained (vocabulary in README)."),
    ("nme_origin", "How nme_pct_used was obtained."),
    ("me_origin", "How me_pct_used was obtained."),
    ("imputed_fields", "Which of MMR / NME / ME were imputed (blank = none)."),
    (
        "imputation_detail",
        "Plain-language description of each imputation, including donor school year.",
    ),
    ("imputation_source_raw", "The pipeline's own imputation_source string."),
    (
        "fields_missing_at_aligned_year",
        "Fields blank at the aligned school year before any imputation.",
    ),
    (
        "vaccine_qc_note",
        "Set if the vaccine row for this county/school year was removed or edited in QC.",
    ),
    (
        "mmr_cov_raw",
        "Raw 'MMR_cov' from the vaccine sheet at the aligned year (blank if absent).",
    ),
    (
        "all_vacc_cov_raw",
        "Raw all-vaccine up-to-date coverage at the aligned year (blank if absent).",
    ),
    # --- model use
    (
        "in_primary_models",
        "TRUE if the county-year is a row of analysis_dataset.csv.",
    ),
    (
        "in_imputed_sensitivity",
        "TRUE if the county-year is a row of analysis_dataset_imputed.csv.",
    ),
    (
        "period_bucket",
        "Fit period bucket in 02_fit_hierarchical_poisson.py: 2011-2020 or 2021-2025.",
    ),
    (
        "model_me_le_5",
        "Row is eligible for the medical-exemption <=5% model variant (primary rows only).",
    ),
    (
        "model_me_le_10",
        "Row is eligible for the medical-exemption <=10% model variant (primary rows only).",
    ),
    (
        "model_has_neighbors_coverage",
        "Row has neighbors_coverage, required by the neighbor-coverage variants (primary rows only).",
    ),
    (
        "model_no_prior_outbreak_2021_2025",
        "Row is in the 'no previous outbreak, 2021-2025' variant (primary rows only).",
    ),
    (
        "flag_oddities_2025",
        "2025 row with <20 outbreak cases; excluded in the 'no 2025 oddities' variants.",
    ),
    (
        "flag_mega_region_2025",
        "2025 multi-county region; excluded in the 'no 2025 mega regions' variants.",
    ),
    (
        "flag_random_walk_excludes_mn_2017",
        "Minnesota 2017 row excluded in the random-walk variant (03_random_walk_poisson.py).",
    ),
    (
        "estimated_mmr_cov_5yr",
        "Five-year-average MMR coverage (school years Y-5..Y-1, population-weighted; blank for imputed rows).",
    ),
    (
        "nonmed_exempt_pct_5yr",
        "Five-year-average nonmedical exemption (same definition).",
    ),
    (
        "med_exempt_pct_5yr",
        "Five-year-average medical exemption (same definition).",
    ),
    (
        "neighbors_coverage",
        "Mean MMR coverage of adjacent counties in school year Y-1.",
    ),
    ("pop", "County population."),
    ("fraction_under_5", "Covariate in the 'All covariates' model."),
    ("fraction_school_aged", "Covariate in the 'All covariates' model."),
    ("avg_house_size", "Covariate in the 'All covariates' model."),
    ("std_house_size", "Covariate in the 'All covariates' model."),
    ("fraction_public", "Covariate in the 'All covariates' model."),
    ("fraction_white", "Covariate in the 'All covariates' model."),
    ("fraction_graduate_degree", "Covariate in the 'All covariates' model."),
    (
        "log_population_density",
        "log10 of people per square km; covariate in the 'All covariates' model.",
    ),
    ("log_income", "Covariate in the 'All covariates' model."),
]
COL_NAMES = [c for c, _ in COLS]

dropped_rec = rec.filter(pl.col("drop_stage").is_not_null()).select(
    "record_id",
    "source_file",
    "year_source",
    "county_source",
    "year",
    "state",
    "county",
    "recode_notes",
    "cases_source",
    cases_used=pl.col("cases"),
    imported_reported=pl.col("imported"),
    indigenous_reported=pl.col("indigenous"),
    record_status=pl.lit("DROPPED"),
    status_stage=pl.col("drop_stage"),
    status_reason=pl.col("drop_reason"),
)
alive_rec = (
    rec.filter(pl.col("drop_stage").is_null())
    .sort("record_id")
    .select(
        "record_id",
        "source_file",
        "year_source",
        "county_source",
        "year",
        "state",
        "county",
        "recode_notes",
        "cases_source",
        cases_used=pl.col("cases"),
        imported_reported=pl.col("imported"),
        indigenous_reported=pl.col("indigenous"),
    )
    .with_columns(
        county_year_n_records=pl.len().over(KEY),
        county_year_representative=pl.col("record_id")
        .rank("ordinal")
        .over(KEY)
        == 1,
    )
)
m = m.rename(
    {
        "imputation_source": "imputation_source_raw",
        "mmr_cov": "mmr_cov_raw",
        "all_vacc_cov": "all_vacc_cov_raw",
    }
)
cy_cols = [
    c
    for c in COL_NAMES
    if c in m.columns
    and c not in alive_rec.columns
    and c not in dropped_rec.columns
] + ["record_status", "status_stage", "status_reason", "cases"]
m_out = m.select(KEY + [c for c in dict.fromkeys(cy_cols)]).rename(
    {"cases": "model_cases"}
)
alive_rec = alive_rec.join(m_out, on=KEY, how="left", validate="m:1")
bool_cols = [
    "in_primary_models",
    "in_imputed_sensitivity",
    "model_me_le_5",
    "model_me_le_10",
    "model_has_neighbors_coverage",
    "model_no_prior_outbreak_2021_2025",
    "flag_oddities_2025",
    "flag_mega_region_2025",
    "flag_random_walk_excludes_mn_2017",
]
out = pl.concat([alive_rec, dropped_rec], how="diagonal_relaxed").with_columns(
    pl.col(bool_cols).fill_null(False)
)
out = out.with_columns(
    _rank=pl.col("record_status").replace_strict(
        {"INCLUDED_DIRECT": 0, "INCLUDED_IMPUTED": 1, "DROPPED": 2}
    )
).sort(["_rank", "state", "county", "year", "record_id"], nulls_last=True)
out = out.select(COL_NAMES)

# --- checks on the master ---------------------------------------------------
assert out.height == rec.height and out["record_id"].is_unique().all()
assert set(out["record_status"].unique()) == {
    "INCLUDED_DIRECT",
    "INCLUDED_IMPUTED",
    "DROPPED",
}
d = out.filter(pl.col("record_status") == "DROPPED")
i = out.filter(pl.col("record_status") != "DROPPED")
assert (
    d["status_stage"].is_not_null().all()
    and d["status_reason"].is_not_null().all()
)
assert i["status_stage"].is_null().all() and i["status_reason"].is_null().all()
rep = out.filter(pl.col("county_year_representative"))
assert (
    rep.filter(pl.col("record_status") == "INCLUDED_DIRECT").height
    == prim.height
)
assert rep.filter(pl.col("record_status") != "DROPPED").height == imp.height
assert rep.filter(pl.col("in_primary_models")).height == prim.height
# included values equal the pipeline's analysis file, county-year by county-year
chk = rep.filter(pl.col("in_imputed_sensitivity")).join(
    imp, on=KEY, suffix="_p", validate="1:1"
)
for a, b in [
    ("mmr_cov_used", "estimated_mmr_cov"),
    ("nme_pct_used", "nonmed_exempt_pct"),
    ("me_pct_used", "med_exempt_pct"),
    ("model_cases", "cases"),
    ("model_cases_outbreak", "cases_outbreak"),
    ("model_imported", "imported"),
]:
    assert chk[a].eq_missing(chk[b]).all(), f"{a} != analysis dataset {b}"
print(
    f"[ok] master: {out.height} records; "
    f"{prim.height} direct + {imp.height - prim.height} imputed county-years match the analysis files exactly"
)

# ---------------------------------------------------------------------------
# 5. Write outputs
# ---------------------------------------------------------------------------
# Records outside the study period and the sheets' total rows are not outbreaks; they are
# left out of both masters (still counted in row_flow.csv and listed in change_log.csv).
# `out` keeps all records for those two files; `out_full` is what the masters are built from.
REMOVED_FROM_MASTERS = ["case_outside_study_period", "case_not_a_record"]
out_full = out.filter(
    ~pl.col("status_stage").is_in(REMOVED_FROM_MASTERS).fill_null(False)
)
out_full.write_csv(os.path.join(out_dir, "outbreak_records_master_full.csv"))
print("wrote outbreak_records_master_full.csv", out_full.shape)

# --- the reviewer-facing main file (trimmed, plain-language columns) --------
county_shapes = gpd.read_file(
    os.path.join(data_dir, "cb_2020_us_county_20m.shp")
)
lsad = dict(zip(county_shapes["GEOID"], county_shapes["LSAD"]))
donor = {
    (r["state"], r["county"], r["year"]): (r["mmr_f"], r["all_f"])
    for r in xl.iter_rows(named=True)
}
addfips = AddFIPS()  # local name lookup; same method 00_explore_data.py uses for missing-vaccine rows

SOURCE = {  # origin (full file) -> whose figure the value is
    "observed_county_aligned_year": "county",
    "observed_county_kansas_separate_query": "county",
    "observed_county_filled_from_Oklahoma_xlsx": "county",
    "observed_manually_corrected_to_94.5": "county",
    "imputed_same_county_FOLLOWING_school_year": "county",
    "imputed_same_county_EARLIER_school_year": "county",
    "observed_popweighted_average_of_member_counties": "region",
    "observed_reporting_region_applied_to_members": "region",
    "observed_state_level_applied_to_county": "state",  # Delaware: its own whole-state figure (observed, not imputed)
    "imputed_state_cdc_value": "state",  # CDC state estimate substituted for a missing county value
    "observed_blank_set_to_zero_state_rule": "assumed_zero",
    "imputed_set_to_zero_no_state_ME": "assumed_zero",
    "MISSING_not_imputed": None,  # no value: source left blank
}
# which school year a value comes from, as a shift from the matched year (outbreak year - 1);
# None = no data year (assumed zero or missing)
SCHOOL_YEAR_SHIFT = {
    "imputed_same_county_FOLLOWING_school_year": 1,
    "imputed_same_county_EARLIER_school_year": -1,
    "observed_blank_set_to_zero_state_rule": None,
    "imputed_set_to_zero_no_state_ME": None,
    "MISSING_not_imputed": None,
}
SPECIAL_NOTE = {
    "observed_popweighted_average_of_member_counties": "vaccine values are a population-weighted average of the region's member counties",
    "observed_reporting_region_applied_to_members": "vaccine values are for the reporting region (Alaska economic region) and apply to all its areas",
    "observed_state_level_applied_to_county": None,  # special, but the Delaware pooling note already says why
    "observed_manually_corrected_to_94.5": "MMR coverage manually set to 94.5",
    "observed_blank_set_to_zero_state_rule": "medical exemption blank in source; set to 0 (rule applied to WY/LA/VT/MS)",
    "observed_county_kansas_separate_query": "Kansas values come from a separate county-level query, not the compiled vaccine sheet",
    "observed_county_filled_from_Oklahoma_xlsx": "a missing value was filled from Oklahoma.xlsx",
}
MEASURE = {
    "MMR": "MMR",
    "all_vaccine_UTD_fallback": "all_vaccines_up_to_date",
    "CDC_state_MMR": "MMR",
}


# measure each county-year of the compiled sheet gives: MMR, unless blank/0, then all-vaccines (same rule as process_vaccinations_df);
# used to label multi-county regions by what their member counties actually used
_sheet = (
    pl.read_excel(os.path.join(data_dir, VACC_COMPILED))
    .rename(str.lower)
    .with_columns(
        year=pl.col("year").cast(pl.Int64),
        mmr_cov=pl.col("mmr_cov").cast(pl.Float64, strict=False),
        all_vacc_cov=pl.col("all_vacc_cov").cast(pl.Float64, strict=False),
    )
)
member_measure = {
    (str(r["fips"]).zfill(5), r["year"]): (
        "MMR"
        if (r["mmr_cov"] is not None and r["mmr_cov"] != 0)
        else (
            "all_vaccines_up_to_date"
            if r["all_vacc_cov"] is not None
            else None
        )
    )
    for r in _sheet.iter_rows(named=True)
    if r["fips"]
}


def region_member_measures(r):
    """measure used by each member county of a multi-county region (sheet school year = outbreak year - 1); None = no value"""
    return [
        member_measure.get((f, r["year"] - 1))
        for f in r["region_member_fips"].split(";")
    ]


def region_measure(r):
    kinds = set(region_member_measures(r)) - {None}
    if kinds == {"MMR"}:
        return "MMR"
    if kinds == {"all_vaccines_up_to_date"}:
        return "all_vaccines_up_to_date"
    return "mixed_MMR_and_all_vaccines" if kinds else "unknown"


def region_note(r):
    kinds = region_member_measures(r)
    n, n_missing = len(kinds), kinds.count(None)
    k = {x for x in kinds if x}
    what = {
        "MMR": "MMR coverage",
        "all_vaccines_up_to_date": "all-vaccines coverage (MMR blank or 0)",
    }
    if len(k) == 1:
        w = n - n_missing
        who = (
            "the only member with data uses"
            if w == 1
            else f"all {w} members with data use"
        )
        text = f"Region value averages member-county coverage: {who} {what[next(iter(k))]}"
    else:
        text = (
            f"Region value averages member-county coverage: {kinds.count('MMR')} members use MMR coverage and "
            f"{kinds.count('all_vaccines_up_to_date')} use all-vaccines coverage"
        )
    return text + (
        f" ({n_missing} of {n} members have no value)" if n_missing else ""
    )


def coverage_measure(r):
    mm = r["mmr_measure"]
    if mm is None:
        return None
    if mm == "regional_average_of_MMR_or_UTD":
        return region_measure(r)
    if (
        mm == "inherited_from_donor_row"
    ):  # look up which measure the donor year's row used
        dy = r["year"] if "FOLLOWING" in r["mmr_origin"] else r["year"] - 2
        mmr_f, all_f = donor.get((r["state"], r["county"], dy), (None, None))
        if mmr_f is not None and mmr_f != 0:
            return "MMR"
        return "all_vaccines_up_to_date" if all_f is not None else "unknown"
    return MEASURE.get(mm, "unknown")


USE = {  # labels used in row_flow.csv and the README
    "INCLUDED_DIRECT": "used (vaccine data observed)",
    "INCLUDED_IMPUTED": "used (vaccine data imputed)",
    "DROPPED": "dropped",
}
DROP_LABEL = {  # internal step -> plain-language inference_excluded_reason shown in the main file and row_flow.csv
    "vaccine_data_missing_not_imputable": "no vaccine data",
    "case_location_unusable": "county not identified",
    "case_report_not_verified": "could not verify case report",
    "case_zero_or_missing": "zero cases reported (no outbreak)",
    "case_outside_study_period": "before 2011 (outside study period)",
    "case_not_a_record": "total row in source sheet",
}


def plain_drop_reason(stage, state, county):
    if stage is None:
        return None
    if (
        stage == "case_outbreak_removed"
    ):  # the three outbreaks left out of the analysis
        return (
            "outbreak in ICE detention facility"
            if (state, county) == ("Arizona", "Pinal")
            else "outbreak among non-US population"
        )
    return DROP_LABEL[stage]


REMOVED_OUTBREAK_NOTE = {  # (state, county, year) -> note for the three outbreaks left out of the analysis
    (
        "Illinois",
        "Cook",
        2024,
    ): "Outbreak among non-US population (migrant shelter): https://doi.org/10.15585/mmwr.mm7319a1",
    (
        "Wisconsin",
        "Monroe",
        2021,
    ): "Outbreak among non-US population (Afghan evacuees, Operation Allies Welcome): https://doi.org/10.1016/S2468-2667(23)00130-5",
    (
        "Arizona",
        "Pinal",
        2016,
    ): "Outbreak in an ICE detention facility: https://doi.org/10.1093/cid/ciy819",
}
# the four county-years in exclude_16_17 (analysis/03_random_walk_poisson.py:56-60)
MN_2017_COUNTIES = {"Hennepin", "Ramsey", "Crow Wing", "Le Sueur"}
MN_2017_NOTE = (
    "2017 Minnesota measles outbreak, a clustered outbreak largely among under-vaccinated Somali-American children "
    "(75 cases across Hennepin, Ramsey, Crow Wing and Le Sueur counties, "
    "Banerjee et al., Perspect Public Health 2019, PMID 31480896, https://doi.org/10.1177/1757913919871072). "
    "Included in all analyses except one version of the time-trend (random-walk) models of how the MMR coverage effect "
    "changes over time, which leaves out these four county-years. Figure 2a averages the versions with and without them, "
    "and Figure S2 (panels e-f) shows the version without them. The exclusion applies only to the time-trend analysis"
)


def imports_reported_or_assumed_zero(r):
    """'reported' if the case source gave an imported count; otherwise it was blank and counted as 0"""
    return "reported" if r["imported_reported"] is not None else "assumed_zero"


def data_school_year(origin, vyear):
    """school year (e.g. '2014-15') that a value actually comes from; vyear = matched year (outbreak year - 1)"""
    shift = SCHOOL_YEAR_SHIFT.get(origin, 0)
    return (
        None
        if (origin in SCHOOL_YEAR_SHIFT and shift is None)
        else sy(vyear + shift)
    )


main_rows = []
for r in out_full.iter_rows(named=True):
    if r["status_stage"] == "case_zero_or_missing":
        continue  # state-by-year placeholder rows: not outbreaks (counted in row_flow.csv)
    has_vax = r["vaccine_school_year"] is not None
    region = r["geography_type"] == "multi_county_region"
    if r["status_stage"] == "case_location_unusable":
        place, cfips = "state_only", None
    elif not has_vax:
        place, cfips = None, None
    elif region:
        place, cfips = "multi_county_region", None
    else:
        cfips = r["fips_5digit"] or addfips.get_county_fips(
            r["county"], r["state"]
        )
        assert cfips and cfips in lsad, (
            f"no FIPS/type for {r['county']}, {r['state']}"
        )
        place = "county" if lsad[cfips] == "06" else "county_equivalent"
    origins = [r["mmr_origin"], r["nme_origin"], r["me_origin"]]
    measure = coverage_measure(r) if has_vax else None
    dropped = r["record_status"] == "DROPPED"
    notes = []
    detroit_cases = 0
    if r["source_file"] == SRC_SC:
        notes.append(
            "South Carolina 2025 cases include cases until the end of January 2026"
        )
    for clause in (r["recode_notes"] or "").split("; "):
        if clause.startswith("Delaware county"):
            notes.append(clause)
        elif "Detroit aggregated" in clause:
            detroit_cases = r[
                "cases_used"
            ]  # cases reported in Detroit City, added to Wayne County
    if r["status_stage"] == "case_outbreak_removed":
        notes.append(
            REMOVED_OUTBREAK_NOTE[(r["state"], r["county"], r["year"])]
        )
    if (
        r["state"] == "Minnesota"
        and r["year"] == 2017
        and r["county"] in MN_2017_COUNTIES
    ):
        notes.append(MN_2017_NOTE)
    if place == "state_only":
        notes.append("state-level information only: no county identified")
        if r["county_source"]:
            notes.append(f"source place name was '{r['county_source']}'")
    if has_vax:
        for o in dict.fromkeys(origins):
            if SPECIAL_NOTE.get(o):
                notes.append(SPECIAL_NOTE[o])
        if r["mmr_measure"] == "regional_average_of_MMR_or_UTD":
            notes.append(region_note(r))
        if region and r["record_status"] == "INCLUDED_IMPUTED":
            notes.append("multi-county region filled with CDC state values")
    main_rows.append(
        {
            "year": r["year"],
            "county": r["county"],
            "state": r["state"],
            "place_type": place,
            "fips": cfips,
            "fips_counties_in_region": r["region_member_fips"]
            if region and has_vax
            else None,
            "cases": r["cases_used"],
            "imported": r["imported_reported"]
            if r["imported_reported"] is not None
            else 0,
            "indigenous": r["indigenous_reported"]
            if r["indigenous_reported"] is not None
            else 0,
            "imports_reported_or_assumed_zero": imports_reported_or_assumed_zero(
                r
            ),
            "cases_outbreak": None if dropped else r["model_cases_outbreak"],
            "county_year_used_inference": not dropped,
            "inference_excluded_reason": plain_drop_reason(
                r["status_stage"], r["state"], r["county"]
            )
            if dropped
            else None,
            "vaccine_data_imputed": None
            if dropped
            else r["record_status"] == "INCLUDED_IMPUTED",
            "vaccine_school_year": r["vaccine_school_year"],
            "estimated_mmr_cov": r["mmr_cov_used"],
            "nonmed_exempt_pct": r["nme_pct_used"],
            "med_exempt_pct": r["me_pct_used"],
            "mmr_or_all_vaccine_coverage": measure,
            "estimated_mmr_cov_source": SOURCE[r["mmr_origin"]]
            if has_vax
            else None,
            "nonmed_exempt_pct_source": SOURCE[r["nme_origin"]]
            if has_vax
            else None,
            "med_exempt_pct_source": SOURCE[r["me_origin"]]
            if has_vax
            else None,
            "estimated_mmr_cov_school_year": data_school_year(
                r["mmr_origin"], r["year"] - 1
            )
            if has_vax
            else None,
            "nonmed_exempt_pct_school_year": data_school_year(
                r["nme_origin"], r["year"] - 1
            )
            if has_vax
            else None,
            "med_exempt_pct_school_year": data_school_year(
                r["me_origin"], r["year"] - 1
            )
            if has_vax
            else None,
            "detroit_cases": detroit_cases,  # folded into notes below
            "notes": notes,  # list for now; joined below
            "record_id": r[
                "record_id"
            ],  # sort tie-break only; dropped from the file below
        }
    )

# One row per county-year: source records that cleaning merged (Detroit City -> Wayne County) are
# combined into one row. Only done among non-dropped rows with an identical (year, state, county).
groups = {}
for row in sorted(main_rows, key=lambda x: x["record_id"]):
    groups.setdefault((row["year"], row["state"], row["county"]), []).append(
        row
    )
merged_rows = []
for key, grp in groups.items():
    if len(grp) == 1 or any(not g["county_year_used_inference"] for g in grp):
        merged_rows.extend(grp)
        continue
    row = dict(grp[0])
    for c in ["cases", "imported", "indigenous", "detroit_cases"]:
        row[c] = sum(g[c] for g in grp)
    row["imports_reported_or_assumed_zero"] = (
        "reported"
        if all(
            g["imports_reported_or_assumed_zero"] == "reported" for g in grp
        )
        else "assumed_zero"
    )
    row["notes"] = [
        n for n in dict.fromkeys(n for g in grp for n in g["notes"])
    ]
    merged_rows.append(row)
n_merged_away = len(main_rows) - len(merged_rows)
for row in merged_rows:
    n_det = row.pop("detroit_cases")
    if n_det:
        row["notes"] = row["notes"] + [
            f"Includes {int(n_det)} case{'s' if n_det != 1 else ''} reported in Detroit City, aggregated to Wayne County"
        ]
    row["notes"] = "; ".join(row["notes"]) if row["notes"] else None

main = pl.DataFrame(merged_rows, infer_schema_length=None).with_columns(
    pl.col("cases", "imported", "indigenous", "cases_outbreak").cast(pl.Int64)
)
main = main.sort(["year", "state", "county", "record_id"], nulls_last=True)
n_zero_rows = out_full.filter(
    pl.col("status_stage") == "case_zero_or_missing"
).height
assert (
    main.height == out_full.height - n_zero_rows - n_merged_away
    and main["record_id"].is_unique().all()
)
assert n_merged_away == 2, (
    f"expected the 2 Wayne County MI county-years to merge, got {n_merged_away}"
)
assert (
    main.filter(pl.col("county_year_used_inference"))
    .select(["year", "state", "county"])
    .is_unique()
    .all()
)
assert "unknown" not in set(
    main["mmr_or_all_vaccine_coverage"].drop_nulls()
), "unlabelled MMR measure"
vax = main.filter(pl.col("vaccine_school_year").is_not_null())
assert vax["place_type"].is_not_null().all()
assert set(vax["place_type"].unique()) <= {
    "county",
    "county_equivalent",
    "multi_county_region",
}
# a source label is blank exactly when its value is blank
for val, src in [
    ("estimated_mmr_cov", "estimated_mmr_cov_source"),
    ("nonmed_exempt_pct", "nonmed_exempt_pct_source"),
    ("med_exempt_pct", "med_exempt_pct_source"),
]:
    assert (vax[val].is_null() == vax[src].is_null()).all(), (
        f"{src} blank/non-blank disagrees with {val}"
    )
# scope: study period only, no zero-case placeholders, state-level-only records are all dropped
assert main["year"].is_not_null().all() and (main["year"] >= 2011).all()
assert (main["cases"] > 0).all(), "zero-case row in main file"
so = main.filter(pl.col("place_type") == "state_only")
assert (
    so.height == 11
    and (~so["county_year_used_inference"]).all()
    and so["state"].is_not_null().all()
)
assert (
    main.filter(pl.col("county_year_used_inference"))["vaccine_data_imputed"]
    .is_not_null()
    .all()
)
assert (
    main.filter(~pl.col("county_year_used_inference"))["vaccine_data_imputed"]
    .is_null()
    .all()
)
removed = main.filter(
    pl.col("inference_excluded_reason")
    .str.starts_with("outbreak")
    .fill_null(False)
)
assert removed.height == 3 and removed["notes"].str.contains("doi.org").all()
# source / school-year columns: matched year (outbreak year - 1), or one later/earlier when imputed from the
# same county's neighbouring year; CDC state values are always for the matched year
FOLLOWING, EARLIER = (
    "imputed_same_county_FOLLOWING_school_year",
    "imputed_same_county_EARLIER_school_year",
)
VAL3 = [
    (
        "estimated_mmr_cov_source",
        "estimated_mmr_cov_school_year",
        "mmr_origin",
    ),
    (
        "nonmed_exempt_pct_source",
        "nonmed_exempt_pct_school_year",
        "nme_origin",
    ),
    ("med_exempt_pct_source", "med_exempt_pct_school_year", "me_origin"),
]
v = vax.with_columns(
    _early=(pl.col("year") - 2).map_elements(sy, return_dtype=pl.String),
    _match=(pl.col("year") - 1).map_elements(sy, return_dtype=pl.String),
    _late=pl.col("year").map_elements(sy, return_dtype=pl.String),
)
assert (v["vaccine_school_year"] == v["_match"]).all(), (
    "vaccine_school_year is not outbreak year - 1"
)
imputed_any = None
for src, yr, org in VAL3:
    h = v.filter(pl.col(yr).is_not_null())
    assert (
        (h[yr] == h["_early"]) | (h[yr] == h["_match"]) | (h[yr] == h["_late"])
    ).all(), f"{yr} outside the three school years"
    st = v.filter(pl.col(src) == "state")
    assert (st[yr] == st["vaccine_school_year"]).all(), (
        f"{src}=state but {yr} is not the matched year"
    )
    assert (
        v.filter((pl.col(src) == "assumed_zero") | pl.col(src).is_null())[yr]
        .is_null()
        .all()
    )
    assert (
        v.filter(pl.col(src).is_in(["county", "region", "state"]))[yr]
        .is_not_null()
        .all()
    )
    obs = v.filter(
        pl.col("county_year_used_inference") & ~pl.col("vaccine_data_imputed")
    )
    assert ((obs[src] != "state") | (obs["state"] == "Delaware")).all() and (
        (obs[yr] == obs["vaccine_school_year"]) | obs[yr].is_null()
    ).all()
    n_shift = h.filter(pl.col(yr) != pl.col("vaccine_school_year")).height
    assert (
        n_shift == rep.filter(pl.col(org).is_in([FOLLOWING, EARLIER])).height
    ), f"{yr}: shifted years != imputed-from-neighbouring-year count"
    c = ((pl.col(src) == "state") & (pl.col("state") != "Delaware")) | (
        pl.col(yr) != pl.col("vaccine_school_year")
    ).fill_null(False)
    imputed_any = c if imputed_any is None else (imputed_any | c)
# the stated rule (a value is imputed if its source is 'state' (other than Delaware's own figure) or its school year differs from the matched year)
# reproduces vaccine_data_imputed exactly
rule = v.filter(pl.col("county_year_used_inference")).with_columns(
    _rule=imputed_any
)
assert (rule["_rule"] == rule["vaccine_data_imputed"]).all(), (
    "imputed rule disagrees with vaccine_data_imputed"
)
dal = main.filter(
    (pl.col("state") == "Texas")
    & (pl.col("county") == "Dallas")
    & (pl.col("year") == 2014)
)
assert (
    dal["estimated_mmr_cov_source"][0] == "county"
    and dal["estimated_mmr_cov_school_year"][0] == "2014-15"
    and dal["vaccine_school_year"][0] == "2013-14"
)
print(
    "[ok] main file scope, state-only labelling, merged Wayne rows and data-school-year columns"
)
main.drop("record_id").write_csv(
    os.path.join(out_dir, "outbreak_records_master.csv")
)
print("wrote outbreak_records_master.csv", main.drop("record_id").shape)

# --- the written main file agrees with analysis_dataset*.csv ----------------
CROSSWALK = [  # (column, note): same name and same values as in analysis_dataset.csv
    ("year", "outbreak year"),
    ("state", ""),
    ("county", "same text, including region names"),
    (
        "fips",
        "identical for single counties; blank here for regions (analysis_dataset holds a region name or 2-digit code there)",
    ),
    (
        "cases",
        "one row per county-year here and there (the 2 Wayne County MI county-years combine two source records each)",
    ),
    ("imported", "blank in the source = 0 in both"),
    ("indigenous", "locally acquired; blank in the source = 0 in both"),
    ("cases_outbreak", "model response = cases - imported"),
    (
        "estimated_mmr_cov",
        "kindergarten MMR (or all-vaccine) coverage, percent",
    ),
    ("nonmed_exempt_pct", "kindergarten nonmedical exemption, percent"),
    ("med_exempt_pct", "kindergarten medical exemption, percent"),
]
written = pl.read_csv(
    os.path.join(out_dir, "outbreak_records_master.csv"),
    schema_overrides={"fips": pl.String},
)
MK = ["year", "state", "county"]
used = written.filter(pl.col("county_year_used_inference"))
PAIRS = [
    "cases",
    "imported",
    "indigenous",
    "cases_outbreak",
    "estimated_mmr_cov",
    "nonmed_exempt_pct",
    "med_exempt_pct",
]
for fname, ref, mine in [
    ("analysis_dataset_imputed.csv", imp, used),
    (
        "analysis_dataset.csv",
        prim,
        used.filter(~pl.col("vaccine_data_imputed")),
    ),
]:
    assert (
        mine.join(ref, on=MK, how="anti").height == 0
        and ref.join(mine, on=MK, how="anti").height == 0
    ), f"row sets differ from {fname}"
    j = mine.join(ref, on=MK, suffix="_a", validate="1:1")
    for c in PAIRS:
        same = (j[c] == j[c + "_a"]) | (j[c].is_null() & j[c + "_a"].is_null())
        assert same.all(), (
            f"{c} != {fname}:{c} in {(~same).sum()} county-years"
        )
    five = j.filter(pl.col("fips_a").str.len_chars() == 5)
    assert (five["fips"] == five["fips_a"]).all(), f"fips != {fname}:fips"
    print(
        f"[ok] main file == {fname}: {mine.height} county-years identical on {len(PAIRS)} value columns + fips ({five.height} 5-digit)"
    )

MAIN_COLS = [
    (
        "year",
        "Outbreak year. Study period 2011-2025 only. Same as `year` in analysis_dataset.",
    ),
    (
        "county",
        "County or place named in the case data (after cleaning). May be a region name (e.g. 'Twin Cities Metro Area, Minnesota'), 'State' (Delaware), or blank/'Unknown' for state-only records. Same as `county` in analysis_dataset.",
    ),
    ("state", "State. Same as `state` in analysis_dataset."),
    (
        "place_type",
        "county | county_equivalent (parish, borough, independent city, DC, etc.: any non-'County' type in the Census 2020 county file) | multi_county_region (one record covers several counties; includes Delaware pooled to the whole state) | state_only (cases known only to the state; no county identified; all are dropped). Blank for other records dropped before the vaccine step.",
    ),
    (
        "fips",
        "5-digit county FIPS. Same as `fips` in analysis_dataset for single counties. Blank here for regions (analysis_dataset holds the region name or a 2-digit code), for state-only records and for other records dropped before the vaccine step. Where the pipeline had no FIPS at the vaccine merge (e.g. Indiana 2012, New Mexico 2011), it comes from the analysis file or a county-name lookup.",
    ),
    (
        "fips_counties_in_region",
        "For multi_county_region rows: the FIPS codes of the counties that make up the region, separated by ';' (`fips` is blank on these rows). The analysis spreads the region's single vaccine value across them. Blank where the repository does not define the members (e.g. Iowa 'Central').",
    ),
    (
        "cases",
        "Total reported cases for this record, after the 2025 QC overrides (see `notes`). Same as `cases` in analysis_dataset, which is per county-year (the 2 Detroit/Wayne pairs are two records here).",
    ),
    (
        "imported",
        "Cases the source classes as imported; a blank in the source is counted as 0. Same as `imported` in analysis_dataset. See imports_reported_or_assumed_zero. What 'imported' means (international or any) is not documented in the repository.",
    ),
    (
        "indigenous",
        "Cases the source classes as 'Indigenous' (locally acquired); a blank in the source is shown as 0. Same as `indigenous` in analysis_dataset. No model uses it; the full file keeps the blank-preserving version.",
    ),
    (
        "imports_reported_or_assumed_zero",
        "reported = the source gave an imported count (which may be a true 0). assumed_zero = the imported count was blank, so imported = 0 is an assumption and every case counts toward cases_outbreak. Refers to `imported` only.",
    ),
    (
        "cases_outbreak",
        "The model response: cases minus imported, at county-year level. Same as `cases_outbreak` in analysis_dataset. Derived, not estimated. Blank for dropped records.",
    ),
    (
        "county_year_used_inference",
        "TRUE if this county-year is used in the inference (the inferential models): it is a row of analysis_dataset_imputed.csv, and of analysis_dataset.csv too when vaccine_data_imputed is FALSE. FALSE if it is left out; inference_excluded_reason says why.",
    ),
    (
        "inference_excluded_reason",
        "Why a county-year was excluded from the inference. One of: 'no vaccine data' | 'county not identified' | 'could not verify case report' | 'outbreak among non-US population' | 'outbreak in ICE detention facility' (meanings in the README). Blank when county_year_used_inference is TRUE.",
    ),
    (
        "vaccine_data_imputed",
        "TRUE if at least one of the three vaccine values was imputed (from the same county in a neighbouring school year, or from CDC state data), so the record is only in the imputed sensitivity analysis (analysis_dataset_imputed.csv). FALSE if all three were observed at the matched school year, so it is in analysis_dataset.csv and analysis_dataset_imputed.csv. Blank when county_year_used_inference is FALSE. A single value counts as imputed when its *_source is 'state' (other than Delaware's own whole-state figure) or its *_school_year differs from vaccine_school_year. The *_source and *_school_year columns show which values and how.",
    ),
    (
        "vaccine_school_year",
        "School year of the vaccine data matched to this outbreak: ALWAYS the outbreak year minus 1 (outbreak 2024 -> '2023-24'), whether or not any value was imputed. Blank only for the 19 records dropped before vaccine matching. A value imputed from a neighbouring year is shown in that value's own *_school_year column.",
    ),
    (
        "estimated_mmr_cov",
        "Kindergarten MMR (or all-vaccine, see mmr_or_all_vaccine_coverage) coverage, percent, as used by the models. Same as `estimated_mmr_cov` in analysis_dataset.",
    ),
    (
        "nonmed_exempt_pct",
        "Kindergarten nonmedical exemption prevalence, percent. Same as `nonmed_exempt_pct` in analysis_dataset.",
    ),
    (
        "med_exempt_pct",
        "Kindergarten medical exemption prevalence, percent. Same as `med_exempt_pct` in analysis_dataset.",
    ),
    (
        "mmr_or_all_vaccine_coverage",
        "What estimated_mmr_cov measures: MMR | all_vaccines_up_to_date (MMR blank/0 so all-vaccine coverage was used) | mixed_MMR_and_all_vaccines (multi-county region whose member counties used both). A multi-county region whose members all used one measure is labelled with that measure; `notes` says so.",
    ),
    (
        "estimated_mmr_cov_source",
        "Whose figure the MMR value is: county (the county's own reported value, or the same county's value from a neighbouring school year) | region (the value for a multi-county region: a population-weighted average of its member counties, or Alaska's economic region) | state (a state-level figure: the CDC state estimate substituted because the county had no value, which is an imputation; or Delaware's own whole-state figure, which is not) | assumed_zero (blank in the source, set to 0 by rule) | blank (no value). Which school year it is from is in estimated_mmr_cov_school_year.",
    ),
    ("nonmed_exempt_pct_source", "Same, for the nonmedical exemption value."),
    ("med_exempt_pct_source", "Same, for the medical exemption value."),
    (
        "estimated_mmr_cov_school_year",
        "The school year the MMR value comes from, e.g. '2013-14'. Equals vaccine_school_year (the matched year, outbreak year minus 1) for observed values and CDC state values. It is one year later (the school year starting in the outbreak year) or one year earlier when the value was imputed from the same county's neighbouring year. Blank if the value is assumed_zero or missing.",
    ),
    (
        "nonmed_exempt_pct_school_year",
        "Same, for the nonmedical exemption value.",
    ),
    ("med_exempt_pct_school_year", "Same, for the medical exemption value."),
    (
        "notes",
        "Special handling: South Carolina 2025 source, QC case-count overrides, name fixes, NYC->Kings, Delaware pooling, merged records, state-level-only records, and what is special about the vaccine data. The README lists every possible note.",
    ),
]
main_out = main.drop("record_id")
assert [c for c, _ in MAIN_COLS] == main_out.columns, (
    "main column order/documentation mismatch"
)

# --- change_log.csv: one row per drop / recode / imputation / vaccine-side edit
log = []


def add(
    record_id, year, state, county, action, stage, field, old, new, reason
):
    log.append(
        {
            k: None if v is None else str(v)
            for k, v in dict(
                record_id=record_id,
                year=year,
                state=state,
                county=county,
                action=action,
                stage=stage,
                field=field,
                old_value=old,
                new_value=new,
                reason=reason,
            ).items()
        }
    )


for r in out.filter(pl.col("record_status") == "DROPPED").iter_rows(
    named=True
):
    add(
        r["record_id"],
        r["year"],
        r["state"],
        r["county"],
        "DROPPED",
        r["status_stage"],
        "record",
        r["cases_source"],
        None,
        r["status_reason"],
    )
for r in out.filter(pl.col("recode_notes").is_not_null()).iter_rows(
    named=True
):
    for n in r["recode_notes"].split("; "):
        add(
            r["record_id"],
            r["year"],
            r["state"],
            r["county"],
            "RECODED",
            "case_cleaning",
            "record",
            None,
            None,
            n,
        )
FIELD = {
    "MMR": ("mmr_cov_used", "MMR coverage"),
    "NME": ("nme_pct_used", "nonmedical exemption"),
    "ME": ("me_pct_used", "medical exemption"),
}
for r in rep.filter(pl.col("record_status") == "INCLUDED_IMPUTED").iter_rows(
    named=True
):
    for f, part in zip(
        r["imputed_fields"].split("+"), r["imputation_detail"].split(" | ")
    ):
        col, _ = FIELD[f]
        add(
            r["record_id"],
            r["year"],
            r["state"],
            r["county"],
            "IMPUTED",
            "vaccine_imputation",
            f,
            None,
            r[col],
            part,
        )
xl_raw = pl.read_excel(os.path.join(data_dir, VACC_COMPILED))
xl_raw = xl_raw.rename({c: c.lower() for c in xl_raw.columns})
master_keys = {
    (r["state"], r["county"], r["vyear"]) for r in m.iter_rows(named=True)
}
for c, s, y in VACC_QC_DROPS:
    old = xl_raw.filter(
        (pl.col("county") == c)
        & (pl.col("state") == s)
        & (pl.col("year") == y)
    )
    desc = (
        "row not present in the compiled sheet"
        if old.height == 0
        else "; ".join(
            f"{k}={old[k][0]}"
            for k in [
                "nonmed_exempt_pct",
                "med_exempt_pct",
                "mmr_cov",
                "all_vacc_cov",
            ]
        )
    )
    hit = (s, c, y) in master_keys
    assert not hit, (
        f"QC-removed vaccine row {c},{s},{y} matches an outbreak county-year; log it with its record"
    )
    add(
        None,
        y,
        s,
        c,
        "VACCINE_ROW_REMOVED",
        "vaccine_qc",
        "all",
        desc,
        None,
        VACC_QC_DROP_REASON + "; no outbreak county-year uses this row",
    )
cocke = xl_raw.filter(
    (pl.col("county") == "Cocke")
    & (pl.col("state") == "Tennessee")
    & (pl.col("year") == 2024)
)
add(
    None,
    2024,
    "Tennessee",
    "Cocke",
    "VACCINE_VALUE_RECODED",
    "vaccine_qc",
    "MMR",
    cocke["mmr_cov"][0] if cocke.height else None,
    94.5,
    "MMR coverage manually set to 94.5 (reason not recorded in the repo)"
    + (
        ""
        if (("Tennessee", "Cocke", 2024) in master_keys)
        else "; no outbreak county-year uses this row"
    ),
)
for r in rep.filter(
    pl.col("me_origin") == "observed_blank_set_to_zero_state_rule"
).iter_rows(named=True):
    add(
        r["record_id"],
        r["year"],
        r["state"],
        r["county"],
        "VACCINE_VALUE_RECODED",
        "vaccine_cleaning",
        "ME",
        None,
        0,
        f"blank medical-exemption value set to 0 for {r['state']} (rule applied to WY/LA/VT/MS)",
    )
change_log = pl.DataFrame(log, schema={k: pl.String for k in log[0]})
change_log.write_csv(os.path.join(out_dir, "change_log.csv"))
print("wrote change_log.csv", change_log.shape)

# --- vaccine_qc_rows.csv: the 7 compiled-sheet rows removed in read_vaccinations (values are the sheet's)
VACC_QC_REASON = "Low vaccine coverage"
RAW4 = ["nonmed_exempt_pct", "med_exempt_pct", "mmr_cov", "all_vacc_cov"]
qc_drop_keys = pl.DataFrame(
    VACC_QC_DROPS, schema=["county", "state", "year"], orient="row"
)
qc_rows = (
    xl_raw.join(qc_drop_keys, on=["state", "county", "year"], how="semi")
    .select(
        "state",
        "county",
        school_year=pl.col("year").cast(pl.String)
        + "-"
        + ((pl.col("year") + 1) % 100).cast(pl.String).str.zfill(2),
        *[pl.col(c).cast(pl.String) for c in RAW4],
        outbreak_use=pl.struct(
            "state", "county", "year"
        ).map_elements(  # True if an outbreak record matches this row
            lambda r: (r["state"], r["county"], r["year"]) in master_keys,
            return_dtype=pl.Boolean,
        ),
        reason=pl.lit(VACC_QC_REASON),
    )
    .sort("state", "county")
)
assert qc_rows.height == len(VACC_QC_DROPS), (
    "QC-removed rows not all found in the compiled sheet"
)
qc_rows.write_csv(os.path.join(out_dir, "vaccine_qc_rows.csv"))
print("wrote vaccine_qc_rows.csv", qc_rows.shape)


# --- row_flow.csv ------------------------------------------------------------
STAGES = [  # (status, inference_excluded_reason, meaning)
    (
        "INCLUDED_DIRECT",
        None,
        "In primary models: vaccine data observed at the aligned school year.",
    ),
    (
        "INCLUDED_IMPUTED",
        None,
        "Only in the imputed sensitivity analysis: >=1 vaccine field imputed.",
    ),
    (
        "DROPPED",
        "no vaccine data",
        "Cases usable, but vaccine data still incomplete after imputation (or a multi-county region that is never imputed).",
    ),
    (
        "DROPPED",
        "county not identified",
        "Cases known only to the state ('Unknown', no county listed, 'Central' Utah).",
    ),
    (
        "DROPPED",
        "could not verify case report",
        "The 2025 QC sheet has no verified count for the record.",
    ),
    (
        "DROPPED",
        "outbreak among non-US population",
        "Chicago migrant shelter 2024 and Operation Allies Welcome (Wisconsin) 2021.",
    ),
    ("DROPPED", "outbreak in ICE detention facility", "Arizona 2016."),
    (
        "DROPPED",
        "zero cases reported (no outbreak)",
        "State-by-year placeholder rows (and 2 QC'd/blank counts): no outbreak to model.",
    ),
    (
        "DROPPED",
        "before 2011 (outside study period)",
        "Study period is 2011-2025.",
    ),
    (
        "DROPPED",
        "total row in source sheet",
        "Summary row in a source sheet, not an outbreak.",
    ),
]
NOT_IN_MAIN = [
    "zero cases reported (no outbreak)",
    "before 2011 (outside study period)",
    "total row in source sheet",
]
assert (
    out.filter(pl.col("status_stage") == "case_source_superseded").height == 0
)  # no such rows; otherwise add it above
out_lab = out.with_columns(
    inference_excluded_reason=pl.Series(
        [
            plain_drop_reason(r["status_stage"], r["state"], r["county"])
            for r in out.iter_rows(named=True)
        ],
        dtype=pl.String,
    )
)
flow = out_lab.group_by("record_status", "inference_excluded_reason").agg(
    n_records=pl.len(),
    n_county_years=pl.col("county_year_representative").fill_null(False).sum(),
    cases_source=pl.sum("cases_source"),
    cases_used=pl.sum("cases_used"),
)
flow = pl.DataFrame(
    [
        {"record_status": s_, "inference_excluded_reason": g, "meaning": t}
        for s_, g, t in STAGES
    ]
).join(
    flow,
    on=["record_status", "inference_excluded_reason"],
    how="left",
    nulls_equal=True,
)
flow = flow.with_columns(
    pl.col("n_records").fill_null(0),
    n_county_years=pl.when(
        pl.col("inference_excluded_reason").is_in(
            [
                "county not identified",
                "could not verify case report",
                "outbreak among non-US population",
                "outbreak in ICE detention facility",
            ]
            + NOT_IN_MAIN
        )
    )
    .then(None)
    .otherwise(pl.col("n_county_years").fill_null(0)),
)
assert flow["n_records"].sum() == out.height, (
    "a record has a inference_excluded_reason that is not listed in row_flow"
)
flow = pl.concat(
    [
        flow,
        pl.DataFrame(
            {
                "record_status": ["TOTAL"],
                "n_records": [flow["n_records"].sum()],
                "meaning": ["All records in the three case files"],
            }
        ),
    ],
    how="diagonal_relaxed",
)
flow = flow.rename({"record_status": "status"}).with_columns(
    pl.col("status").replace(USE),
    in_main_file=pl.when(
        pl.col("inference_excluded_reason").is_in(NOT_IN_MAIN)
    )
    .then(pl.lit("no"))
    .when(pl.col("status") == "TOTAL")
    .then(None)
    .otherwise(pl.lit("yes")),
)
# main-file rows = records marked in_main_file, less the 2 records merged into Wayne County county-years
assert (
    flow.filter(pl.col("status") != "TOTAL")
    .filter(pl.col("in_main_file") == "yes")["n_records"]
    .sum()
    == main.height + n_merged_away
)
flow.write_csv(os.path.join(out_dir, "row_flow.csv"))
print(
    flow.select(
        "status",
        "inference_excluded_reason",
        "n_records",
        "in_main_file",
        "n_county_years",
        "cases_used",
    )
)

# --- verbatim copies of pipeline files + manifest ---------------------------
PIPE_FILES = [
    "analysis_dataset.csv",
    "analysis_dataset_imputed.csv",
    "outbreaks_vaccinations_cleaned.csv",
    "outbreaks_vaccinations_cleaned_imputed.csv",
    "processed_vaccinations_data.csv",
]
copy_dir = os.path.join(out_dir, "pipeline_files")
os.makedirs(copy_dir, exist_ok=True)
for f in PIPE_FILES:
    shutil.copy2(os.path.join(data_dir, f), os.path.join(copy_dir, f))
SOURCES = [
    SRC_COMBINED,
    SRC_2025,
    SRC_SC,
    VACC_COMPILED,
    "Oklahoma.xlsx",
    "state_level_coverages.csv",
    "raw_merged_outbreaks_vaccinations.csv",
    "missing_counties.csv",
    "Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260526.csv",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


man = []
for role, files, base in [
    ("copied_pipeline_file", PIPE_FILES, copy_dir),
    ("source_input", SOURCES, data_dir),
]:
    for f in files:
        p = os.path.join(base, f)
        n = (
            pl.read_csv(p, infer_schema_length=0).height
            if f.endswith(".csv")
            else None
        )
        man.append(
            {
                "role": role,
                "file": f,
                "bytes": os.path.getsize(p),
                "rows": n,
                "sha256": sha(p),
            }
        )
for f in PIPE_FILES:  # copies are byte-identical to ./data
    assert sha(os.path.join(copy_dir, f)) == sha(os.path.join(data_dir, f))
pl.DataFrame(man).write_csv(os.path.join(out_dir, "MANIFEST.csv"))
print("wrote MANIFEST.csv and copied", len(PIPE_FILES), "pipeline files")


# --- README.md ----------------------------------------------------------------
def n_rec(status):  # rows in the MAIN file with this status
    if status == "DROPPED":
        return main_out.filter(~pl.col("county_year_used_inference")).height
    return main_out.filter(
        pl.col("county_year_used_inference")
        & (pl.col("vaccine_data_imputed") == (status == "INCLUDED_IMPUTED"))
    ).height


DROP_DOC = [  # (inference_excluded_reason, meaning) for the options that appear in the main file
    (
        "no vaccine data",
        "The county-year has cases, but kindergarten MMR / nonmedical / medical exemption values were still incomplete at the matched school year after imputation (same county one school year earlier or later, then CDC state value), or it is a multi-county region that is never imputed. Any value that does exist is shown; the rest is blank.",
    ),
    (
        "county not identified",
        "Cases are known only to the state, so no county vaccine data can be matched; shown with place_type = state_only. The source sheet lists no county for Nebraska 2014 and 2015, North Carolina 2011, Oklahoma 2018, Rhode Island 2011 and Tennessee 2011, 2014, 2016 and 2018; the 2025 South Carolina file lists the place as 'Unknown'; the 2025 Utah sheet lists 'Central' (a multi-county region the repository cannot map to counties). 32 cases in total.",
    ),
    (
        "could not verify case report",
        "The 2025 QC sheet has no verified case count for the record.",
    ),
    (
        "outbreak among non-US population",
        "Chicago migrant shelter 2024 (Cook IL) and Operation Allies Welcome Afghan evacuees 2021 (Monroe WI); papers linked in notes.",
    ),
    (
        "outbreak in ICE detention facility",
        "Pinal AZ 2016; paper linked in notes.",
    ),
]
DROP_DOC = [
    (k, main_out.filter(pl.col("inference_excluded_reason") == k).height, m_)
    for k, m_ in DROP_DOC
]
assert (
    sum(n for _, n, _ in DROP_DOC)
    == main_out.filter(~pl.col("county_year_used_inference")).height
)


SRC_OPTIONS = [
    (
        "county",
        "The county's own reported value, or the same county's value from a neighbouring school year (see the school-year column).",
    ),
    (
        "region",
        "The value for a multi-county region: a population-weighted average of its member counties, or Alaska's economic region.",
    ),
    (
        "state",
        "A state-level figure: the CDC state estimate substituted because the county had no value (an imputation), or Delaware's own whole-state figure (Delaware has no county data; not an imputation).",
    ),
    (
        "assumed_zero",
        "Blank in the source and set to 0 by rule (medical exemption in WY/LA/VT/MS).",
    ),
]
VAL_COLS = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
SRC_ROWS = [
    (
        o,
        *[
            main_out.filter(pl.col(f"{c}_source") == o).height
            for c in VAL_COLS
        ],
        m_,
    )
    for o, m_ in SRC_OPTIONS
]
SRC_ROWS.append(
    (
        "(blank)",
        *[
            main_out.filter(
                pl.col("vaccine_school_year").is_not_null()
                & pl.col(f"{c}_source").is_null()
            ).height
            for c in VAL_COLS
        ],
        "No value (county-years with no vaccine data).",
    )
)


def _rel(c, op):
    d_ = main_out.filter(pl.col(f"{c}_school_year").is_not_null())
    return d_.filter(
        op(pl.col(f"{c}_school_year"), pl.col("vaccine_school_year"))
    ).height


YEAR_ROWS = [
    (
        "same as `vaccine_school_year`",
        *[_rel(c, lambda a_, b_: a_ == b_) for c in VAL_COLS],
        "The matched school year, i.e. outbreak year minus 1 (outbreak 2014 -> 2013-14). All observed values, Delaware's state figure and all CDC state values.",
    ),
    (
        "one year later",
        *[_rel(c, lambda a_, b_: a_ > b_) for c in VAL_COLS],
        "The school year starting in the outbreak year (outbreak 2014 -> 2014-15). Imputed from the same county.",
    ),
    (
        "one year earlier",
        *[_rel(c, lambda a_, b_: a_ < b_) for c in VAL_COLS],
        "The school year before the matched one (outbreak 2014 -> 2012-13). Imputed from the same county.",
    ),
    (
        "(blank)",
        *[
            main_out.filter(
                pl.col("vaccine_school_year").is_not_null()
                & pl.col(f"{c}_school_year").is_null()
            ).height
            for c in VAL_COLS
        ],
        "assumed_zero, or no value.",
    ),
]


def n_cy(status):
    return rep.filter(pl.col("record_status") == status).height


imp_rep = rep.filter(pl.col("record_status") == "INCLUDED_IMPUTED")
imp_counts = {
    c: imp_rep.group_by(c).len().sort("len", descending=True).rows()
    for c in ["mmr_origin", "nme_origin", "me_origin"]
}
status_counts = (
    rep.group_by("import_reporting_status")
    .len()
    .sort("len", descending=True)
    .rows()
)
inc_main = main_out.filter(pl.col("county_year_used_inference"))
n_with_vax_records = inc_main.height
n_inc_assumed = inc_main.filter(
    pl.col("imports_reported_or_assumed_zero") == "assumed_zero"
).height
NOTE_KINDS = [  # (text that identifies the note, plain-language meaning)
    (
        "South Carolina 2025",
        "South Carolina 2025 case counts come from a separate source and include cases until the end of January 2026.",
    ),
    (
        "reported in Detroit City",
        "Wayne County, MI row that includes the stated number of cases reported in Detroit City (Detroit has no FIPS or vaccine data of its own).",
    ),
    (
        "pooled to state level",
        "Delaware county cases pooled to one state-level row (Delaware reports vaccine data only for the whole state).",
    ),
    (
        "Outbreak among non-US population",
        "Outbreak among a non-US population (Chicago migrant shelter 2024; Afghan evacuees, Operation Allies Welcome 2021): left out of the analysis; paper linked.",
    ),
    (
        "Outbreak in an ICE detention facility",
        "Outbreak in an ICE detention facility (Arizona 2016): left out of the analysis; paper linked.",
    ),
    (
        "2017 Minnesota measles outbreak",
        "The four 2017 Minnesota county-years (Hennepin, Ramsey, Crow Wing, Le Sueur), a clustered outbreak largely among Somali-American children: used everywhere except one version of the time-trend (random-walk) models; paper linked.",
    ),
    (
        "state-level information only",
        "Cases known only at state level (no county identified); these records are dropped.",
    ),
    (
        "Kansas values come from",
        "Kansas county values come from a separate county-level query, not the compiled vaccine sheet.",
    ),
    (
        "population-weighted average",
        "Multi-county region: vaccine values are a population-weighted average of member counties.",
    ),
    (
        "Region value averages",
        "Multi-county region: which coverage measure the member counties used (MMR or all-vaccines), and members with no value.",
    ),
    (
        "medical exemption blank in source",
        "Blank medical exemption set to 0 (rule applied to WY/LA/VT/MS).",
    ),
    (
        "filled with CDC state values",
        "Multi-county region (Iowa 2025) imputed with CDC state values.",
    ),
    (
        "reporting region (Alaska",
        "Alaska economic-region value applied to its member areas.",
    ),
]
note_rows = [
    (
        k,
        main_out.filter(
            pl.col("notes").str.contains(k, literal=True).fill_null(False)
        ).height,
        m_,
    )
    for k, m_ in NOTE_KINDS
]
n_with_notes = main_out.filter(pl.col("notes").is_not_null()).height
main_scope = out_full.filter(
    pl.col("status_stage") != "case_zero_or_missing"
)  # records behind the main file
n_override = main_scope.filter(
    pl.col("recode_notes").str.contains("cases overridden").fill_null(False)
).height
n_nyc = main_scope.filter(
    pl.col("recode_notes").str.contains("New York City").fill_null(False)
).height
not_verified = "; ".join(
    f"{r['county']} ({r['state']})"
    for r in out_full.filter(
        pl.col("status_stage") == "case_report_not_verified"
    )
    .sort("state", "county")
    .iter_rows(named=True)
)
vacc_qc_drops = "; ".join(f"{c} ({s_}) {y}" for c, s_, y in VACC_QC_DROPS)
proc = pl.read_csv(
    os.path.join(data_dir, "processed_vaccinations_data.csv"),
    schema_overrides=S,
)
n_null_fips = proc.filter(pl.col("fips").is_null()).height
slc = pl.read_csv(os.path.join(data_dir, "state_level_coverages.csv"))
slc_years = f"{slc['year'].min()}-{slc['year'].max()}"
n_zero_me = rep.filter(
    pl.col("me_origin") == "observed_blank_set_to_zero_state_rule"
).height
n_imp_regions = imp_rep.filter(
    pl.col("geography_type") == "multi_county_region"
).height
n_after = imp_rep.filter(
    pl.col("imputation_detail").str.contains("ONE YEAR AFTER")
).height


def md_table(rows, header):
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    )


readme = f"""# Measles outbreak-size analysis: reviewer data bundle

Everything here is generated by `analysis/12_export_reviewer_data.py`
(`uv run analysis/12_export_reviewer_data.py`). It reads only `./data`, makes no network
calls, and does not change any analysis file. Values come from the pipeline's own files;
this script only adds labels, flags and logs, and asserts that its county-year rows equal
`analysis_dataset.csv` and `analysis_dataset_imputed.csv` exactly.

## Start here: `outbreak_records_master.csv`

One row per outbreak record from the three case files, **2011-2025 only**, with every row
tagged by `county_year_used_inference` and `vaccine_data_imputed` ({
    main_out.height
} rows, {main_out.width} columns;
the 2 Wayne County, MI county-years that were split across a "Detroit City" and a "Wayne" source
record are one row each). Not in this file
(but counted in `row_flow.csv` and listed in `change_log.csv`): records before 2011, the source
sheets' total rows, and the placeholder rows saying a state had zero cases that year (no
county, no outbreak).
`outbreak_records_master_full.csv` ({out_full.height} rows, {
    out_full.width
} columns) has the same
records plus the zero-case placeholder rows and detailed columns (imputation description, 5-year
averages, model covariates, model-variant flags). Join the two on (`year`, `state`, `county`);
the full file has two rows for each of the 2 Wayne County county-years.

{
    md_table(
        [
            (
                USE["INCLUDED_DIRECT"],
                n_rec("INCLUDED_DIRECT"),
                n_cy("INCLUDED_DIRECT"),
                "`county_year_used_inference` = TRUE, `vaccine_data_imputed` = FALSE. In the primary models (`analysis_dataset.csv`) and also in the imputed sensitivity analysis. Kindergarten MMR coverage and both exemption measures were all present at the matched school year.",
            ),
            (
                USE["INCLUDED_IMPUTED"],
                n_rec("INCLUDED_IMPUTED"),
                n_cy("INCLUDED_IMPUTED"),
                "`county_year_used_inference` = TRUE, `vaccine_data_imputed` = TRUE. Only in the imputed sensitivity analysis (`analysis_dataset_imputed.csv`). At least one of the three vaccine fields was imputed; `estimated_mmr_cov_source`, `nonmed_exempt_pct_source` and `med_exempt_pct_source` say which and from where.",
            ),
            (
                USE["DROPPED"],
                n_rec("DROPPED"),
                n_cy("DROPPED"),
                "`county_year_used_inference` = FALSE. In neither analysis; `inference_excluded_reason` says why (options below).",
            ),
        ],
        ["status", "rows", "county-years", "meaning"],
    )
}

Rows are **county-years with at least one reported case**, not a full county panel. A
county-year is (state, county, year). `params.yml: inclusion_threshold` is not applied.


## How rows flow (`row_flow.csv`)

{
    md_table(
        [
            (
                r["status"],
                r["inference_excluded_reason"] or "",
                r["n_records"],
                r["in_main_file"] or "",
                "" if r["n_county_years"] is None else r["n_county_years"],
                r["meaning"],
            )
            for r in flow.iter_rows(named=True)
        ],
        [
            "status",
            "inference_excluded_reason",
            "records",
            "in main file",
            "county-years",
            "meaning",
        ],
    )
}

Case counts at each step are in `row_flow.csv`. The two "total row in source sheet" records are the
total rows of the source sheets; their case counts are not outbreak cases. The "zero cases reported"
records are almost all state-by-year placeholder rows (no county) saying the state had zero cases that
year; the rest are one DC record whose QC'd 2025 count is 0 and one state row with a blank count.

### `inference_excluded_reason` options (rows in the main file)

{md_table(DROP_DOC, ["inference_excluded_reason", "rows", "meaning"])}

## How vaccine data are matched to outbreaks

An outbreak in calendar year **Y** is matched to the kindergarten vaccine data for school
year **Y-1** (e.g. outbreak 2024 -> school year 2023-24). Column `vaccine_school_year`
shows the match. The vaccine sheet's data dictionary defines the measures as the percent of
**kindergartners** with MMR coverage / nonmedical exemption / medical exemption. Where MMR
coverage is blank or 0 the all-vaccine up-to-date coverage is used instead
(`mmr_measure = all_vaccine_UTD_fallback`). Original reporting source, dose count and grade
are not recorded per state in this repository.

## How imputation works

For each outbreak county-year missing any of the three fields at school year Y-1, the
pipeline fills each missing field, in this order, from:
1. the same county, school year **Y** (one school year *after* the matched year);
2. the same county, school year **Y-2** (one school year *before* it);
3. the CDC state value for school year Y-1.

A county-year enters the imputed analysis only if all three fields end up filled. The
pipeline's list of multi-county regions (e.g. Twin Cities Metro, Region 1 Louisiana) is
never imputed; {
    n_imp_regions
} other multi-county places (Iowa 2025 'Central', 'Eastern',
'Western') were filled with CDC state values. {n_after} of the {
    n_cy("INCLUDED_IMPUTED")
}
imputed county-years use step 1 for at least one field; that donor school year starts in
the fall of the outbreak year, so the value may postdate the outbreak itself.

Origin of each imputed field (county-years):

{
    md_table(
        [
            (c.replace("_origin", ""), k, v)
            for c, rows_ in imp_counts.items()
            for k, v in rows_
        ],
        ["field", "origin", "county-years"],
    )
}

## Known issues a reviewer should know about

1. **Imputed nonmedical-exemption values from CDC state data.** The {
    n_nme_state
} such values
   (`nonmed_exempt_pct_source` = state) are CDC's *Non-Medical Exemption*. For state-years where CDC
   reports only *Any Exemption* (which includes medical exemptions), they are Any minus *Medical
   Exemption* (floored at 0; Any itself when the state has no medical figure), so medical
   exemptions are not counted as nonmedical. An earlier version of the pipeline used *Any
   Exemption* throughout (a text-filter error in `analysis/00_explore_data.py`); this is fixed.
   They affect only the imputed sensitivity analysis.
2. **A blank imported count is treated as 0 by the pipeline.** `imported` shows that 0, and
   `imports_reported_or_assumed_zero` says whether it was `reported` or `assumed_zero`. Of the
   {n_with_vax_records} main-file records used in the sensitivity analysis, {
    n_inc_assumed
} have an assumed-zero import
   count, so all of their cases count toward `cases_outbreak`. `cases_outbreak = cases -
   imported` is a derived quantity, not an estimate. `indigenous` is the source's
   "Indigenous" column with blanks shown as 0, as in analysis_dataset; no model uses it. The full
   file keeps the blank-preserving values and a four-way `import_reporting_status` for all
   {rep.height} county-years that reached the vaccine merge: {
    ", ".join(f"{k} {v}" for k, v in status_counts)
}.
   The meaning of "imported" (international vs. any) is not documented in the repository.
3. **2025 case counts are the QC sheet's counts.** For {
    n_override
} records the verified count
   differs from the source sheet's (old -> new in `change_log.csv`); `cases` uses the verified count
   and `notes` does not mention it. Imported/locally acquired counts were not adjusted, so
   they may not sum to the verified total.
4. **NYC is assigned to Kings County.** {
    n_nyc
} New York City records are assigned to Kings
   (Brooklyn), an assumption made in the analysis code (`change_log.csv` lists them).
5. **Delaware** is pooled to one state-level row (vaccine data are state-level), and other
   multi-county places use a region value (`place_type`, `fips_counties_in_region`).
6. **Manual QC items.** *Case side:* the QC sheet has no verified count for {
    not_verified
}, so those
   records are dropped as "could not verify case report". *Vaccine side* (none is used by any
   outbreak record in the main file; all are in `change_log.csv`): 7 vaccine rows removed ({
    vacc_qc_drops
})
   and Cocke TN 2024 MMR set to 94.5. The reasons for the vaccine-side edits are not recorded in
   the repository.
   *Three outbreaks left out of the analysis* (`inference_excluded_reason` starts with 'outbreak'; paper linked in `notes`):
   Cook IL 2024, a Chicago migrant shelter (MMWR, PMID 38753539); Monroe WI 2021, Afghan evacuees under
   Operation Allies Welcome (Lancet Public Health, PMID 37516478); Pinal AZ 2016, an ICE detention facility
   (Clin Infect Dis, PMID 30256908).
7. **Blank medical exemption set to 0** for WY/LA/VT/MS ({
    n_zero_me
} county-years;
   `med_exempt_pct_source = assumed_zero`).
8. **`processed_vaccinations_data.csv`** (read for `neighbors_coverage` in
   `01_assemble_analysis_dataset.py`, the NME policy scenarios and two figures) fills 13 states with state-level values from `state_level_coverages.csv`
   (school years {
    slc_years
} only; other years use that state's latest value), and the
   `_is_imputed` flag was not saved. It has {
    n_null_fips
} rows with a blank FIPS: rows that are not a
   single county (regions, state-level rows, school districts) are not written. The outbreak analysis datasets are
   built before that step.
9. **No pseudo-import is added to the outbreak data.** The Gamma(1,1) prior in
   `analysis/08_importation_model.py` (one pseudo-import and one pseudo-year in the import-rate
   estimate) affects import probabilities only; it does not change any case column here. The
   mechanistic model separately assumes outbreaks start from a single index case.
10. **Vaccine data version.** The analysis reads
    `Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied_AK_CA_Updated.xlsx` for the school years up to
    2024-25. For 2025-26 (`year` 2025) it adds the counties that file lacks from
    `vaxexempt2026 - forprod [Nov 2].csv` (New York's 2025-26 rows come from `New York.xlsx`; West
    Virginia keeps the spreadsheet's exemptions and takes the state's MMR coverage from the csv).

## What `notes` can say

{n_with_notes} of the {
    main_out.height
} records in the main file have a note. Counts are records.

{md_table(note_rows, ["note (text contains)", "records", "meaning"])}

## Columns shared with `analysis_dataset.csv`

These main-file columns have the **same name and the same values** as in `analysis_dataset.csv`
(and `analysis_dataset_imputed.csv`) on (year, state, county). The script asserts this on the
written file: every row with `county_year_used_inference` = TRUE and `vaccine_data_imputed` = FALSE against
`analysis_dataset.csv`, and every row with `county_year_used_inference` = TRUE against `analysis_dataset_imputed.csv`.

{md_table(CROSSWALK, ["column", "note"])}

`vaccine_school_year` is the string form of `year - 1`; analysis_dataset drops the numeric
equivalent (`year_relevant_vaccination`, kept in `outbreaks_vaccinations_cleaned*.csv`). All other
main-file columns are labels added for this bundle and have no counterpart.

## Files

* `outbreak_records_master.csv`: main file (columns below).
* `outbreak_records_master_full.csv`: same records plus the zero-case placeholder rows, all detailed columns (dictionary at the end).
* `row_flow.csv`: counts and case totals at each step.
* `change_log.csv`: every drop, recode, imputation and vaccine-side edit, one row each.
* `vaccine_qc_rows.csv`: the 7 rows of the compiled vaccine sheet removed in QC, with the sheet's original values and the reason.
* `pipeline_files/`: byte-identical copies of `analysis_dataset.csv`,
  `analysis_dataset_imputed.csv`, `outbreaks_vaccinations_cleaned.csv`,
  `outbreaks_vaccinations_cleaned_imputed.csv`, `processed_vaccinations_data.csv`.
* `MANIFEST.csv`: SHA-256, size and rows of the copies and of the source inputs used.

## Column dictionary: `outbreak_records_master.csv` (main file)

{md_table(MAIN_COLS, ["column", "description"])}

### Where each vaccine value comes from: `*_source` and `*_school_year`

Each of the three vaccine values (MMR, nonmedical, medical) has two columns beside it:
`*_source` says **whose figure** it is, and `*_school_year` says **which school year** it is from
(an actual school year such as `2013-14`, compared with `vaccine_school_year`).

**`*_source` options** (rows in the main file; counts are for MMR / nonmedical / medical):

{md_table(SRC_ROWS, ["option", "MMR", "nonmedical", "medical", "meaning"])}

**`*_school_year` relative to `vaccine_school_year`** (counts are for MMR / nonmedical / medical):

{
    md_table(
        YEAR_ROWS,
        ["school year is...", "MMR", "nonmedical", "medical", "meaning"],
    )
}

**Observed or imputed, per value.** A value is imputed when its source is `state` (other than Delaware's own
whole-state figure) or its school year differs from `vaccine_school_year`; otherwise it is observed. `vaccine_data_imputed` is TRUE when any of the
three is imputed (the script checks that this rule reproduces `vaccine_data_imputed` exactly).

## Column dictionary: `outbreak_records_master_full.csv`

{md_table([(c, d) for c, d in COLS], ["column", "description"])}

## `*_origin` vocabulary (full file)

* `observed_county_aligned_year`: the county's own value at school year Y-1.
* `observed_county_kansas_separate_query` / `observed_county_filled_from_Oklahoma_xlsx`: county value from the separate Kansas query / `Oklahoma.xlsx`.
* `observed_popweighted_average_of_member_counties`: multi-county region; population-weighted mean of member counties.
* `observed_reporting_region_applied_to_members`: Alaska economic region applied to member counties.
* `observed_state_level_applied_to_county`: Delaware (state value).
* `observed_manually_corrected_to_94.5`, `observed_blank_set_to_zero_state_rule`: edited observed values.
* `imputed_same_county_FOLLOWING_school_year` / `..._EARLIER_school_year`: steps 1 and 2 above.
* `imputed_state_cdc_value`: step 3 (CDC state value, school year Y-1); for nonmedical exemption, CDC's Non-Medical Exemption (Any minus Medical where only Any exists; known issue 1).
* `imputed_set_to_zero_no_state_ME`: medical exemption set to 0 because CDC has no state figure.
* `MISSING_not_imputed`: field still missing (dropped county-years only).
"""
with open(os.path.join(out_dir, "README.md"), "w") as fh:
    fh.write(readme)
print("wrote README.md")
