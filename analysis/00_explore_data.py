import argparse
import os
import re

import polars as pl
import yaml
from addfips import AddFIPS
from dotenv import load_dotenv
from tqdm.auto import tqdm

from outbreak_size import synth_pop_clusters, utils, vaccination_wrangling
from outbreak_size.adjacency import CONNECTICUT_COUNTY_PAIRS
from outbreak_size.analysis_dataset import (
    VACCINE_FIELDS,
    assemble_analysis_dataset,
    fill_from_earlier_school_year,
    merge_outbreaks_vaccinations,
    require_vaccine_data,
)
from outbreak_size.baseline import place_slopes
from outbreak_size.case_assignments import (
    ARIZONA_2026_COUNTIES_WITHOUT_ROWS,
    read_case_assignments,
    shapefile_counties,
)
from outbreak_size.data_year import county_data_year_of_outbreak
from outbreak_size.interventions import STATES_WITH_BANS
from outbreak_size.outbreaks import process_outbreaks_df
from outbreak_size.regions import (
    expand_to_residual_members,
    higher_level_regions,
    members_with_own_row,
    members_without_own_row,
    require_filled_rows_use_same_members,
    require_no_own_row_members,
    require_own_row_members_in_main_rows,
)
from outbreak_size.vaccination_wrangling import (
    apply_qc_exclusions,
    impute_from_bordering_years,
    process_vaccinations_df,
    query_avg_vaccinations_region_years,
)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--five-year",
    help="Whether to include five year averaged vaccination data.",
    type=bool,
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()
five_year = args.five_year
# One run writes every case-side file twice: with the cases of 2026
# (name_with_2026.csv) and, as the files always were, with the years up to
# LAST_YEAR_WITHOUT_2026 (name.csv). The two come from the same frame, so the
# Census is asked once.
LAST_YEAR_WITHOUT_2026 = 2025


def write_without_and_with_2026(df: pl.DataFrame, name: str) -> None:
    """Write `df` (all the years) as {name}_with_2026.csv, and its rows up
    to 2025, without the column that only the rows of 2026 can have, as
    {name}.csv."""
    df.write_csv(os.path.join(data_dir, f"{name}_with_2026.csv"))
    df.filter(pl.col("year") <= LAST_YEAR_WITHOUT_2026).drop(
        "vaccine_from_earlier_year", strict=False
    ).write_csv(os.path.join(data_dir, f"{name}.csv"))


def read_outbreaks(data_dir=data_dir) -> pl.DataFrame:
    df = pl.read_excel(
        os.path.join(data_dir, "Combined Measles Dataset QCed.xlsx")
    )

    # to ensure consistent lower case column names
    df = df.rename({col: col.lower() for col in df.columns})

    # The spreadsheet row of Georgia 2019, "Forsyth, Fulton, Paulding", has 7
    # cases, 0 indigenous and 18 imported (more imported than cases, so its
    # outbreak cases would be 7 - 18 = -11). The state-level file
    # IndigenousImportedFullData.xlsx has Georgia 2019 as 18 cases, 18
    # indigenous and 0 imported, so the row seems to have the state's figures
    # with indigenous and imported swapped. Its imported cases are set to 0
    # here until the spreadsheet is corrected.
    georgia_2019_region = (
        (pl.col("year") == 2019)
        & (pl.col("state") == "Georgia")
        & (pl.col("county") == "Forsyth, Fulton, Paulding")
    )
    assert df.filter(georgia_2019_region).height == 1
    df = df.with_columns(
        imported=pl.when(georgia_2019_region)
        .then(0)
        .otherwise(pl.col("imported"))
    )

    df_2025 = pl.read_csv(
        os.path.join(data_dir, "2025CasesImports QCed clean.csv"),
        schema_overrides={"fips": pl.String},
    )

    # correct a year = 2925 to a 2025
    df_2025 = df_2025.with_columns(
        year=pl.when(pl.col("year") == 2925)
        .then(2025)
        .otherwise(pl.col("year"))
    )

    # Iowa "Central", "Eastern" and "Western" (2025): Iowa reports some cases
    # by region only, with no county. Polk and Johnson have rows of their own;
    # the "Lauren Case" column used below is each region's cases net of those,
    # so a region row holds the cases that could not be placed in a county, on
    # top of the county rows (no case is counted twice). No source defines the
    # three labels (by case counts they match Iowa HHS districts D5, D7 and
    # D1, an unverified inference), so they stay places of their own.
    # # drop Central, Eastern, and Western Iowa bc no clue where these are
    # # county is "Central" and state is "Iowa"
    # df_2025 = df_2025.filter(
    #     ~(
    #         (pl.col("county") == "Central")
    #         & (pl.col("state") == "Iowa")
    #     )
    # )
    # df_2025 = df_2025.filter(
    #     ~(
    #         (pl.col("county") == "Eastern")
    #         & (pl.col("state") == "Iowa")
    #     )
    # )
    # df_2025 = df_2025.filter(
    #     ~(
    #         (pl.col("county") == "Western")
    #         & (pl.col("state") == "Iowa")
    #     )
    # )
    # Utah's "Central" is the Central Utah health district (Juab, Millard,
    # Piute, Sanpete, Sevier and Wayne: outbreak_size.regions), a place like
    # the other Utah districts; the Iowa "Central" is not renamed
    central_utah = (pl.col("county") == "Central") & (
        pl.col("state") == "Utah"
    )
    df_2025 = df_2025.with_columns(
        county=pl.when(central_utah)
        .then(pl.lit("Central Utah, Utah"))
        .otherwise(pl.col("county")),
        fips=pl.when(central_utah)
        .then(pl.lit("Central Utah, Utah"))
        .otherwise(pl.col("fips")),
    )

    # set the fips for any county that is a higher level region to be the name
    # of the county
    # override the cases column with Lauren Case
    df_2025 = df_2025.filter(pl.col("Lauren Case").is_not_null()).with_columns(
        cases=pl.col("Lauren Case")
    )

    # drop anything south carolina because we have our own south carolina data
    df_2025 = df_2025.filter(pl.col("state") != "South Carolina")

    # South Carolina's calendar-2025 cases (210; the file has no imports)
    sc, _ = read_case_assignments(
        os.path.join(data_dir, "measles_2025_case_assignments_SC.csv"),
        2025,
        shapefile_counties(data_dir),
    )
    # concatenate to df_2025
    df_2025 = pl.concat([df_2025, sc], how="diagonal_relaxed")

    # concatenate and return
    df = pl.concat([df, df_2025], how="diagonal_relaxed")

    return df


def read_outbreaks_2026(data_dir=data_dir) -> pl.DataFrame:
    """The 2026 cases (measles_2026_case_assignments.csv, through Sept/Oct
    2026), in the layout of read_outbreaks. Printed: the report by state."""
    cases, report = read_case_assignments(
        os.path.join(data_dir, "measles_2026_case_assignments.csv"),
        2026,
        shapefile_counties(data_dir),
        extra_counties=ARIZONA_2026_COUNTIES_WITHOUT_ROWS,
    )
    print(
        "\n2026 cases by state (control_total: the state's own count; "
        "assigned: the file's rows; state_only and dropped_unit: not placed; "
        "extra: counties without a row; cases: placed; unplaced: placed "
        "without a county fips, i.e. a multi-county unit):"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(report)
    return cases


def read_vaccinations(data_dir=data_dir) -> pl.DataFrame:
    df = pl.read_excel(
        os.path.join(
            data_dir,
            "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied_AK_CA_Updated.xlsx",
        )
    )
    # the 2025-26 school year ("year" 2025) the spreadsheet lacks
    df = vaccination_wrangling.add_2025_vaccination_rows(df, data_dir)
    # Tennessee's 2025-26 rows, from its own file
    df = vaccination_wrangling.add_tennessee_2025_rows(df, data_dir)

    # cells of the spreadsheet the data owner's revised file (or the raw
    # state files) have right, before the places that failed the QC are removed
    df = vaccination_wrangling.fix_known_values(df, data_dir)

    # remove the places that failed the data QC (params.yml)
    df = apply_qc_exclusions(df, params["qc_exclusions"])
    df = df.with_columns(
        MMR_cov=pl.when(
            (pl.col("county") == "Cocke")
            & (pl.col("state") == "Tennessee")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(94.5))
        # the sheet has the text ".89%" for Faulkner County, Arkansas, 2024; the
        # raw data give 89.89
        .when(
            (pl.col("county") == "Faulkner")
            & (pl.col("state") == "Arkansas")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(89.89))
        .otherwise(pl.col("MMR_cov"))
    )
    df = df.with_columns(
        med_exempt_pct=pl.when(
            (
                pl.col("state").is_in(
                    ["Wyoming", "Louisiana", "Vermont", "Mississippi"]
                )
            )
            & pl.col("med_exempt_pct").is_null()
        )
        .then(pl.lit(0))
        .otherwise(pl.col("med_exempt_pct"))
    )

    # upper case lower case seem to be inconsistent across datasets
    df = df.rename({col: col.lower() for col in df.columns})

    # Put the columns in the types that we need
    df = df.with_columns(
        [
            pl.col("nonmed_exempt_pct").cast(pl.Float64, strict=False),
            pl.col("med_exempt_pct").cast(pl.Float64, strict=False),
            pl.col("mmr_cov").cast(pl.Float64, strict=False),
            pl.col("all_vacc_cov").cast(pl.Float64, strict=False),
        ],
        fips=pl.when(pl.col("fips").str.len_chars() > 2)
        .then(pl.col("fips").str.zfill(5))
        .otherwise(pl.col("fips")),
    )
    # Wyoming's year labels (recording year, not school year) and the 2024
    # Hawaii exemptions of the data owner's revised file
    df = vaccination_wrangling.fix_wyoming_year_labels(df, data_dir)
    df = vaccination_wrangling.fill_hawaii_2024_exemptions(df, data_dir)
    # settle the fips of every row: keep the spreadsheet's fips when it is a
    # valid county, and only look the county up by name when it is missing or
    # not a county (a name lookup alone turns counties that share a name with
    # an independent city, e.g. Baltimore or St. Louis, into the city)
    df = vaccination_wrangling.resolve_vaccination_fips(df, data_dir)

    # add in kansas data
    kansas_vacc = vaccination_wrangling.read_kansas_data(data_dir=data_dir)
    df = pl.concat([df, kansas_vacc], how="diagonal_relaxed")

    # replace the spreadsheet's New York rows with New York.xlsx (this
    # df's coverage column is "mmr_cov"; estimated_mmr_cov is computed
    # downstream in process_vaccinations_df)
    new_york_vacc = vaccination_wrangling.read_new_york_data(
        data_dir=data_dir
    ).rename({"MMR_cov": "mmr_cov"})
    df = vaccination_wrangling.replace_state_rows(
        df, "New York", new_york_vacc
    )

    # fill in oklahoma data (mainly adds 2024 coverage, which was previously
    # missing). We only fill in nulls, since where the new file overlaps with
    # existing data (2019-2023 MMR coverage) the values already match, and we
    # don't want to overwrite existing exemption values with the new file's.
    # at this point df's columns are lowercase ("mmr_cov"), but the reader
    # returns "MMR_cov" (to match Montana's convention) and an
    # "estimated_mmr_cov" column that df doesn't have yet (it's computed
    # downstream in process_vaccinations_df). Align names before joining so
    # the join suffix logic below actually collides on the right columns.
    oklahoma_vacc = (
        vaccination_wrangling.read_oklahoma_data(data_dir=data_dir)
        .rename({"MMR_cov": "mmr_cov"})
        .drop("estimated_mmr_cov")
    )
    df = df.join(
        oklahoma_vacc,
        on=["state", "county", "year"],
        how="left",
        suffix="_oklahoma",
        maintain_order="left",
    )
    df = df.with_columns(
        all_vacc_cov=pl.when(
            (pl.col("state") == "Oklahoma") & pl.col("all_vacc_cov").is_null()
        )
        .then(pl.col("all_vacc_cov_oklahoma"))
        .otherwise(pl.col("all_vacc_cov")),
        mmr_cov=pl.when(
            (pl.col("state") == "Oklahoma") & pl.col("mmr_cov").is_null()
        )
        .then(pl.col("mmr_cov_oklahoma"))
        .otherwise(pl.col("mmr_cov")),
        nonmed_exempt_pct=pl.when(
            (pl.col("state") == "Oklahoma")
            & pl.col("nonmed_exempt_pct").is_null()
        )
        .then(pl.col("nonmed_exempt_pct_oklahoma"))
        .otherwise(pl.col("nonmed_exempt_pct")),
        med_exempt_pct=pl.when(
            (pl.col("state") == "Oklahoma")
            & pl.col("med_exempt_pct").is_null()
        )
        .then(pl.col("med_exempt_pct_oklahoma"))
        .otherwise(pl.col("med_exempt_pct")),
    ).drop(
        [
            "mmr_cov_oklahoma",
            "all_vacc_cov_oklahoma",
            "nonmed_exempt_pct_oklahoma",
            "med_exempt_pct_oklahoma",
        ]
    )

    # Montana.xlsx: the 2019 county coverage (the exemptions are those of the
    # spreadsheet, 2015-2018)
    df = vaccination_wrangling.add_montana_data(df, data_dir)

    return df


def add_pop_variables(
    merged_df: pl.DataFrame, census_key: str
) -> pl.DataFrame:
    census_query = utils.CensusQuery(
        census_key,
        cache_path=utils.CENSUS_CACHE_PATH,
        workers=utils.CENSUS_WORKERS,
    )
    fips = merged_df["fips"].to_list()
    fips_unique = merged_df["fips"].unique(maintain_order=True).to_list()
    # the census of the year before the outbreak year, as the vaccination data
    years_census = [
        county_data_year_of_outbreak(year)
        for year in merged_df["year"].to_list()
    ]
    pop_df = census_query.query_5_9_pop(years_census, fips)
    merged_df = merged_df.with_columns(
        pop_df[
            [
                "pop",
                "pop_under_5",
                "pop_5_9",
                "school_aged_population",
                "public_schooled",
                "private_schooled",
                "median_household_income",
                "number_graduate_degree",
                "pop_25_up",
                "pop_15_up",
                "fraction_white",
            ]
        ]
    )
    land_df = census_query.query_land_size(fips_unique)
    merged_df = merged_df.join(
        land_df, on="fips", how="left", maintain_order="left"
    )

    merged_df = merged_df.with_columns(
        pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
        fraction_public=(pl.col("public_schooled") / pl.col("pop_5_9")),
        fraction_private=(pl.col("private_schooled") / pl.col("pop_5_9")),
        fraction_graduate_degree=(
            pl.col("number_graduate_degree") / pl.col("pop_25_up")
        ),
    )
    return merged_df


def print_2026_fill(raw_merged_df: pl.DataFrame, filled_df: pl.DataFrame):
    """
    What the fill from the earlier school year did to the 2026 rows: how many
    cases have all three vaccine values from 2025-26, how many took their
    county's 2024-25 values, and how many stay without data. Raises if a
    row that now has all three values has no usable fips (a 5-digit county
    fips, a multi-county place of outbreak_size.regions, or Delaware's
    state-level row).
    """
    complete = pl.all_horizontal(
        [pl.col(c).is_not_null() for c in VACCINE_FIELDS]
    )
    rows = filled_df.filter(pl.col("year") == 2026)
    own = rows.filter(complete & ~pl.col("vaccine_from_earlier_year"))
    filled = rows.filter(complete & pl.col("vaccine_from_earlier_year"))
    still = rows.filter(~complete)
    total = rows["cases"].sum()
    print("\n2026 rows by the vaccination data they have:")
    for label, frame in [
        ("2025-26 data", own),
        ("2024-25 data of the county itself (the fill)", filled),
        ("none (they stay in the missing-data file)", still),
    ]:
        print(
            f"  {label}: {frame.height} rows, {frame['cases'].sum()} cases "
            f"({100 * frame['cases'].sum() / total:.1f}%)"
        )
    if still.height:
        print(
            still.group_by("state", maintain_order=True)
            .agg(rows=pl.len(), cases=pl.col("cases").sum())
            .sort(["cases", "state"], descending=[True, False])
        )
    regions = set(higher_level_regions())
    usable = (
        (pl.col("fips").str.len_chars() == 5)
        | (pl.col("fips").str.len_chars() == 2)  # Delaware's "State"
        | (pl.col("fips").is_null() & pl.col("county").is_in(regions))
    ).fill_null(False)
    no_fips = rows.filter(complete & ~usable)
    if no_fips.height:
        raise ValueError(
            "2026 rows with vaccine data but no usable fips: "
            f"{no_fips.select('state', 'county', 'fips').rows()}"
        )


def print_vaccine_data_availability(
    vaccinations_df: pl.DataFrame,
    data_dir: str = data_dir,
    first_year: int = 2011,
    last_year: int = 2024,
) -> None:
    """
    Share of all US county-years (50 states + DC, from county_adjacencies.txt)
    in first_year..last_year that have vaccine data, before any imputation:
    all three fields (MMR-or-all-vaccine coverage, nonmedical and medical
    exemptions) vs MMR-or-all-vaccine coverage alone.
    """
    adjacency = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={"County GEOID": pl.String},
    )
    counties = adjacency["County GEOID"].drop_nulls().unique()
    counties = counties.filter(
        counties.str.slice(0, 2).cast(pl.Int64) <= 56
    )  # 50 states + DC
    # the file lists Connecticut by its planning regions (2022 on), the
    # vaccination data by its eight counties
    counties = pl.concat(
        [
            counties.filter(counties.str.slice(0, 2) != "09"),
            pl.Series(
                sorted(
                    {
                        county
                        for pair in CONNECTICUT_COUNTY_PAIRS
                        for county in pair
                        if county.startswith("09")
                    }
                )
            ),
        ]
    )
    years = list(range(first_year, last_year + 1))
    available = (
        vaccinations_df.filter(
            pl.col("fips").is_in(counties.implode())
            & pl.col("year").is_in(years)
        )
        .group_by(["fips", "year"])
        .agg(
            mmr=pl.col("estimated_mmr_cov").is_not_null().any(),
            all_three=(
                pl.col("estimated_mmr_cov").is_not_null()
                & pl.col("nonmed_exempt_pct").is_not_null()
                & pl.col("med_exempt_pct").is_not_null()
            ).any(),
        )
    )
    # every county x every year, so county-years absent from the data count
    # as missing
    grid = (
        pl.DataFrame({"fips": counties})
        .join(pl.DataFrame({"year": years}), how="cross")
        .join(available, on=["fips", "year"], how="left")
        .with_columns(pl.col("mmr", "all_three").fill_null(False))
    )
    print(
        f"\nVaccine data availability before imputation: {counties.len()} "
        f"counties x {first_year}-{last_year} = {grid.height} county-years"
    )
    print(
        f"  all three fields (MMR/all-vaccine coverage + nonmedical + medical "
        f"exemptions): {100 * grid['all_three'].mean():.1f}% "  # type: ignore
        f"({grid['all_three'].sum()})"
    )
    print(
        f"  MMR/all-vaccine coverage alone: {100 * grid['mmr'].mean():.1f}% "  # type: ignore
        f"({grid['mmr'].sum()})"
    )
    print(
        grid.group_by("year")
        .agg(
            pct_all_three=100 * pl.col("all_three").mean(),
            pct_mmr_alone=100 * pl.col("mmr").mean(),
        )
        .sort("year")
        .with_columns(pl.col("pct_all_three", "pct_mmr_alone").round(1))
    )


def print_missing_vaccine_data_breakdown(
    raw_merged_df: pl.DataFrame, imputed_from_bordering: pl.DataFrame
) -> None:
    """
    Where the missing vaccine data for outbreak county-years comes from, and
    which imputation step (impute_from_bordering_years) fills it.

    raw_merged_df: outbreaks merged with vaccinations, before imputation.
    imputed_from_bordering: its output (rows with at least one field filled;
    empty if none). MMR = MMR-or-all-vaccine coverage, NME = nonmedical and
    MED = medical exemptions.
    """
    key = ["state", "county", "year"]
    n_all = raw_merged_df.height
    missing = (
        raw_merged_df.with_columns(
            mmr_missing=pl.col("estimated_mmr_cov").is_null(),
            nme_missing=pl.col("nonmed_exempt_pct").is_null(),
            med_missing=pl.col("med_exempt_pct").is_null(),
            multi_county_region=pl.col("county").is_in(
                list(higher_level_regions().keys())
            ),
        )
        .filter(
            pl.col("mmr_missing")
            | pl.col("nme_missing")
            | pl.col("med_missing")
        )
        .with_columns(
            pattern=pl.concat_str(
                [
                    pl.when(pl.col("mmr_missing")).then(pl.lit("MMR")),
                    pl.when(pl.col("nme_missing")).then(pl.lit("NME")),
                    pl.when(pl.col("med_missing")).then(pl.lit("MED")),
                ],
                separator=" + ",
                ignore_nulls=True,
            )
        )
    )
    n_missing = missing.height

    if imputed_from_bordering.height > 0:
        imputed = imputed_from_bordering.select(
            *key,
            "imputation_source",
            complete=pl.col("estimated_mmr_cov").is_not_null()
            & pl.col("nonmed_exempt_pct").is_not_null()
            & pl.col("med_exempt_pct").is_not_null(),
        )
    else:
        imputed = pl.DataFrame(
            schema={
                **{k: missing.schema[k] for k in key},
                "imputation_source": pl.String,
                "complete": pl.Boolean,
            }
        )
    assert imputed.select(key).is_duplicated().sum() == 0
    missing = missing.join(imputed, on=key, how="left", maintain_order="left")

    def tier(source):
        steps = {
            "state-year"
            if s.split(":")[1].startswith("state")
            else "neighboring-year"
            for s in source.split("; ")
        }
        if steps == {"neighboring-year"}:
            return "completed by neighboring-year data only"
        if steps == {"state-year"}:
            return "completed by state-year data only"
        return "completed by neighboring-year + state-year data"

    missing = missing.with_columns(
        outcome=pl.when(pl.col("complete").fill_null(False))
        .then(
            pl.col("imputation_source").map_elements(
                tier, return_dtype=pl.String
            )
        )
        .when(pl.col("complete").is_not_null())
        .then(pl.lit("partly filled, still incomplete"))
        .when(pl.col("multi_county_region"))
        .then(pl.lit("not imputed: multi-county region"))
        .otherwise(pl.lit("nothing found for this county"))
    )

    pct = lambda n, d: round(100 * n / d, 1)
    print("\n" + "=" * 80)
    print("MISSING VACCINE DATA FOR OUTBREAK COUNTY-YEARS")
    print("=" * 80)
    print(
        f"{n_all} outbreak county-years before imputation: "
        f"{n_all - n_missing} complete ({pct(n_all - n_missing, n_all)}%), "
        f"{n_missing} missing at least one field ({pct(n_missing, n_all)}%), "
        f"holding {missing['cases'].sum()} of {raw_merged_df['cases'].sum()} "
        f"cases ({pct(missing['cases'].sum(), raw_merged_df['cases'].sum())}%)"
    )
    print(
        "fields missing (a row can miss several): "
        f"MMR {missing['mmr_missing'].sum()}, "
        f"NME {missing['nme_missing'].sum()}, "
        f"MED {missing['med_missing'].sum()}"
    )

    print("\nA. which fields are missing:")
    print(
        missing.group_by("pattern", maintain_order=True)
        .agg(
            rows=pl.len(),
            completed_by_imputation=pl.col("complete").fill_null(False).sum(),
        )
        .with_columns(
            pct_of_missing=(100 * pl.col("rows") / n_missing).round(1),
            pct_of_all=(100 * pl.col("rows") / n_all).round(1),
        )
        .sort(["rows", "pattern"], descending=[True, False])
    )

    print("\nB. what imputation does with them:")
    print(
        missing.group_by("outcome", maintain_order=True)
        .agg(rows=pl.len())
        .with_columns(
            pct_of_missing=(100 * pl.col("rows") / n_missing).round(1),
            pct_of_all=(100 * pl.col("rows") / n_all).round(1),
        )
        .sort(["rows", "outcome"], descending=[True, False])
    )

    print(
        "\nC. step that filled each field, among rows completed by imputation "
        "(year+0 / year-2 = same county, outbreak year / two years before; "
        "state_cdc = CDC state-level data for the vaccination year):"
    )
    done = missing.filter(pl.col("complete").fill_null(False))
    steps = ["year+0", "year-2", "state_cdc", "state_cdc_zero"]
    filled = {}
    for field, missing_col in [
        ("mmr", "mmr_missing"),
        ("nonmed", "nme_missing"),
        ("med", "med_missing"),
    ]:
        # "(^|; )" anchors the field name, so "med:" does not match inside
        # "nonmed:"
        filled[field] = [
            done.filter(
                pl.col("imputation_source").str.contains(
                    f"(^|; ){field}:{re.escape(step)}(;|$)"
                )
            ).height
            for step in steps
        ]
        # every field that was missing in a completed row was filled by
        # exactly one step
        assert sum(filled[field]) == done.filter(pl.col(missing_col)).height
    print(pl.DataFrame({"filled by": steps, **filled}))

    print("\nD. fields missing x what imputation does:")
    print(
        missing.group_by("pattern", "outcome", maintain_order=True)
        .agg(rows=pl.len())
        .pivot(on="outcome", index="pattern", values="rows", sort_columns=True)
        .fill_null(0)
        .sort("pattern")
    )

    print("\nE. by outbreak year (all rows, missing, completed):")
    print(
        raw_merged_df.group_by("year")
        .agg(all_rows=pl.len())
        .join(
            missing.group_by("year").agg(
                missing=pl.len(),
                completed=pl.col("complete").fill_null(False).sum(),
            ),
            on="year",
            how="left",
        )
        .fill_null(0)
        .with_columns(
            pct_missing=(100 * pl.col("missing") / pl.col("all_rows")).round(
                1
            ),
            never_completed=pl.col("missing") - pl.col("completed"),
        )
        .sort("year")
    )


if __name__ == "__main__":
    load_dotenv()

    # %% LOAD DATA
    census_key = os.getenv("CENSUS_API_KEY")

    outbreaks_df = read_outbreaks(data_dir)
    outbreaks_df = process_outbreaks_df(outbreaks_df)
    # the 2026 cases follow the cases of 2011-2025 (the same places and
    # years keep their rows; 2026 is processed on its own)
    outbreaks_df = pl.concat(
        [
            outbreaks_df,
            process_outbreaks_df(read_outbreaks_2026(data_dir)),
        ],
        how="diagonal_relaxed",
    )

    vaccinations_df = read_vaccinations(data_dir)
    # a region stands for its counties that have no case row of their own
    # that year (a county with its own row has its own cases and its own
    # people): its vaccination rows average only those counties
    vaccinations_df = process_vaccinations_df(
        vaccinations_df, case_places=outbreaks_df
    )
    # the members with a case row of their own, by region and year, found
    # the same way (the county names of the case data are those of the
    # vaccination data)
    own_rows = members_with_own_row(
        outbreaks_df, higher_level_regions(), vaccinations_df
    )
    print(
        "\nmembers of regions with a case row of their own (left out of "
        "the region that year):"
    )
    print(
        own_rows.group_by("place", "year")
        .agg(members=pl.len())
        .sort("year", "place")
    )
    print_vaccine_data_availability(vaccinations_df, data_dir)

    # --------------------------------------------------------------
    # Write processed_vaccinations_data.csv using a state-imputed copy.
    # The original vaccinations_df is used unchanged for all downstream
    # outbreak merging and analysis.
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        "IMPUTING MISSING COUNTY FIPS FOR TARGET STATES (processed_vaccinations_data.csv only)"
    )
    print("=" * 80)
    vaccinations_df_imputed = (
        vaccination_wrangling.impute_from_state_level_coverages(
            vaccinations_df, data_dir
        )
    )
    # _is_imputed: the state values the imputation filled in (no policy
    # estimate may use them)
    vaccinations_df_imputed = vaccination_wrangling.flag_imputed_rows(
        vaccinations_df_imputed, vaccinations_df
    )
    # a table of counties: rows with a null FIPS (regions, "State" rows,
    # school districts) are not written, as every reader of this file
    # looks counties up by FIPS or by county name
    vaccinations_df_imputed.filter(pl.col("fips").is_not_null()).write_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv")
    )

    # Merge using the original (non-imputed) vaccinations_df
    raw_merged_df = merge_outbreaks_vaccinations(outbreaks_df, vaccinations_df)
    write_without_and_with_2026(
        raw_merged_df, "raw_merged_outbreaks_vaccinations"
    )
    # the 2026 cases whose county has no 2025-26 vaccination data take
    # the county's own values of 2024-25 (the main data of 2026; the
    # rows are marked in vaccine_from_earlier_year and keep a null
    # imputation_source)
    # and are trended one school year along the place's slope (its own, else
    # its state's, else the national average; data before 2024-25 only)
    fill_slopes = place_slopes(
        vaccinations_df,
        vaccination_wrangling.state_level_series(data_dir),
        county_data_year_of_outbreak(2026),
        STATES_WITH_BANS,
    )
    filled_merged_df = fill_from_earlier_school_year(
        raw_merged_df,
        vaccinations_df,
        2026,
        list(higher_level_regions()),
        slopes=fill_slopes,
    )
    print_2026_fill(raw_merged_df, filled_merged_df)
    # a filled row of a region takes the row of the school year before the
    # usual one, built for the counties of the year before: the same
    # counties, or the region's coverage would describe other people
    require_filled_rows_use_same_members(
        filled_merged_df, higher_level_regions(), own_rows
    )
    merged_df, merged_df_no_vaccine_data = require_vaccine_data(
        filled_merged_df
    )
    merged_df = assemble_analysis_dataset(merged_df)
    merged_df_no_vaccine_data = assemble_analysis_dataset(
        merged_df_no_vaccine_data
    )

    # fill in fips codes for real counties (e.g. "Kings", "Johnson") among
    # the missing-vaccine-data rows by name; leaves rows whose "county" is a
    # multi-county region name (no single matching county) unchanged
    missing_addfips = AddFIPS()
    merged_df_no_vaccine_data = merged_df_no_vaccine_data.with_columns(
        fips=pl.when(pl.col("fips").is_null())
        .then(
            pl.struct(["county", "state"]).map_elements(
                lambda row: missing_addfips.get_county_fips(
                    row["county"], row["state"]
                ),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.col("fips"))
    )

    # save the merged df no vaccine data
    write_without_and_with_2026(merged_df_no_vaccine_data, "missing_counties")

    # %% IMPUTE MISSING VACCINATION DATA FROM BORDERING YEARS
    print("\n" + "=" * 80)
    print("IMPUTING MISSING VACCINATION DATA FROM BORDERING YEARS")
    print("=" * 80)

    # Perform imputation
    imputed_from_bordering = impute_from_bordering_years(
        merged_df_no_vaccine_data,
        vaccinations_df,
        os.path.join(
            data_dir,
            "Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260526.csv",
        ),
    )

    # Combine original data with imputed data
    if imputed_from_bordering.height > 0:
        # Filter to only rows that now have complete vaccine data
        imputed_complete = imputed_from_bordering.filter(
            pl.col("estimated_mmr_cov").is_not_null()
            & pl.col("nonmed_exempt_pct").is_not_null()
            & pl.col("med_exempt_pct").is_not_null()
        )

        if imputed_complete.height > 0:
            # Add imputation_source column to original data (all None)
            merged_df = merged_df.with_columns(
                imputation_source=pl.lit(None, dtype=pl.Utf8)
            )

            # Combine for processing through pipeline
            merged_df = pl.concat(
                [merged_df, imputed_complete], how="diagonal_relaxed"
            )
            print(f"Combined dataset size: {merged_df.shape[0]} rows")
            print(
                f"  - Original (non-imputed): {merged_df.filter(pl.col('imputation_source').is_null()).shape[0]}"
            )
            print(
                f"  - Imputed: {merged_df.filter(pl.col('imputation_source').is_not_null()).shape[0]}"
            )
        else:
            merged_df = merged_df.with_columns(
                imputation_source=pl.lit(None, dtype=pl.Utf8)
            )
            print("No imputed rows had complete data after imputation")
    else:
        merged_df = merged_df.with_columns(
            imputation_source=pl.lit(None, dtype=pl.Utf8)
        )
        print("No rows could be imputed from bordering years")

    print_missing_vaccine_data_breakdown(raw_merged_df, imputed_from_bordering)

    # remove "District 1" as well
    merged_df = merged_df.filter(
        ~(
            (pl.col("county") == "District 1")
            & (pl.col("state") == "Louisiana")
        )
    )
    # (Georgia's "Fulton, Gwinett" and "Forsyth, Fulton, Paulding" are places
    # of outbreak_size.regions now, which the imputation leaves alone, so
    # they are no longer removed here)
    # remove blue ridge
    merged_df = merged_df.filter(
        ~((pl.col("county") == "Blue Ridge") & (pl.col("state") == "Virginia"))
    )

    # we need to fill in the fips codes for places that we used state-level data for
    # imputation
    addfips = AddFIPS()
    merged_df = merged_df.with_columns(
        fips=pl.when(
            pl.col("fips").is_null()
            & pl.col("imputation_source").is_not_null()
        )
        .then(
            pl.struct(["county", "state"]).map_elements(
                lambda row: addfips.get_county_fips(
                    row["county"], row["state"]
                ),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.col("fips"))
    )

    # %% VACCINATION COVERAGE AND OUTBREAK SIZE
    assert census_key is not None, (
        "A Census API key is required to run this analysis."
    )
    special_places_dict = higher_level_regions()
    special = merged_df.filter(
        pl.col("county").is_in(special_places_dict.keys())
    )
    merged_df = merged_df.filter(
        ~pl.col("county").is_in(special_places_dict.keys())
    )
    merged_df = add_pop_variables(merged_df, census_key)
    used_members = []
    for place in special_places_dict:
        temp_df = special.filter(pl.col("county") == place)
        if temp_df.height == 0:
            continue
        # one row per county of the place for every year of the place (the
        # place can have rows in several years): the counties the place
        # stands for that year, not the members with a case row of their
        # own (their cases and their people are in their own rows)
        temp_df = expand_to_residual_members(
            temp_df, place, special_places_dict, own_rows
        )
        used_members.append(
            temp_df.select(place=pl.lit(place), year="year", fips="fips")
        )
        temp_df = add_pop_variables(temp_df, census_key)
        temp_df = temp_df.with_columns(
            weight_15_up=pl.col("pop_15_up")
            / pl.sum("pop_15_up").over("year"),
            weight_25_up=pl.col("pop_25_up")
            / pl.sum("pop_25_up").over("year"),
            weight_overall=pl.col("pop") / pl.sum("pop").over("year"),
        )
        temp_df = temp_df.group_by(["year"], maintain_order=True).agg(
            state=pl.first("state"),
            county=pl.lit(place),
            cases=pl.first("cases"),
            cases_outbreak=pl.first("cases_outbreak"),
            imported=pl.first("imported"),
            imports_known=pl.first("imports_known"),
            coverage_source=pl.first("coverage_source"),
            vaccine_from_earlier_year=pl.first("vaccine_from_earlier_year"),
            indigenous=pl.first("indigenous"),
            estimated_mmr_cov=pl.first("estimated_mmr_cov"),
            nonmed_exempt_pct=pl.first("nonmed_exempt_pct"),
            med_exempt_pct=pl.first("med_exempt_pct"),
            total_exempt_pct=pl.first("total_exempt_pct"),
            pop=pl.sum("pop"),
            school_aged_population=pl.sum("school_aged_population"),
            pop_5_9=pl.sum("pop_5_9"),
            public_schooled=pl.sum("public_schooled"),
            private_schooled=pl.sum("private_schooled"),
            pop_15_up=pl.sum("pop_15_up"),
            pop_25_up=pl.sum("pop_25_up"),
            pop_under_5=pl.sum("pop_under_5"),
            number_graduate_degree=pl.sum("number_graduate_degree"),
            land_size_sq_km=pl.sum("land_size_sq_km"),
            median_household_income=(
                pl.col("median_household_income") * pl.col("weight_15_up")
            ).sum(),
            fraction_white=(
                pl.col("fraction_white") * pl.col("weight_overall")
            ).sum(),
        )
        temp_df = temp_df.with_columns(
            # public_schooled / private_schooled count the 5-9 year olds, so
            # the denominator is the 5-9 population (as for single counties)
            fraction_public=(pl.col("public_schooled") / pl.col("pop_5_9")),
            fraction_private=(pl.col("private_schooled") / pl.col("pop_5_9")),
            fraction_graduate_degree=(
                pl.col("number_graduate_degree") / pl.col("pop_25_up")
            ),
            pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
            fips=pl.lit(None),
        )
        merged_df = pl.concat([merged_df, temp_df], how="diagonal_relaxed")

    # no county may be in a region and in a row of its own: checked against
    # every case row, also those without vaccine data, by the fips the merge
    # (or the name) gave them, which also catches a member that the names
    # above did not place
    require_no_own_row_members(
        pl.concat(used_members),
        pl.concat(
            [
                merged_df.select("year", "fips"),
                merged_df_no_vaccine_data.select("year", "fips"),
            ]
        ),
    )

    merged_df = merged_df.with_columns(
        total_exempt_pct=pl.col("nonmed_exempt_pct")
        + pl.col("med_exempt_pct"),
        fraction_nonmed_exempt=pl.col("nonmed_exempt_pct")
        / pl.col("total_exempt_pct"),
        fraction_school_aged=pl.col("school_aged_population") / pl.col("pop"),
        fraction_under_5=pl.col("pop_under_5") / pl.col("pop"),
        fraction_5_9=pl.col("pop_5_9") / pl.col("pop"),
        log_income=pl.col("median_household_income").log10(),
        log_population=pl.col("pop").log10(),
        log_population_density=pl.col("pop_density").log10(),
    )

    # 5 year MMR vaccination coverage
    # add new columns
    if five_year:
        # Split into imputed and non-imputed
        imputed_rows = merged_df.filter(
            pl.col("imputation_source").is_not_null()
        )
        non_imputed_rows = merged_df.filter(
            pl.col("imputation_source").is_null()
        )

        # Add 5yr columns to imputed rows (will stay None)
        imputed_rows = imputed_rows.with_columns(
            estimated_mmr_cov_5yr=pl.lit(None),
            nonmed_exempt_pct_5yr=pl.lit(None),
            med_exempt_pct_5yr=pl.lit(None),
        )

        # Process non-imputed rows
        non_imputed_rows = non_imputed_rows.with_columns(
            estimated_mmr_cov_5yr=pl.lit(None),
            nonmed_exempt_pct_5yr=pl.lit(None),
            med_exempt_pct_5yr=pl.lit(None),
        )

        new_df = []
        for row in non_imputed_rows.iter_rows(named=True):
            if row["fips"] is None:
                # this is one of our special places
                assert row["county"] in special_places_dict.keys()
            vaccinations_5_yrs = query_avg_vaccinations_region_years(
                vaccinations_df,
                # the five school years up to the one the outbreak year uses
                years=list(
                    range(
                        county_data_year_of_outbreak(row["year"]) - 4,
                        county_data_year_of_outbreak(row["year"]) + 1,
                    )
                ),
                fips=[row["fips"]]
                if row["fips"] is not None
                else members_without_own_row(
                    special_places_dict, own_rows, row["county"], row["year"]
                ),
            )
            # update the row
            row = pl.DataFrame(row)
            row = row.with_columns(
                estimated_mmr_cov_5yr=pl.Series(
                    vaccinations_5_yrs["estimated_mmr_cov"]
                ),
                nonmed_exempt_pct_5yr=pl.Series(
                    vaccinations_5_yrs["nonmed_exempt_pct"]
                ),
                med_exempt_pct_5yr=pl.Series(
                    vaccinations_5_yrs["med_exempt_pct"]
                ),
            )
            new_df.append(row)

        # Combine back together
        non_imputed_rows = pl.concat(new_df, how="diagonal_relaxed")
        merged_df = pl.concat(
            [non_imputed_rows, imputed_rows], how="diagonal_relaxed"
        )
        # make total exemption columns
        merged_df = merged_df.with_columns(
            total_exempt_pct_5yr=pl.col("nonmed_exempt_pct_5yr")
            + pl.col("med_exempt_pct_5yr"),
        ).with_columns(
            # fractions of exemptions
            fraction_nonmed_exempt_5yr=pl.col("nonmed_exempt_pct_5yr")
            / pl.col("total_exempt_pct_5yr"),
        )

    # %% SYNTHETIC POPULATION CLUSTERS
    # let's make a geoid for easy merging later
    merged_df = merged_df.with_columns(
        geoid=pl.when(pl.col("fips").is_null())
        .then(pl.col("county"))
        .otherwise(pl.col("county") + pl.col("fips"))
    )
    # grab the counties and fips codes (by year: a region stands for other
    # counties in a year in which a member has a case row of its own)
    population_pockets = merged_df[["geoid", "year", "county", "fips"]].unique(
        maintain_order=True
    )
    population_pockets = population_pockets.with_columns(
        population_pockets_proxy=pl.lit(None),
        avg_house_size=pl.lit(None),
        std_house_size=pl.lit(None),
        # make the fips code a list for the special places
        fips=pl.struct("county", "fips", "year").map_elements(
            lambda cols: (
                [cols["fips"]]
                if cols["fips"] is not None
                else members_without_own_row(
                    special_places_dict,
                    own_rows,
                    cols["county"],
                    cols["year"],
                )
            ),
            return_dtype=pl.List(pl.Utf8),
        ),
    )

    population_pockets_proxy = []
    avg_house_size = []
    std_house_size = []
    # a list of counties is computed once, however many years it serves
    pockets_of = {}
    for fips in tqdm(population_pockets["fips"]):
        if tuple(fips) not in pockets_of:
            pockets_of[tuple(fips)] = (
                synth_pop_clusters.cluster_fips(fips)[0],
                *synth_pop_clusters.summarize_household_size(fips),
            )
        pockets_proxy, avg_house_size_fips, std_house_size_fips = pockets_of[
            tuple(fips)
        ]
        population_pockets_proxy.append(pockets_proxy)
        avg_house_size.append(avg_house_size_fips)
        std_house_size.append(std_house_size_fips)
    # merge back to main df
    population_pockets = population_pockets.with_columns(
        population_pockets_proxy=pl.Series(population_pockets_proxy),
        avg_house_size=pl.Series(avg_house_size),
        std_house_size=pl.Series(std_house_size),
    )
    n_rows = merged_df.height
    merged_df = merged_df.join(
        population_pockets[
            [
                "geoid",
                "year",
                "population_pockets_proxy",
                "avg_house_size",
                "std_house_size",
            ]
        ],
        on=["geoid", "year"],
        how="left",
        maintain_order="left",
    )
    assert merged_df.height == n_rows, (
        "a place has two rows in a year, so the join of its household sizes "
        "repeated its row"
    )

    # %% EXPORT DATA FOR DOWNSTREAM ANALYSIS
    # the rows imputed above have no mark of the 2026 fill: False
    merged_df = merged_df.with_columns(
        vaccine_from_earlier_year=pl.col(
            "vaccine_from_earlier_year"
        ).fill_null(False)
    )
    main_rows = merged_df.filter(pl.col("imputation_source").is_null())
    # a member left out of a region because it has a case row of its own
    # needs a main row (vaccine data): the main rows are the train rows of
    # the projection datasets, where a member without one is dropped as a
    # member of the region and its people are in no row
    require_own_row_members_in_main_rows(
        own_rows,
        main_rows.filter(
            pl.col("county").is_in(list(higher_level_regions()))
        ).select(place="county", year="year"),
        main_rows,
    )
    # every row that took its county's earlier school year is a row of
    # the main data (the fill must not set imputation_source)
    n_filled = filled_merged_df.filter(
        pl.col("vaccine_from_earlier_year")
        & pl.all_horizontal([pl.col(c).is_not_null() for c in VACCINE_FIELDS])
    ).height
    assert (
        main_rows.filter(pl.col("vaccine_from_earlier_year")).height
        == n_filled
    ), "a row filled from the earlier school year is not in the main data"

    # Save original data only (non-imputed)
    write_without_and_with_2026(
        merged_df.filter(pl.col("imputation_source").is_null()),
        "outbreaks_vaccinations_cleaned",
    )

    # Save full dataset (original + imputed)
    write_without_and_with_2026(
        merged_df, "outbreaks_vaccinations_cleaned_imputed"
    )

    print("\nFinal datasets saved:")
    print(
        f"  - outbreaks_vaccinations_cleaned_with_2026.csv: {merged_df.filter(pl.col('imputation_source').is_null()).shape[0]} rows (original only)"
    )
    print(
        f"  - outbreaks_vaccinations_cleaned_imputed_with_2026.csv: {merged_df.shape[0]} rows (original + imputed)"
    )
