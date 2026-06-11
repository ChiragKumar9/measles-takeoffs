import glob
import os

import geopandas as gpd
import polars as pl
from addfips import AddFIPS


# kansas data is special because it's usually at the school district level,
# but we need it at the county level
# So we hda it queried separately at the county level
def read_kansas_data(data_dir):
    # get all MMR coverage data
    mmr_files = glob.glob(os.path.join(data_dir, "MMR", "**", "*.csv"))
    kansas_mmr_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in mmr_files
    ]
    kansas_mmr = pl.concat(kansas_mmr_dfs)
    # do the same for exemption data
    nme_files = glob.glob(
        os.path.join(data_dir, "Religious Exemption", "**", "*.csv")
    )
    kansas_nme_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in nme_files
    ]
    kansas_nme = pl.concat(kansas_nme_dfs)
    # do the same for medical exemption data
    me_files = glob.glob(
        os.path.join(data_dir, "Medical Exemption", "**", "*.csv")
    )
    kansas_me_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in me_files
    ]
    kansas_me = pl.concat(kansas_me_dfs)

    # now we have to clean and merge
    kansas_mmr = kansas_mmr.with_columns(
        mmr_cov=pl.col("County Coverage")
        .str.replace("%", "")
        .cast(pl.Float64, strict=False),
        year=pl.col("Academic Year")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False),
    ).select(["County", "mmr_cov", "year"])

    kansas_nme = kansas_nme.with_columns(
        nonmed_exempt_pct=pl.col("Exemption String")
        .str.replace("%", "")
        .cast(pl.Float64, strict=False),
        year=pl.col("Academic Year")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False),
    ).select(["County", "nonmed_exempt_pct", "year"])

    kansas_me = kansas_me.with_columns(
        med_exempt_pct=pl.col("Exemption String")
        .str.replace("%", "")
        .cast(pl.Float64, strict=False),
        year=pl.col("Academic Year")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False),
    ).select(["County", "med_exempt_pct", "year"])

    kansas_vacc = kansas_mmr.join(
        kansas_nme, on=["County", "year"], how="inner"
    ).join(kansas_me, on=["County", "year"], how="inner")

    kansas_vacc = kansas_vacc.with_columns(state=pl.lit("Kansas"))

    # we need to add fips codes into this data
    addfips = AddFIPS()
    kansas_vacc = kansas_vacc.with_columns(
        fips=pl.col("County").map_elements(
            lambda county: addfips.get_county_fips(county, "Kansas")
        )
    )

    # rename "County" to be "county" for merging purposes
    kansas_vacc = kansas_vacc.rename({"County": "county"})

    return kansas_vacc


def read_montana_data(data_dir):
    montana = pl.read_excel(
        os.path.join(data_dir, "Montana.xlsx"), read_options={"skip_rows": 1}
    )

    # make the column names actually useful
    # first of all, the columns are:
    columns = [
        "year",
        "state",
        "county",
        "Race",
        "nonmed_exempt_pct",
        "nonmed_exempt_num",
        "med_exempt_pct",
        "MMR_cov",
        "all_vacc_cov",
        "number",
    ]
    # set the column names
    montana.columns = columns
    # drop things we don't need
    montana = montana.drop(["Race", "nonmed_exempt_num", "number"])

    # sometimes the county names are in all caps, sometimes they're not, so let's
    # make them normal case for merging purposes
    montana = montana.with_columns(county=pl.col("county").str.to_titlecase())

    # we notice that we only have coverage data for 2019, and we only have exemption
    # data prior to that
    # so let's just fill in the missing exemption data for 2019 with the exemption
    # data for 2018

    # this is ok because we are only using this data for risk projections
    montana = montana.sort(["county", "year"]).with_columns(
        pl.col("nonmed_exempt_pct").forward_fill().over("county"),
        pl.col("med_exempt_pct").forward_fill().over("county"),
    )

    # case mmr cov and all vacc cov as floats
    montana = montana.with_columns(
        MMR_cov=pl.col("MMR_cov").cast(pl.Float64, strict=False),
        all_vacc_cov=pl.col("all_vacc_cov").cast(pl.Float64, strict=False),
    )

    # since we can't really QC this data, we want to remove any places that have
    # mmr cov equal to 0, since that's probably a data error
    montana = montana.filter(pl.col("MMR_cov") > 0)
    montana = montana.filter(pl.col("all_vacc_cov") > 0)

    # since we do the merging of estimated_mmr_cov post adding in the montana data,
    # do it here
    montana = montana.with_columns(
        # use mmr cov when available or not zero, otherwise use all vacc cov
        estimated_mmr_cov=pl.when(
            pl.col("MMR_cov").is_not_null() & (pl.col("MMR_cov") > 0)
        )
        .then(pl.col("MMR_cov"))
        .otherwise(pl.col("all_vacc_cov"))
    )

    return montana


# Mapping of 2-digit state FIPS codes to full state names for the 13 target states
_TARGET_STATE_FIPS_TO_NAME = {
    "41": "Oregon",
    "32": "Nevada",
    "49": "Utah",
    "30": "Montana",
    "31": "Nebraska",
    "48": "Texas",
    "22": "Louisiana",
    "46": "South Dakota",
    "38": "North Dakota",
    "51": "Virginia",
    "50": "Vermont",
    "02": "Alaska",
    "29": "Missouri",
}

_TARGET_STATES = set(_TARGET_STATE_FIPS_TO_NAME.values())


def impute_from_state_level_coverages(
    vaccination_data: pl.DataFrame,
    data_dir: str,
) -> pl.DataFrame:
    """
    For 13 target states, ensure every county-level FIPS code has data for
    ALL years present in the dataset. Fills missing counties and years with
    state-level values from state_level_coverages.csv, using year-specific
    values when available.

    This operates on and returns a COPY of vaccination_data. The original is
    not modified.

    state_level_coverages.csv must have columns: year, state, coverage, nme, me
    """
    vaccination_data = vaccination_data.clone()

    # ------------------------------------------------------------------
    # 1. Load state-level fallback values (now with year)
    # ------------------------------------------------------------------
    state_coverages = pl.read_csv(
        os.path.join(data_dir, "state_level_coverages.csv")
    )

    # Create lookup: {(state, year): {coverage, nme, me}}
    state_lookup: dict[tuple[str, int], dict] = {}
    for row in state_coverages.iter_rows(named=True):
        state_lookup[(row["state"], row["year"])] = {
            "estimated_mmr_cov": float(row["coverage"]),
            "nonmed_exempt_pct": float(row["nme"]),
            "med_exempt_pct": float(row["me"]),
        }

    # Also create a fallback with just the most recent year per state
    state_latest: dict[str, dict] = {}
    for state_name in _TARGET_STATES:
        state_years = [
            (s, y) for (s, y) in state_lookup.keys() if s == state_name
        ]
        if state_years:
            latest_year = max(y for _, y in state_years)
            state_latest[state_name] = state_lookup[(state_name, latest_year)]

    # ------------------------------------------------------------------
    # 2. Load shapefile to get ALL county FIPS for target states
    # ------------------------------------------------------------------
    shapefile_path = os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    counties_gdf = gpd.read_file(shapefile_path)

    target_statefps = set(_TARGET_STATE_FIPS_TO_NAME.keys())
    target_counties = counties_gdf[
        counties_gdf["STATEFP"].isin(target_statefps)
    ][["STATEFP", "GEOID", "NAME"]].copy()
    target_counties["state"] = target_counties["STATEFP"].map(
        _TARGET_STATE_FIPS_TO_NAME
    )

    # ------------------------------------------------------------------
    # 3. Get all years in the dataset
    # ------------------------------------------------------------------
    all_years = sorted(
        vaccination_data.filter(pl.col("year").is_not_null())["year"]
        .unique()
        .to_list()
    )
    print(f"Years in vaccination_data: {min(all_years)} to {max(all_years)}")

    # ------------------------------------------------------------------
    # 4. Create complete skeleton: all target counties × all years
    #    Using year-specific state values when available
    # ------------------------------------------------------------------
    complete_rows = []
    for _, row in target_counties.iterrows():
        fips = row["GEOID"]
        state_name = row["state"]
        county_name = row["NAME"]

        if state_name not in state_latest:
            print(
                f"WARNING: No state-level coverage for '{state_name}' - skipping"
            )
            continue

        for year in all_years:
            # Try to get year-specific values, fall back to latest
            if (state_name, year) in state_lookup:
                vals = state_lookup[(state_name, year)]
            else:
                # Use the most recent available year for this state
                vals = state_latest[state_name]

            complete_rows.append(
                {
                    "fips": fips,
                    "state": state_name,
                    "county": county_name,
                    "year": year,
                    "estimated_mmr_cov": vals["estimated_mmr_cov"],
                    "nonmed_exempt_pct": vals["nonmed_exempt_pct"],
                    "med_exempt_pct": vals["med_exempt_pct"],
                    "_is_imputed": True,
                }
            )

    print(
        f"Created {len(complete_rows)} skeleton rows for "
        f"{len(target_counties)} counties × {len(all_years)} years"
    )

    skeleton_df = pl.DataFrame(
        complete_rows,
        schema_overrides={
            "fips": pl.String,
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "_is_imputed": pl.Boolean,
        },
    )

    # ------------------------------------------------------------------
    # 5. Mark existing data and combine
    # ------------------------------------------------------------------
    vaccination_data = vaccination_data.with_columns(_is_imputed=pl.lit(False))

    combined = pl.concat(
        [vaccination_data, skeleton_df], how="diagonal_relaxed"
    )

    # ------------------------------------------------------------------
    # 6. Prioritize real data over imputed: group by fips+year, take first non-imputed
    # ------------------------------------------------------------------
    combined = (
        combined.sort("_is_imputed")  # False (real) comes before True
        .group_by(["fips", "year"], maintain_order=True)
        .agg(pl.all().first())  # Takes first row = real data preferred
    )

    # ------------------------------------------------------------------
    # 7. Fill any remaining nulls with state-level values (year-specific)
    # ------------------------------------------------------------------
    def get_state_value(state: str, year: int, key: str) -> float:
        """Get state-level value for a given year, falling back to latest"""
        if (state, year) in state_lookup:
            return state_lookup[(state, year)].get(key)  # type: ignore
        elif state in state_latest:
            return state_latest[state].get(key)  # type: ignore
        return None  # type: ignore

    for col, lookup_key in [
        ("estimated_mmr_cov", "estimated_mmr_cov"),
        ("nonmed_exempt_pct", "nonmed_exempt_pct"),
        ("med_exempt_pct", "med_exempt_pct"),
    ]:
        combined = combined.with_columns(
            pl.when(pl.col(col).is_null())
            .then(
                pl.struct(["state", "year"]).map_elements(
                    lambda x: get_state_value(
                        x["state"], x["year"], lookup_key
                    ),
                    return_dtype=pl.Float64,
                )
            )
            .otherwise(pl.col(col))
            .alias(col)
        )

    combined = combined.drop("_is_imputed")

    # ------------------------------------------------------------------
    # 8. Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IMPUTATION SUMMARY BY STATE")
    print("=" * 80)
    for state_name in sorted(_TARGET_STATES):
        state_rows = combined.filter(
            (pl.col("state") == state_name)
            & (pl.col("fips").str.len_chars() == 5)
        )
        n_fips = state_rows["fips"].n_unique()
        n_nulls = state_rows.filter(
            pl.col("estimated_mmr_cov").is_null()
            | pl.col("nonmed_exempt_pct").is_null()
            | pl.col("med_exempt_pct").is_null()
        ).height

        # Show year coverage
        years_with_data = sorted(state_rows["year"].unique().to_list())
        year_range = (
            f"{min(years_with_data)}-{max(years_with_data)}"
            if years_with_data
            else "none"
        )

        print(
            f"  {state_name:20s}: {n_fips:3d} counties, "
            f"years {year_range}, {n_nulls:4d} rows with remaining nulls"
        )
    print("=" * 80 + "\n")

    return combined
