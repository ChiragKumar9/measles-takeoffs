import os
import pickle

import polars as pl
import yaml
from dotenv import load_dotenv

from outbreak_size import import_data, import_validation
from outbreak_size.data_year import county_data_year_of_outbreak
from outbreak_size.importation import (
    calculate_yearly_import_probability,
    distribute_state_level_risk_to_county,
)
from outbreak_size.regions import higher_level_regions

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def read_outbreaks(
    data_dir=data_dir,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_excel(
        os.path.join(data_dir, "Combined Measles Dataset QCed.xlsx")
    )
    df_2025 = pl.read_csv(
        os.path.join(data_dir, "2025CasesImports QCed clean.csv"),
        schema_overrides={"fips": pl.String},
    )
    return import_data.clean_case_records(df, df_2025)


def read_state_year_imports(df_2025, data_dir=data_dir) -> pl.DataFrame:
    """The state-level imports of each year (state, year, imported, ...)."""
    df = pl.read_excel(
        os.path.join(data_dir, "IndigenousImportedFullData.xlsx")
    )
    return import_data.clean_state_year_imports(df, df_2025)


def tally_state_importations(
    df_2025, data_dir=data_dir
) -> tuple[pl.DataFrame, int]:
    return import_data.tally_state_importations(
        read_state_year_imports(df_2025, data_dir)
    )


def write_import_validation_county_years(
    known_county_rows, df_2025, state_posterior
):
    """
    Write the county-years the importation model is validated on: every one
    with known imports, also the counties that are not in the analysis or
    projection data (see outbreak_size/import_validation.py), each with its
    state's pre-2021 posterior times its share of the state.

    A county's share here is its population over the population of all the
    counties of its state in the 2020 shapefile, so it differs slightly from
    the share used for the projections (over the counties of the 2026 rows)
    in states where some counties are missing from those rows. The alpha
    columns hold the state's alpha times the share, so the county's rate is
    share * Gamma(alpha, beta), as for the projections.
    """
    from outbreak_size import utils

    years = [2021, 2022, 2023, 2024, 2025]
    universe = import_validation.county_universe(data_dir)
    resolved, dropped = import_validation.resolve_known_rows(
        known_county_rows.filter(pl.col("year") >= years[0]),
        universe,
        higher_level_regions(),
    )
    print(
        f"import validation: {dropped.height} known rows match no county or "
        f"region and are left out: {dropped.rows()}"
    )
    state_year_totals = (
        read_state_year_imports(df_2025)
        .group_by(["state", "year"])
        .agg(total=pl.sum("imported"))
    )

    # the population of each county in each year: the one of the projection
    # dataset of that year, the rest (counties that are not in it) from the
    # Census
    populations = pl.concat(
        [
            pl.read_csv(
                os.path.join(output_dir, f"projection_dataset_{year}.csv"),
                schema_overrides={"fips": pl.String},
            ).select("fips", pl.lit(year, dtype=pl.Int64).alias("year"), "pop")
            for year in years
        ]
    ).unique(subset=["fips", "year"], keep="first", maintain_order=True)

    def ask_census(query_years, fips):
        load_dotenv()
        census_key = os.getenv("CENSUS_API_KEY")
        if census_key is None:
            raise ValueError(
                "CENSUS_API_KEY is needed for the populations of counties "
                "that are not in the projection datasets"
            )
        return (
            utils.CensusQuery(
                census_key,
                cache_path=utils.CENSUS_CACHE_PATH,
                workers=utils.CENSUS_WORKERS,
            )
            # the census of the year before each outbreak year, as the
            # populations of the projection datasets
            .query_5_9_pop(
                [county_data_year_of_outbreak(y) for y in query_years], fips
            )["pop"]
            .to_list()
        )

    populations = import_validation.fill_missing_populations(
        universe, years, populations, ask_census
    )
    rows = import_data.attach_state_posterior(
        import_validation.validation_county_years(
            resolved, dropped, state_year_totals, universe, populations, years
        ),
        state_posterior,
    )
    print(
        "import validation county-years (positives = imported >= 1):\n"
        + str(
            rows.group_by("source")
            .agg(
                county_years=pl.len(),
                positives=(pl.col("imported") >= 1).sum(),
            )
            .sort("source")
        )
    )
    rows.write_csv(
        os.path.join(output_dir, "import_validation_county_years.csv")
    )


if __name__ == "__main__":
    counties_with_known_imports, df_2025 = read_outbreaks()
    # no import is known after 2025 (the importation posteriors count the years
    # of data up to 2025, tally_state_importations)
    assert counties_with_known_imports["year"].max() <= 2025  # type: ignore

    df, years_of_data = tally_state_importations(df_2025)
    df = calculate_yearly_import_probability(df, years_of_data)
    write_import_validation_county_years(
        counties_with_known_imports, df_2025, df
    )
    # read in our county data
    validation_years = [2021, 2022, 2023, 2024, 2025, 2026]
    projection_frames = {
        year: pl.read_csv(
            os.path.join(output_dir, f"projection_dataset_{year}.csv"),
            schema_overrides={"fips": pl.String},
        )
        for year in validation_years
    }
    county_data = import_data.stack_county_years(projection_frames)
    df, counties_with_known_imports = distribute_state_level_risk_to_county(
        df, county_data, counties_with_known_imports, params["seed"]
    )
    # save
    # save the values separately from the samples

    samples = df.select("importation_probability_samples")
    importation_count_samples = df.select("expected_importations_samples")
    df = df.drop(
        ["importation_probability_samples", "expected_importations_samples"]
    )

    with open(
        os.path.join(
            output_dir, "empirical_importation_probability_samples.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(samples, f)

    with open(
        os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
        "wb",
    ) as f:
        pickle.dump(importation_count_samples, f)

    df.write_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv")
    )
    counties_with_known_imports.write_csv(
        os.path.join(output_dir, "counties_with_known_imports.csv")
    )
