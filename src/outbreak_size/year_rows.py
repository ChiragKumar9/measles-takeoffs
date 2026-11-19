"""
The places of one validation year, with their observed cases and the imports
drawn for them: the rows that the draws of every model of the year are scored
on (analysis/table_model_performance_validation.py, and the scoring of the
headline model on 2026).
"""

import os

import numpy as np
import polars as pl

from outbreak_size.importation import draw_imports
from outbreak_size.regions import higher_level_regions
from outbreak_size.validation import (
    projection_twin_mask,
    region_member_mask,
    require_no_nulls,
    require_unique_places,
)


def validation_year_rows(
    year: int,
    output_dir: str,
    import_df: pl.DataFrame,
    analysis: pl.DataFrame,
    seed: int,
    covariates: list[str],
):
    """
    The places of the validation year `year`: the rows of
    projection_dataset_{year}.csv, one per county-year, with the importation
    samples of the validation year and the observed cases.

    `import_df` has the importation rows with their samples
    (importation_probability_samples and expected_importations_samples);
    `analysis` has the observed county-years (the analysis dataset and the
    counties without vaccine data), with the columns state_county_year,
    cases_outbreak and cases (a county-year is keyed by fips and year, a place
    without a county fips by name and year). `covariates` are the columns kept
    for the place files.

    Returns the rows, the boolean array of the rows of the file that are kept
    (to apply to the draws of the models, which have a column for every row of
    the file) and the imports drawn for the places (n_draws, n_places), the
    same for every model of the year.
    """
    temp = pl.read_csv(
        os.path.join(output_dir, f"projection_dataset_{year}.csv"),
        schema_overrides={"fips": pl.String},
    )
    temp = temp.with_columns(validation_year=pl.lit(year))

    temp = temp.select(
        [
            "county",
            "state",
            "year",
            "validation_year",
            "cases_outbreak",
            "type",
            "fips",
            "pop",
            # the sociodemographic and vaccination covariates, for the place file
            *covariates,
        ]
    )
    require_no_nulls(temp, ["pop"], f"population of the {year} places")

    # keyed on fips and type: county names are not unique within a state
    # (the counties of Delaware, Rhode Island, West Virginia, ... are
    # named "State" or by region in the exports), and a train row must not
    # take the importation row of a projection row or the other way round
    temp = temp.join(
        import_df.select(
            [
                "fips",
                "validation_year",
                "type",
                "importation_probability_samples",
                "expected_importations_samples",
            ]
        ),
        on=["fips", "validation_year", "type"],
        how="left",
        maintain_order="left",
    )
    require_no_nulls(
        temp,
        [
            "importation_probability_samples",
            "expected_importations_samples",
        ],
        f"importation rows of the {year} validation rows",
    )

    # 05 writes every observed county-year twice: as a train row (the one
    # matched to the observed outbreak) and as a projection row (like
    # every other county). The two have the same importation rate (a
    # share of the state's rate over all its counties). Keep one row per
    # county-year, the train row, and drop the projection twin
    keep = ~projection_twin_mask(temp)
    # a region (a place of several counties, such as a health district)
    # is one row with its own prediction and the observed cases of the
    # whole region: drop the rows of its member counties, so the same
    # places are not predicted twice (a member with an observed row of
    # its own stays)
    is_member = region_member_mask(
        temp.filter(pl.Series(keep)), higher_level_regions()
    )
    print(f"{year}: {int(is_member.sum())} member rows of regions dropped")
    keep[keep] = ~is_member
    temp = temp.filter(pl.Series(keep))
    require_unique_places(temp, f"{year} validation rows")

    temp = temp.with_columns(
        fips_year=pl.concat_str(
            [pl.col("fips"), pl.col("validation_year")], separator="_"
        ),
        name_year=pl.concat_str(
            [pl.col("state"), pl.col("county"), pl.col("validation_year")],
            separator="_",
        ),
    )
    # same key as in analysis: fips and year, or name and year for the
    # places analysis keys by name
    temp = temp.with_columns(
        state_county_year=pl.when(
            pl.col("fips_year").is_in(analysis["state_county_year"])
        )
        .then(pl.col("fips_year"))
        # only a place without a county fips may match by name
        .when((pl.col("fips").str.len_chars() != 5).fill_null(True))
        .then(pl.col("name_year"))
        .otherwise(pl.col("fips_year"))
    ).drop("fips_year", "name_year")
    temp = temp.with_columns(
        type=pl.when(
            pl.col("state_county_year").is_in(analysis["state_county_year"])
        )
        .then(pl.lit("train"))
        .otherwise(pl.col("type"))
    )
    # year of these places also needs to be updated to be the current year
    temp = temp.with_columns(
        year=pl.when(pl.col("type") == "train")
        .then(pl.col("validation_year"))
        .otherwise(pl.col("year"))
    )
    # update cases outbreak as well
    temp = temp.with_columns(
        cases_outbreak=pl.when(pl.col("type") == "train")
        # get the particular cases outbreak for this state county year from the train data
        .then(
            pl.col("state_county_year").replace(
                dict(
                    zip(
                        analysis["state_county_year"],
                        analysis["cases_outbreak"],
                    )
                )
            )
        )
        .otherwise(pl.lit(0))
    )

    # cases of the observed places, from the same key
    temp = temp.with_columns(
        cases=pl.when(pl.col("type") == "train")
        .then(
            pl.col("state_county_year").replace_strict(
                dict(
                    zip(
                        analysis["state_county_year"],
                        analysis["cases"],
                    )
                ),
                default=0,
                return_dtype=pl.Int64,
            )
        )
        .otherwise(pl.lit(0))
    )

    import_counts = np.vstack(
        temp["expected_importations_samples"].to_numpy()  # type: ignore
    ).T

    # in every draw the importation probability is 1 - exp(-expected
    # importations), the chance that a Poisson number of imports with
    # that mean is at least one
    import_probability = np.vstack(
        temp["importation_probability_samples"].to_numpy()  # type: ignore
    ).T
    np.testing.assert_allclose(
        import_probability,
        1 - np.exp(-np.asarray(import_counts)),
        atol=1e-6,
    )
    # a county's cases in a draw: its imported cases and, if there is at
    # least one, the outbreak that follows (see total_case_draws)
    import_draws = draw_imports(
        np.asarray(import_counts),
        np.random.default_rng([seed, year]),
    )
    return temp, keep, import_draws


def observed_county_years(analysis, missing):
    """
    The observed county-years that validation_year_rows matches to the places:
    the cases of the analysis dataset and of the counties without vaccine data
    (from 2021 on, the validation years), each with its key `state_county_year`
    (fips and year, or state, county and year for a place without a county
    fips, as the places of the validation rows are keyed).
    """
    return pl.concat(
        [analysis, missing.filter(pl.col("year") >= 2021)],
        how="diagonal_relaxed",
    ).with_columns(
        state_county_year=pl.when(pl.col("fips").str.len_chars() == 5)
        .then(pl.concat_str([pl.col("fips"), pl.col("year")], separator="_"))
        .otherwise(
            pl.concat_str(
                [pl.col("state"), pl.col("county"), pl.col("year")],
                separator="_",
            )
        )
    )
