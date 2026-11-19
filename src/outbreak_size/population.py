"""Population variables of counties."""

import polars as pl


def add_census_ratios(merged_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the ratios the models use to a frame with the census counts of each
    county: pop, land_size_sq_km, public_schooled, private_schooled and pop_5_9
    (children aged 5-9), number_graduate_degree and pop_25_up (people aged 25
    and over), school_aged_population, pop_under_5 and median_household_income.
    The population shares of the school-aged groups are of the total population
    pop; the school and degree fractions are of their own age group.
    """
    return merged_df.with_columns(
        pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
        # these quantities are calculated only for people aged 5-9
        fraction_public=(pl.col("public_schooled") / pl.col("pop_5_9")),
        fraction_private=(pl.col("private_schooled") / pl.col("pop_5_9")),
        # public + private + not enrolled add up to the age group's total
        fraction_graduate_degree=(
            pl.col("number_graduate_degree") / pl.col("pop_25_up")
        ),
        fraction_school_aged=pl.col("school_aged_population") / pl.col("pop"),
        fraction_under_5=pl.col("pop_under_5") / pl.col("pop"),
        fraction_5_9=pl.col("pop_5_9") / pl.col("pop"),
        log_income=pl.col("median_household_income").log10(),
        log_population=pl.col("pop").log10(),
    ).with_columns(log_population_density=pl.col("pop_density").log10())
