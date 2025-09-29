import os

import matplotlib.pyplot as plt
import polars as pl
import statsmodels.api as sm
import yaml
from dotenv import load_dotenv
from matplotlib import rc

from outbreak_size import utils

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def read_outbreaks(data_dir=data_dir) -> pl.DataFrame:
    df = pl.read_excel(os.path.join(data_dir, "Combined Measles Dataset.xlsx"))

    # to ensure consistent lower case column names
    df = df.rename({col: col.lower() for col in df.columns})

    return df


def process_outbreaks_df(outbreaks_df: pl.DataFrame) -> pl.DataFrame:
    # the last row in outbreaks is a null row with the total number of cases over
    # the study period
    outbreaks_df = outbreaks_df.filter(pl.col("year").is_not_null())
    # for now, let us drop outbreaks with no case count
    # because that doesn't tell us anything about what would have happened had
    # there been an importation
    outbreaks_df = outbreaks_df.filter(pl.col("cases") != 0)

    # last column seems to be urls of sources, drop it
    outbreaks_df = outbreaks_df.select(
        ["year", "state", "county", "cases", "indigenous", "imported"]
    )

    # for now, we don't want total importations by state, so drop rows with null county
    outbreaks_df = outbreaks_df.filter(pl.col("county").is_not_null())
    # also remove counties with bogus county names
    outbreaks_df = outbreaks_df.filter(
        ~pl.col("county").is_in(["Unknown", "TOTAL", "Out of state resident"])
    )

    # there are a bunch of mispellings of places
    # correct them
    outbreaks_df = outbreaks_df.with_columns(
        [
            pl.col("county")
            .str.replace_all("San Bernadino", "San Bernardino")
            .str.replace_all("Los Angleles", "Los Angeles")
            .str.replace_all("shasta", "Shasta")
            .str.replace_all("Alamakee", "Allamakee")
            .str.replace_all("Detroit City", "Detroit")
            .str.replace_all("St Clair", "St. Clair")
            .str.replace_all("Washentaw", "Washtenaw")
            .str.replace_all("LaSueur", "Le Sueur")
            .str.replace_all("LeSueur", "Le Sueur")
            .str.replace_all("St Louis City", "St. Louis City")
            .str.replace_all("St Charles", "St. Charles")
            # Knowing that the 2019 outbreak in New York City was in Brooklyn, for now
            # we assume that NYC cases are Kings County outbreaks
            .str.replace_all("New York City", "Kings")
            .str.replace_all("Salt lake", "Salt Lake")
            .str.replace_all("Albermarle", "Albemarle")
            .str.replace_all("Spokane\xa0", "Spokane"),
        ]
    )

    # For now, drop Alaska's Gulf Coast
    outbreaks_df = outbreaks_df.filter(pl.col("county") != "Gulf Coast")

    # Detroit by itself doesn't have a fips code, so aggregate detroit up to Wayne Co
    outbreaks_df = (
        outbreaks_df.with_columns(
            [
                pl.col("county").str.replace_all("Detroit", "Wayne")
                # Now we have multiple entries for Wayne County (in 2019 and 2024)
            ]
        )
        .group_by(["year", "state", "county"])
        .agg(
            [
                # If all values in group are null, keep null; otherwise sum
                pl.when(pl.col("cases").is_null().all())
                .then(None)
                .otherwise(pl.col("cases").sum())
                .alias("cases"),
                pl.when(pl.col("indigenous").is_null().all())
                .then(None)
                .otherwise(pl.col("indigenous").sum())
                .alias("indigenous"),
                pl.when(pl.col("imported").is_null().all())
                .then(None)
                .otherwise(pl.col("imported").sum())
                .alias("imported"),
            ]
        )
    )

    return outbreaks_df


def read_vaccinations(data_dir=data_dir) -> pl.DataFrame:
    df = pl.read_excel(
        os.path.join(
            data_dir, "Compiled_Vaccine_Exemption_Data_Final_v1_092425.xlsx"
        )
    )
    # upper case lower case seem to be inconsistent across datasets
    df = df.rename({col: col.lower() for col in df.columns})

    # Put the columns in the types that we need
    df = df.with_columns(
        [
            pl.col("nonmed_exempt_pct").cast(pl.Float64, strict=False),
            pl.col("nonmed_exempt_total").cast(pl.Int64, strict=False),
            pl.col("med_exempt_pct").cast(pl.Float64, strict=False),
            pl.col("mmr_cov").cast(pl.Float64, strict=False),
            pl.col("all_vacc_cov").cast(pl.Float64, strict=False),
            pl.col("enrollment").cast(pl.Int64, strict=False),
            # fips codes are not always 5 digits -- those with leading zeros are 4
            pl.col("fips").str.zfill(5),
        ]
    )

    return df


def process_vaccinations_df(vaccinations_df: pl.DataFrame) -> pl.DataFrame:
    # when we don't have mmr coverage, use all coverage
    vaccinations_df = vaccinations_df.with_columns(
        [
            pl.when(pl.col("mmr_cov").is_null())
            .then(pl.col("all_vacc_cov"))
            .otherwise(pl.col("mmr_cov"))
            .alias("estimated_mmr_cov")
        ]
    )
    # for baltimore city in 2019, mmr coverage is 0%, but other years have data
    # so we average the surroundings years together to get an estimate
    vaccinations_df = vaccinations_df.with_columns(
        [
            pl.when(
                (pl.col("state") == "Maryland")
                & (pl.col("county") == "Baltimore City")
                & (pl.col("year") == 2019)
            )
            .then(
                pl.lit(
                    vaccinations_df.filter(
                        (pl.col("state") == "Maryland")
                        & (pl.col("county") == "Baltimore City")
                        & ((pl.col("year") == 2018) | (pl.col("year") == 2020))
                    )["estimated_mmr_cov"].mean()
                )
            )
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov")
        ]
    )

    # align namings of places
    vaccinations_df = vaccinations_df.with_columns(
        [
            pl.col("county")
            .str.replace_all("Mcdonough", "McDonough")
            .str.replace_all("Dupage", "DuPage")
            .str.replace_all("Mcculloch", "McCulloch"),
        ]
    )

    return vaccinations_df


def merge_outbreaks_vaccinations(
    outbreaks_df: pl.DataFrame, vaccinations_df: pl.DataFrame
) -> pl.DataFrame:
    merged_df = outbreaks_df.join(
        vaccinations_df, on=["year", "state", "county"], how="left"
    )
    return merged_df


def add_pop_variables(
    merged_df: pl.DataFrame, census_key: str
) -> pl.DataFrame:
    census_query = utils.CensusQuery(census_key)
    fips = merged_df["fips"].to_list()
    # make sure years are capped at 2023 since census data isn't available beyond that
    years = merged_df["year"].clip(upper_bound=2023).to_list()
    pop_df = census_query.query_5_9_pop(years, fips)
    merged_df = merged_df.with_columns(pop_df["pop_5_9"])
    # add a land size variable
    land_df = census_query.query_land_size(fips)
    merged_df = merged_df.join(land_df, on="fips", how="left")
    merged_df = merged_df.with_columns(
        (pl.col("pop_5_9") / pl.col("land_size_sq_km")).alias(
            "pop_density_5_9"
        )
    )
    return merged_df


def add_hoc_merged_df_adjustments(merged_df: pl.DataFrame) -> pl.DataFrame:
    # in this merged dataset, there is an instance of a six-digit fips code
    # it looks to me like this just fairfax county, VA (fips 51059), so truncate
    # just for now while playing around
    merged_df = merged_df.with_columns(
        [
            pl.when(pl.col("fips") == "510591")
            .then(pl.lit("51059"))
            .otherwise(pl.col("fips"))
            .alias("fips")
        ]
    )
    return merged_df


def require_vaccine_data(merged_df: pl.DataFrame) -> pl.DataFrame:
    merged_df = merged_df.filter(pl.col("estimated_mmr_cov").is_not_null())
    return merged_df


def plot_outbreak_sizes(sizes: pl.Series):
    fig, ax = plt.subplots(1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)

    # plot outbreak sizes
    ax.hist(sizes, bins=range(0, sizes.max() + 1), color="blue", alpha=0.7)  # type: ignore
    ax.set_xlabel("Outbreak Size (Number of Cases)")
    ax.set_ylabel("Number of Outbreaks")
    ax.set_yscale("log")

    return fig, ax


def compare_distributions(
    data1: pl.Series,
    data2: pl.Series,
    xlabel: str,
    label1: str,
    label2: str,
    log_scale: bool = False,
    bins=None,
):
    fig, ax = plt.subplots(1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)

    # plot outbreak sizes
    if bins is None:
        bins = range(
            min(data1.min(), data2.min()),  # type: ignore
            max(data1.max(), data2.max()) + 1,  # type: ignore
        )
    ax.hist(data1, bins=bins, color="blue", alpha=0.7, label=label1)
    ax.hist(data2, bins=bins, color="red", alpha=0.7, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Outbreaks")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()

    return fig, ax


def plot_cases_vaccinations(cases: pl.Series, vaccinations: pl.Series, ylabel):
    fig, ax = plt.subplots(1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)

    # plot cases and vaccinations
    ax.scatter(vaccinations, cases, c="blue")
    ax.set_xlabel("MMR Vaccination Coverage (%)")
    ax.set_ylabel(ylabel)
    if cases.min() > 0:  # type: ignore
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog")

    return fig, ax


def assemble_analysis_dataset(
    merged_df: pl.DataFrame, params: dict = params
) -> pl.DataFrame:
    # keep county-years where imports equal observed cases
    include = pl.col("imported") == pl.col("cases")
    # keep county-years where the number of known importations is less than cases
    include = include | (pl.col("imported") < pl.col("cases"))
    # keep county-years above the inclusion threshold
    include = include | (pl.col("cases") >= params["inclusion_threshold"])
    merged_df = merged_df.filter(include)

    # Now make a new column of cases that is the number of cases generated in the
    # outbreak -- basically excluding the imports
    merged_df = merged_df.with_columns(
        [
            pl.when(pl.col("imported").is_null())
            .then(
                pl.col("cases") - 1
            )  # if we don't know imports, assume 1 import
            .otherwise(pl.col("cases") - pl.col("imported"))
            .alias("cases_outbreak")
        ]
    )

    return merged_df


if __name__ == "__main__":
    load_dotenv()
    census_key = os.getenv("CENSUS_API_KEY")

    outbreaks_df = read_outbreaks(data_dir)
    outbreaks_df = process_outbreaks_df(outbreaks_df)

    # outbreak sizes
    fig, ax = plot_outbreak_sizes(outbreaks_df["cases"])
    plt.show()

    vaccinations_df = read_vaccinations(data_dir)
    vaccinations_df = process_vaccinations_df(vaccinations_df)
    raw_merged_df = merge_outbreaks_vaccinations(outbreaks_df, vaccinations_df)
    raw_merged_df = add_hoc_merged_df_adjustments(raw_merged_df)

    # Compare distributions of outbreak sizes with and without vaccination data
    merged_df = require_vaccine_data(raw_merged_df)

    fig, ax = compare_distributions(
        raw_merged_df["cases"],
        merged_df["cases"],
        xlabel="Cases",
        label1="Without Vaccination Data",
        label2="With Vaccination Data",
        log_scale=True,
        bins=[1, 2, 3, 4, 5, 10, 20, 50, 750],
    )
    plt.show()

    # Compare distribution of timing of outbreaks with and without vaccination data
    fig, ax = compare_distributions(
        raw_merged_df["year"],
        merged_df["year"],
        xlabel="Year of Outbreak",
        label1="Without Vaccination Data",
        label2="With Vaccination Data",
        log_scale=False,
    )
    plt.show()

    # subset to at least 3 cases
    # merged_df = merged_df.filter(pl.col("cases") >= params['inclusion_threshold'])
    merged_df = assemble_analysis_dataset(merged_df, params)

    # plot cases vs vaccination coverage
    fig, ax = plot_cases_vaccinations(
        merged_df["cases_outbreak"],
        merged_df["estimated_mmr_cov"],
        ylabel="Cases",
    )
    plt.show()
    # linear regression between the two
    exog = sm.add_constant(merged_df["estimated_mmr_cov"].to_numpy())
    model = sm.OLS(merged_df["cases_outbreak"].log().to_numpy(), exog).fit()
    print(model.summary())

    # add a population variable to plot incidence
    assert census_key is not None, (
        "A Census API key is required to run this analysis."
    )
    merged_df = add_pop_variables(merged_df, census_key)

    fig, ax = plot_cases_vaccinations(
        merged_df["cases_outbreak"] / merged_df["pop_5_9"],
        merged_df["estimated_mmr_cov"],
        ylabel="Incidence (Cases per 5-9 year olds)",
    )
    plt.show()
    # linear regression between the two
    exog = sm.add_constant(merged_df["estimated_mmr_cov"].to_numpy())
    model = sm.OLS(
        (merged_df["cases_outbreak"] / merged_df["pop_5_9"]).log().to_numpy(),
        exog,
    ).fit()
    print(model.summary())

    # plot cases vs vaccination coverage with incidence by density
    fig, ax = plot_cases_vaccinations(
        merged_df["cases_outbreak"] / merged_df["pop_density_5_9"],
        merged_df["estimated_mmr_cov"],
        ylabel=r"Incidence per km$^2$",
    )
    plt.show()
    # linear regression between the two
    exog = sm.add_constant(merged_df["estimated_mmr_cov"].to_numpy())
    model = sm.OLS(
        (merged_df["cases_outbreak"] / merged_df["pop_density_5_9"])
        .log()
        .to_numpy(),
        exog,
    ).fit()
    print(model.summary())

    merged_df.write_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )
