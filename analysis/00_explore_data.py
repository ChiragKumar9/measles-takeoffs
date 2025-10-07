import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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
            # If all values in group are null, keep null; otherwise sum
            cases=pl.when(pl.col("cases").is_null().all())
            .then(None)
            .otherwise(pl.col("cases").sum()),
            indigenous=pl.when(pl.col("indigenous").is_null().all())
            .then(None)
            .otherwise(pl.col("indigenous").sum()),
            imported=pl.when(pl.col("imported").is_null().all())
            .then(None)
            .otherwise(pl.col("imported").sum()),
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
        estimated_mmr_cov=pl.when(pl.col("mmr_cov").is_null())
        .then(pl.col("all_vacc_cov"))
        .otherwise(pl.col("mmr_cov"))
    )
    # for baltimore city in 2019, mmr coverage is 0%, but other years have data
    # so we average the surroundings years together to get an estimate
    vaccinations_df = vaccinations_df.with_columns(
        estimated_mmr_cov=pl.when(
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
    years_census = merged_df["year"].clip(upper_bound=2023).to_list()
    pop_df = census_query.query_5_9_pop(years_census, fips)
    merged_df = merged_df.with_columns(
        pop_df[
            [
                "pop_5_9",
                "public_schooled_5_9",
                "private_schooled_5_9",
                "not_enrolled_5_9",
                "median_household_income",
                "fraction_graduate_degree",
                "fraction_white",
            ]
        ]
    )
    # add a land size variable
    land_df = census_query.query_land_size(fips)
    merged_df = merged_df.join(land_df, on="fips", how="left")
    years_mobility = merged_df["year"].to_list()
    mobility_df = census_query.query_within_county_mobility(
        years_mobility, fips
    )
    merged_df = merged_df.with_columns(
        year_mobility=pl.col("year").clip(lower_bound=2019)
    )
    merged_df = merged_df.join(
        mobility_df,
        left_on=["year_mobility", "fips"],
        right_on=["year", "fips"],
        how="left",
    )
    merged_df = merged_df.with_columns(
        pop_density_5_9=(pl.col("pop_5_9") / pl.col("land_size_sq_km")),
        fraction_public=(pl.col("public_schooled_5_9") / pl.col("pop_5_9")),
        fraction_private=(pl.col("private_schooled_5_9") / pl.col("pop_5_9")),
        # I previously checked by hand that public+private+not_enrolled = total pop 5-9
        fraction_not_enrolled=(pl.col("not_enrolled_5_9") / pl.col("pop_5_9")),
        within_county_mobility_proxy=pl.col("within_county_estimate")
        / pl.mean("within_county_estimate"),
    )
    return merged_df


def add_hoc_merged_df_adjustments(merged_df: pl.DataFrame) -> pl.DataFrame:
    # in this merged dataset, there is an instance of a six-digit fips code
    # it looks to me like this just fairfax county, VA (fips 51059), so truncate
    # just for now while playing around
    merged_df = merged_df.with_columns(
        fips=pl.when(pl.col("fips") == "510591")
        .then(pl.lit("51059"))
        .otherwise(pl.col("fips"))
    )
    return merged_df


def require_vaccine_data(merged_df: pl.DataFrame) -> pl.DataFrame:
    merged_df = merged_df.filter(
        pl.col("estimated_mmr_cov").is_not_null()
        & pl.col("nonmed_exempt_pct").is_not_null()
        & pl.col("med_exempt_pct").is_not_null()
    )
    return merged_df


def compare_distributions(
    data1: pl.Series,
    data2: pl.Series,
    xlabel: str,
    label1: str,
    label2: str,
    log_scale: bool = False,
    bins=None,
    density=False,
    cumulative=False,
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
    ax.hist(
        data1,
        bins=bins,
        color="blue",
        alpha=0.7,
        label=label1,
        density=density,
        cumulative=cumulative,
    )
    ax.hist(
        data2,
        bins=bins,
        color="red",
        alpha=0.7,
        label=label2,
        density=density,
        cumulative=cumulative,
    )
    ax.set_xlabel(xlabel)
    if density:
        if cumulative:
            ax.set_ylabel("Cumulative probability density")
        else:
            ax.set_ylabel("Probability density")
    else:
        if cumulative:
            ax.set_ylabel("Cumulative number of outbreaks")
        else:
            ax.set_ylabel("Number of outbreaks")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()

    return fig, ax


def pairplot(
    x: pl.Series,
    y: pl.Series,
    ylabel,
    xlabel="MMR vaccination coverage (%)",
    log=True,
    **kwargs,
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

    # plot cases and vaccinations
    if "c" not in kwargs:
        kwargs["c"] = "blue"
    scatter = ax.scatter(x, y, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        if y.min() > 0:  # type: ignore
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog")

    return fig, ax, scatter


def assemble_analysis_dataset(
    merged_df: pl.DataFrame, params: dict = params
) -> pl.DataFrame:
    # keep county-years with 1 case -- no transmission, just an import
    include = pl.col("cases") == 1
    # keep county-years where imports equal observed cases
    # these are also years with no transmission
    include = include | (pl.col("imported") == pl.col("cases"))
    # keep county-years where the number of known importations is less than cases
    # (with the above requirement, this condenses to cases where imports are known)
    include = include | (pl.col("imported") < pl.col("cases"))
    # keep county-years above the inclusion threshold
    include = include | (pl.col("cases") >= params["inclusion_threshold"])
    merged_df = merged_df.filter(include)

    # Now make a new column of cases that is the number of cases generated in the
    # outbreak -- basically excluding the imports
    merged_df = merged_df.with_columns(
        cases_outbreak=pl.when(pl.col("imported").is_null())
        .then(pl.col("cases") - 1)  # if we don't know imports, assume 1 import
        .otherwise(pl.col("cases") - pl.col("imported")),
        # Since we'll need it, make a total pct exempt
        total_exempt_pct=pl.col("nonmed_exempt_pct")
        + pl.col("med_exempt_pct"),
    )

    return merged_df


if __name__ == "__main__":
    load_dotenv()

    # %% LOAD DATA
    census_key = os.getenv("CENSUS_API_KEY")

    outbreaks_df = read_outbreaks(data_dir)
    outbreaks_df = process_outbreaks_df(outbreaks_df)

    vaccinations_df = read_vaccinations(data_dir)
    vaccinations_df = process_vaccinations_df(vaccinations_df)
    # Merge and make ad hoc adjustments
    raw_merged_df = merge_outbreaks_vaccinations(outbreaks_df, vaccinations_df)
    raw_merged_df = add_hoc_merged_df_adjustments(raw_merged_df)
    # Subset data to require vaccinations
    merged_df = require_vaccine_data(raw_merged_df)
    # Assemble analysis dataset by estimating a "cases_outbreak" variable
    # that is the number of cases in the outbreak excluding importations
    merged_df = assemble_analysis_dataset(merged_df, params)

    # %% HOW DO THE EXCLUSION CRITERIA LIMIT THE DATA
    # Compare distribution of outbreak sizes with and without vaccination data
    fig, ax = compare_distributions(
        raw_merged_df["cases"],
        merged_df["cases"],
        xlabel="Cases",
        label1="Without vaccination data",
        label2="With vaccination data",
        log_scale=True,
        bins=[1, 2, 3, 4, 5, 10, 20, 50, 750],
    )
    plt.show()

    # Compare distribution of timing of outbreaks with and without vaccination data
    fig, ax = compare_distributions(
        raw_merged_df["year"],
        merged_df["year"],
        xlabel="Year",
        label1="Without vaccination data",
        label2="With vaccination data",
        log_scale=False,
    )
    plt.show()

    # %% VACCINATION COVERAGE AND OUTBREAK SIZE
    # add population variables to make it easy to consider population-dependent
    # variables in the exploration
    # add a population variable to plot incidence
    assert census_key is not None, (
        "A Census API key is required to run this analysis."
    )
    merged_df = add_pop_variables(merged_df, census_key)

    # MMR vaccination coverage
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["estimated_mmr_cov"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["estimated_mmr_cov"],
        xlabel="Estimated MMR vaccination coverage (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(75, 100, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # 5 year MMR vaccination coverage
    merged_df = merged_df.with_columns(
        estimated_mmr_cov_5yr=pl.col("estimated_mmr_cov")
        .rolling_mean(window_size=5, min_samples=1)
        .over("fips")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "estimated_mmr_cov_5yr"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "estimated_mmr_cov_5yr"
        ],
        xlabel="Estimated 5 yr MMR vaccination coverage (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(75, 100, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% NONMEDICAL VACCINATION EXEMPTIONS AND OUTBREAK SIZE
    # Nonmedical vaccination exemptions
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["nonmed_exempt_pct"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["nonmed_exempt_pct"],
        xlabel="Nonmedical vaccination exemptions (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0, 20, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # 5 year nonmedical vaccination exemptions
    merged_df = merged_df.with_columns(
        nonmed_exempt_pct_5yr=pl.col("nonmed_exempt_pct")
        .rolling_mean(window_size=5, min_samples=1)
        .over("fips")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "nonmed_exempt_pct_5yr"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "nonmed_exempt_pct_5yr"
        ],
        xlabel="Estimated 5 yr nonmedical vaccination exemptions (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0, 20, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% TOTAL VACCINATION EXEMPTIONS AND OUTBREAK SIZE
    # Total vaccination exemptions
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["total_exempt_pct"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["total_exempt_pct"],
        xlabel="Total vaccination exemptions (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0, 20, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # 5 year total vaccination exemptions
    merged_df = merged_df.with_columns(
        total_exempt_pct_5yr=pl.col("total_exempt_pct")
        .rolling_mean(window_size=5, min_samples=1)
        .over("fips")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "total_exempt_pct_5yr"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "total_exempt_pct_5yr"
        ],
        xlabel="Estimated 5 yr total vaccination exemptions (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0, 20, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% POPULATION SIZE AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["pop_5_9"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["pop_5_9"],
        xlabel="Population size (ages 5-9)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(100, 650000, 100),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% POPULATION DENSITY AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["pop_density_5_9"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["pop_density_5_9"],
        xlabel=r"Population density (ages 5-9 / km$^2$)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0.1, 1000, 0.1),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% FRACTION PUBLIC SCHOOL AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["fraction_public"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["fraction_public"],
        xlabel="Fraction of students in public schools",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0.45, 1.0, 0.001),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% FRACTION PRIVATE SCHOOL AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["fraction_private"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["fraction_private"],
        xlabel="Fraction of students in private schools",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0.0, 0.5, 0.001),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% FRACTION NOT ENROLLED AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "fraction_not_enrolled"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "fraction_not_enrolled"
        ],
        xlabel="Fraction of students not in public/private school",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0.0, 0.15, 0.001),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% INTERACTION BETWEEN VACCINATION COVERAGE AND PUBLIC SCHOOL
    merged_df = merged_df.with_columns(
        vaccine_coverage_school_interaction=pl.col("estimated_mmr_cov")
        * pl.col("fraction_public")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "vaccine_coverage_school_interaction"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "vaccine_coverage_school_interaction"
        ],
        xlabel="Vaccination coverage * fraction public school",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(45, 100, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # five year version
    merged_df = merged_df.with_columns(
        vaccine_coverage_school_interaction_5yr=pl.col(
            "vaccine_coverage_school_interaction"
        )
        .rolling_mean(window_size=5, min_samples=1)
        .over("fips")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "vaccine_coverage_school_interaction_5yr"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "vaccine_coverage_school_interaction_5yr"
        ],
        xlabel="5 yr average vaccination coverage * fraction public school",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(45, 100, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% INTERACTION BETWEEN VACCINATION EXEMPTIONS AND PUBLIC SCHOOL
    merged_df = merged_df.with_columns(
        exempt_school_interaction=pl.col("total_exempt_pct")
        * pl.col("fraction_private")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "exempt_school_interaction"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "exempt_school_interaction"
        ],
        xlabel="Vaccination exemptions * fraction private school",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        log_scale=False,
        bins=np.arange(0, 3, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% EXEMPTIONS VS COVERAGE
    fig, ax, scatter = pairplot(
        merged_df["estimated_mmr_cov"],
        100.0 - merged_df["total_exempt_pct"],
        xlabel="Estimated MMR vaccination coverage (%)",
        ylabel="Not vaccine exempt (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    ax.plot(np.arange(75, 101), np.arange(75, 101), c="black", ls="--", lw=2)
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # %% COVERAGE VS FRACTION PUBLIC SCHOOL
    fig, ax, scatter = pairplot(
        merged_df["estimated_mmr_cov"],
        merged_df["fraction_public"] * 100.0,
        xlabel="Estimated MMR vaccination coverage (%)",
        ylabel="Public school enrollment (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # %% EXEMPTIONS VS FRACTION PUBLIC SCHOOL
    fig, ax, scatter = pairplot(
        merged_df["total_exempt_pct"],
        merged_df["fraction_public"] * 100.0,
        xlabel="Total vaccination exemptions (%)",
        ylabel="Public school enrollment (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # %% COVERAGE*PUBLIC SCHOOL INTERACTION VS CASES
    fig, ax, scatter = pairplot(
        merged_df["vaccine_coverage_school_interaction"],
        merged_df["total_exempt_pct"],
        xlabel="Vaccination coverage * fraction public school",
        ylabel="Total vaccination exemptions (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # %% INCOME AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "median_household_income"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "median_household_income"
        ],
        xlabel="Median household income (2024 USD)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        bins=np.arange(40000, 185000, 100),
        density=True,
        cumulative=True,
    )
    plt.show()

    merged_df = merged_df.with_columns(
        median_household_income_5yr=pl.col("median_household_income")
        .rolling_mean(window_size=5, min_samples=1)
        .over("fips")
    )

    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "median_household_income_5yr"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "median_household_income_5yr"
        ],
        xlabel="5 yr median household income (2024 USD)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        bins=np.arange(40000, 185000, 100),
        density=True,
        cumulative=True,
    )
    plt.show()

    # income-exemptions correlation?
    fig, ax, scatter = pairplot(
        merged_df["median_household_income"],
        merged_df["nonmed_exempt_pct"],
        xlabel="Median household income (2024 USD)",
        ylabel="Nonmedical vaccination exemptions (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # income-exemptions interaction?
    merged_df = merged_df.with_columns(
        income_exempt_interaction=pl.col("median_household_income")
        * (100.0 - pl.col("nonmed_exempt_pct"))
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "income_exempt_interaction"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "income_exempt_interaction"
        ],
        xlabel="Income * Nonmedical exemptions interaction",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        bins=np.arange(4000000, 18000000, 10000),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% EDUCATION AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "fraction_graduate_degree"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "fraction_graduate_degree"
        ],
        xlabel="Fraction with graduate degree",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        bins=np.arange(0, 0.33, 0.001),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% EDUCATION-EXEMPTIONS AND OUTBREAK SIZE
    fig, ax, scatter = pairplot(
        merged_df["fraction_graduate_degree"],
        merged_df["nonmed_exempt_pct"],
        xlabel="Fraction with a graduate degree",
        ylabel="Nonmedical vaccination exemptions (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # %% RACE AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["fraction_white"],
        merged_df.filter(pl.col("cases_outbreak") >= 3)["fraction_white"],
        xlabel="Fraction white",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        bins=np.arange(0.18, 1.0, 0.001),
        density=True,
        cumulative=True,
    )
    plt.show()

    # %% RACE-EXEMPTIONS AND OUTBREAK SIZE
    fig, ax, scatter = pairplot(
        merged_df["fraction_white"],
        merged_df["nonmed_exempt_pct"],
        xlabel="Fraction white",
        ylabel="Nonmedical vaccination exemptions (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # %% MOBILITY AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "within_county_mobility_proxy"
        ],
        merged_df.filter(pl.col("cases_outbreak") >= 3)[
            "within_county_mobility_proxy"
        ],
        xlabel="Within-county mobility proxy",
        label1="No outbreak",
        label2="Outbreak (Cases >= 3)",
        bins=np.arange(0, 6.7, 0.01),
        density=True,
        cumulative=True,
    )
    plt.show()

    # Mobility, coverage, and cases
    fig, ax, scatter = pairplot(
        merged_df["within_county_mobility_proxy"],
        merged_df["estimated_mmr_cov"],
        xlabel="Within-county mobility proxy",
        ylabel="Estimated MMR vaccination coverage (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    ax.set_xscale("log")
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # Mobility * coverage, and cases
    fig, ax, scatter = pairplot(
        merged_df["within_county_mobility_proxy"]
        * merged_df["estimated_mmr_cov"],
        merged_df["total_exempt_pct"],
        xlabel="Within-county mobility proxy * coverage interaction",
        ylabel="Total vaccine exemptions (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    ax.set_xscale("log")
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # medical vs nonmedical exemptions
    fig, ax, scatter = pairplot(
        merged_df["med_exempt_pct"],
        merged_df["nonmed_exempt_pct"],
        xlabel="Medical vaccination exemptions (%)",
        ylabel="Nonmedical vaccination exemptions (%)",
        log=False,
        c=np.log10(merged_df["cases_outbreak"] + 1),
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=ax, label="Log10(Outbreak cases + 1)")
    plt.show()

    # histogram of county counts
    fig, ax = compare_distributions(
        merged_df.group_by("fips").agg(pl.len())["len"],
        pl.Series(dtype=pl.Int64),
        xlabel="Number of years with data",
        label1="",
        label2="",
        log_scale=False,
        bins=range(1, 11),
    )
    plt.show()

    # histogram of variances of county case counts
    fig, ax = compare_distributions(
        merged_df.group_by("fips").agg(pl.col("cases_outbreak").std())[
            "cases_outbreak"
        ],
        pl.Series(dtype=pl.Float64),
        xlabel="Standard deviation of outbreak case counts",
        label1="",
        label2="",
        log_scale=True,
        bins=np.arange(0, 225, 1),
    )
    plt.show()

    # %% EXPORT DATA FOR DOWNSTREAM ANALYSIS
    # save the cleaned dataset
    merged_df.write_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )
