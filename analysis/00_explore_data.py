import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from addfips import AddFIPS
from dotenv import load_dotenv
from matplotlib import rc
from tqdm.auto import tqdm

from outbreak_size import utils, vaccination_wrangling

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

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


def read_outbreaks(data_dir=data_dir) -> pl.DataFrame:
    df = pl.read_excel(
        os.path.join(data_dir, "Combined Measles Dataset QCed.xlsx")
    )

    # to ensure consistent lower case column names
    df = df.rename({col: col.lower() for col in df.columns})

    df_2025 = pl.read_csv(
        os.path.join(data_dir, "2025CasesImports QCed clean.csv"),
        schema_overrides={"fips": pl.String},
    )

    # set the fips for any county that is a higher level region to be the name
    # of the county
    # override the cases column with Lauren Case
    df_2025 = df_2025.with_columns(
        cases=pl.when(pl.col("Lauren Case").is_not_null())
        .then(pl.col("Lauren Case"))
        .otherwise(pl.col("cases"))
    )

    # drop anything south carolina because we have our own south carolina data
    df_2025 = df_2025.filter(pl.col("state") != "South Carolina")

    sc = pl.read_excel(os.path.join(data_dir, "SC2025Cases.xlsx"))
    sc = sc.rename({col: col.lower() for col in sc.columns})
    # add indigenous and imported as null
    sc = sc.with_columns(
        indigenous=pl.lit(None),
        imported=pl.lit(None),
        fips=pl.lit(None),
        year=pl.lit(2025),
        state=pl.lit("South Carolina"),
    )
    # concatenate to df_2025
    df_2025 = pl.concat([df_2025, sc], how="diagonal_relaxed")

    # concatenate and return
    df = pl.concat([df, df_2025], how="diagonal_relaxed")

    return df


def process_outbreaks_df(outbreaks_df: pl.DataFrame) -> pl.DataFrame:
    # the last row in outbreaks is a null row with the total number of cases over
    # the study period
    outbreaks_df = outbreaks_df.filter(pl.col("year").is_not_null())
    # let us drop outbreaks with no case count
    # because that doesn't tell us anything about what would have happened had
    # there been an importation
    outbreaks_df = outbreaks_df.filter(pl.col("cases") != 0)

    # when there are fips codes present, we need to do the usual cleaning
    outbreaks_df = outbreaks_df.with_columns(
        fips=pl.when(pl.col("fips").str.len_chars() > 2)
        .then(pl.col("fips").str.zfill(5))
        .otherwise(pl.col("fips"))
    )
    # for now, let's drop 2025 outbreaks with one or two digit fips codes because
    # in these cases the data are not at the state level but rather in an unknown
    # county
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2025)
            & (pl.col("fips").str.len_chars() < 5)
            & pl.col("fips").is_not_null()
        )
    )

    # last column seems to be urls of sources, drop it
    outbreaks_df = outbreaks_df.select(
        ["year", "state", "county", "cases", "indigenous", "imported", "fips"]
    )

    # Delaware reports vaccination coverage at the state only, so aggregate
    # the outbreak data up to that level as well
    outbreaks_df = outbreaks_df.with_columns(
        county=pl.when(pl.col("state") == "Delaware")
        .then(pl.lit("State"))
        .otherwise(pl.col("county"))
    )

    # Group by year, state, and county
    outbreaks_df = outbreaks_df.group_by(["year", "state", "county"]).agg(
        cases=pl.sum("cases"),
        indigenous=pl.sum("indigenous"),
        imported=pl.sum("imported"),
        fips=pl.first("fips"),
    )

    # District of Columbia cases don't have a county listed, so they're never going to merge
    # list the county as `District of Columbia` for merging purposes
    outbreaks_df = outbreaks_df.with_columns(
        county=pl.when(pl.col("state") == "District of Columbia")
        .then(pl.lit("District of Columbia"))
        .otherwise(pl.col("county"))
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

    # Detroit by itself doesn't have a fips code, so aggregate detroit up to Wayne Co
    outbreaks_df = (
        outbreaks_df.with_columns(
            [pl.col("county").str.replace_all("Detroit", "Wayne")]
        )
        .group_by(["year", "state", "county"])
        .agg(
            cases=pl.when(pl.col("cases").is_null().all())
            .then(None)
            .otherwise(pl.col("cases").sum()),
            indigenous=pl.when(pl.col("indigenous").is_null().all())
            .then(None)
            .otherwise(pl.col("indigenous").sum()),
            imported=pl.when(pl.col("imported").is_null().all())
            .then(None)
            .otherwise(pl.col("imported").sum()),
            fips=pl.first("fips"),
        )
    )

    # Drop Chicago 2024 -- migrant shelter outbreak
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2024)
            & (pl.col("state") == "Illinois")
            & (pl.col("county") == "Cook")
        )
    )
    # Drop Wisconsin 2021 -- Operation Allies Welcome Afghan refugee outbreak
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2021)
            & (pl.col("state") == "Wisconsin")
            & (pl.col("county") == "Monroe")
        )
    )
    # Drop ICE detention facility 2016
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2016)
            & (pl.col("state") == "Arizona")
            & (pl.col("county") == "Pinal")
        )
    )

    return outbreaks_df


def higher_level_regions() -> dict[str, list[str]]:
    assignment_special_places = {
        "Fulton and Gwynett": ["13121", "13135"],
        "Gulf Coast": ["02150", "02122", "02063", "02066"],
        "East tennessee?": [
            "47001",
            "47007",
            "47009",
            "47011",
            "47013",
            "47019",
            "47025",
            "47029",
            "47035",
            "47057",
            "47059",
            "47063",
            "47065",
            "47067",
            "47073",
            "47089",
            "47091",
            "47093",
            "47105",
            "47107",
            "47121",
            "47123",
            "47129",
            "47139",
            "47143",
            "47145",
            "47151",
            "47155",
            "47163",
            "47171",
            "47173",
            "47179",
        ],
        "Southwest Health District, Utah": [
            "49001",
            "49017",
            "49021",
            "49025",
            "49053",
        ],
        "Mid-Cumberland Region, Tennessee": [
            "47043",
            "47125",
            "47147",
            "47161",
            "47187",
            "47189",
        ],
        "Bear River, Utah": ["49003", "49005", "49033"],
        "Southeast Health District, Utah": ["49007", "49015", "49019"],
        "Upper Cumberland Region, Tennessee": [
            "47015",
            "47027",
            "47041",
            "47049",
            "47061",
            "47087",
            "47111",
            "47133",
            "47137",
            "47141",
            "47159",
            "47175",
            "47177",
            "47185",
        ],
        "Nashville-Davidson County Region, Tennessee": ["47037"],
        "Twin Cities Metro Area, Minnesota": [
            "27003",
            "27025",
            "27053",
            "27079",
            "27139",
            "27171",
            "27019",
            "27037",
            "27059",
            "27095",
            "27123",
            "27141",
            "27163",
        ],
        "Region 9, Louisiana": ["22063", "22091", "22103", "22105", "22107"],
        "Northwest Region, Virginia": [
            "51015",
            "51017",
            "51091",
            "51163",
            "51165",
            "51530",
            "51660",
            "51790",
            "51820",
            "51043",
            "51069",
            "51139",
            "51171",
            "51187",
            "51840",
            "51033",
            "51099",
            "51177",
            "51179",
            "51630",
            "51047",
            "51061",
            "51113",
            "51137",
            "51157",
            "51003",
            "51065",
            "51079",
            "51109",
            "51125",
            "51540",
        ],
        "Eastern Region, Virginia": [
            "51550",
            "51001",
            "51131",
            "51650",
            "51710",
            "51095",
            "51199",
            "51700",
            "51735",
            "51830",
            "51740",
            "51057",
            "51073",
            "51097",
            "51101",
            "51103",
            "51115",
            "51119",
            "51133",
            "51159",
            "51193",
            "51810",
            "51093",
            "51175",
            "51620",
            "51800",
        ],
    }
    return assignment_special_places


def exclude_county_state_year(df, county, state, year):
    return df.filter(
        ~(
            (pl.col("county") == county)
            & (pl.col("state") == state)
            & (pl.col("year") == year)
        )
    )


def read_vaccinations(data_dir=data_dir) -> pl.DataFrame:
    df = pl.read_excel(
        os.path.join(
            data_dir,
            "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied_AK_CA_Updated.xlsx",
        )
    )

    # remove the funky places lauren found in her QC
    df = exclude_county_state_year(df, "Butts", "Georgia", 2024)
    df = exclude_county_state_year(df, "Jasper", "Georgia", 2024)
    df = exclude_county_state_year(df, "Lumpkin", "Georgia", 2024)
    df = exclude_county_state_year(df, "Richmond", "Georgia", 2024)
    df = exclude_county_state_year(df, "Schley", "Georgia", 2024)
    df = exclude_county_state_year(df, "Baltimore City", "Maryland", 2019)
    df = exclude_county_state_year(df, "Slope", "North Dakota", 2021)
    df = df.with_columns(
        MMR_cov=pl.when(
            (pl.col("county") == "Cocke")
            & (pl.col("state") == "Tennessee")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(94.5))
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
    # add in missing fips
    addfips = AddFIPS()
    df = df.with_columns(
        fips=pl.when(pl.col("state") != "Delaware")
        .then(
            pl.struct(["fips", "county", "state"]).map_elements(
                lambda row: addfips.get_county_fips(
                    row["county"], row["state"]
                ),
                return_dtype=pl.String,
            ),
        )
        .otherwise(pl.col("fips"))
    )

    # add in kansas data
    kansas_vacc = vaccination_wrangling.read_kansas_data(data_dir=data_dir)
    df = pl.concat([df, kansas_vacc], how="diagonal_relaxed")

    return df


def query_avg_vaccinations_region_years(
    vaccinations_df: pl.DataFrame, years: list[int], fips: list[str]
) -> dict:
    # get the vaccination data for the region and years
    temp_df = vaccinations_df.filter(
        (pl.col("year").is_in(years)) & (pl.col("fips").is_in(fips))
    )
    if temp_df.height == 0:
        return {
            "estimated_mmr_cov": None,
            "nonmed_exempt_pct": None,
            "med_exempt_pct": None,
        }
    # get population to weight the average values
    census_query = utils.CensusQuery(os.getenv("CENSUS_API_KEY"))  # type: ignore
    pop_df = census_query.query_5_9_pop(
        temp_df["year"].to_list(), temp_df["fips"].to_list()
    )
    temp_df = temp_df.with_columns(pop_5=pop_df["pop_5_9"] / 5)
    temp_df = temp_df.with_columns(
        weight=pl.col("pop_5") / pl.sum("pop_5").over("year")
    )
    temp_df = temp_df.group_by("year").agg(
        estimated_mmr_cov=(
            pl.col("estimated_mmr_cov") * pl.col("weight")
        ).sum(),
        nonmed_exempt_pct=(
            pl.col("nonmed_exempt_pct") * pl.col("weight")
        ).sum(),
        med_exempt_pct=(pl.col("med_exempt_pct") * pl.col("weight")).sum(),
        pop_5=pl.col("pop_5").sum(),
    )
    temp_df = temp_df.with_columns(weight=pl.col("pop_5") / pl.sum("pop_5"))
    temp_df = temp_df.with_columns(
        estimated_mmr_cov=pl.col("estimated_mmr_cov") * pl.col("weight"),
        nonmed_exempt_pct=pl.col("nonmed_exempt_pct") * pl.col("weight"),
        med_exempt_pct=pl.col("med_exempt_pct") * pl.col("weight"),
    )
    temp_df = temp_df.select(
        estimated_mmr_cov=pl.col("estimated_mmr_cov").sum(),
        nonmed_exempt_pct=pl.col("nonmed_exempt_pct").sum(),
        med_exempt_pct=pl.col("med_exempt_pct").sum(),
    )

    return temp_df.to_dict(as_series=False)


def process_vaccinations_df(vaccinations_df: pl.DataFrame) -> pl.DataFrame:
    # when we don't have mmr coverage, use all coverage
    vaccinations_df = vaccinations_df.with_columns(
        estimated_mmr_cov=pl.when(
            pl.col("mmr_cov").is_null() | (pl.col("mmr_cov") == 0.0)
        )
        .then(pl.col("all_vacc_cov"))
        .otherwise(pl.col("mmr_cov"))
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

    # edit the washington dc fips code to be 11001
    vaccinations_df = vaccinations_df.with_columns(
        fips=pl.when(pl.col("fips") == "00011")
        .then(pl.lit("11001"))
        .otherwise(pl.col("fips"))
    )

    nebraska_fips = {
        "old_nebraska": [
            *["310012"] * 3,
            "310021",
            "310002",
            *["310005"] * 4,
            *["310008"] * 4,
            *["310007"] * 4,
            "310003",
            *["310015"] * 9,
            *["310020"] * 9,
            *["310010"] * 4,
            *["310018"] * 12,
            *["310011"] * 5,
            *["310006"] * 2,
            *["310013"] * 4,
            *["310009"] * 5,
            *["310017"] * 9,
            *["310004"] * 3,
            *["310014"] * 7,
            *["310016"] * 6,
        ],
        "new_nebraska": [
            "31079",
            "31081",
            "31121",
            "31043",
            "31055",
            "31011",
            "31037",
            "31125",
            "31141",
            "31021",
            "31039",
            "31119",
            "31167",
            "31023",
            "31143",
            "31159",
            "31185",
            "31109",
            "31009",
            "31041",
            "31071",
            "31077",
            "31093",
            "31115",
            "31163",
            "31175",
            "31183",
            "31003",
            "31015",
            "31017",
            "31031",
            "31089",
            "31103",
            "31107",
            "31139",
            "31149",
            "31027",
            "31051",
            "31173",
            "31179",
            "31007",
            "31013",
            "31033",
            "31045",
            "31049",
            "31069",
            "31075",
            "31165",
            "31157",
            "31123",
            "31161",
            "31105",
            "31059",
            "31067",
            "31095",
            "31151",
            "31169",
            "31153",
            "31025",
            "31001",
            "31035",
            "31129",
            "31181",
            "31097",
            "31127",
            "31131",
            "31133",
            "31147",
            "31029",
            "31057",
            "31063",
            "31065",
            "31085",
            "31087",
            "31101",
            "31135",
            "31145",
            "31053",
            "31177",
            "31155",
            "31019",
            "31047",
            "31061",
            "31073",
            "31083",
            "31099",
            "31137",
            "31005",
            "31091",
            "31111",
            "31113",
            "31117",
            "31171",
        ],
    }

    nebraska_fips = pl.DataFrame(nebraska_fips)
    vaccinations_df = vaccinations_df.join(
        nebraska_fips, left_on="fips", right_on="old_nebraska", how="left"
    )
    vaccinations_df = vaccinations_df.with_columns(
        fips=pl.when(pl.col("fips") == "11")
        .then(pl.lit("11001"))
        .otherwise(pl.col("fips"))
    )

    assignment_special_places = higher_level_regions()

    for place, fips_list in assignment_special_places.items():
        if place == "Gulf Coast":
            continue

        state = vaccinations_df.filter(pl.col("fips") == fips_list[0])[
            "state"
        ][0]

        for year in vaccinations_df["year"].unique().to_list():
            temp = query_avg_vaccinations_region_years(
                vaccinations_df, years=[year], fips=fips_list
            )

            temp_df = pl.DataFrame(temp)
            temp_df = temp_df.with_columns(
                year=pl.lit(year),
                state=pl.lit(state),
                county=pl.lit(place),
                fips=pl.lit(None),
            )

            vaccinations_df = pl.concat(
                [vaccinations_df, temp_df], how="diagonal_relaxed"
            )

    vaccinations_df = vaccinations_df.with_columns(
        fips=pl.when(
            (pl.col("fips").str.len_chars() > 5)
            & (pl.col("state") != "Kansas")
        )
        .then(pl.col("fips").str.slice(0, 5))
        .otherwise(pl.col("fips"))
    )

    return vaccinations_df


def merge_outbreaks_vaccinations(
    outbreaks_df: pl.DataFrame, vaccinations_df: pl.DataFrame
) -> pl.DataFrame:
    outbreaks_df = outbreaks_df.with_columns(
        year_relevant_vaccination=pl.col("year") - 1
    )
    print("Outbreaks df shape before merge:", outbreaks_df.shape)
    merged_df = outbreaks_df.join(
        vaccinations_df,
        left_on=["year_relevant_vaccination", "state", "county"],
        right_on=["year", "state", "county"],
        how="left",
    )
    merged_df = merged_df.with_columns(
        fips=pl.when(pl.col("fips_right").is_not_null())
        .then(pl.col("fips_right"))
        .otherwise(pl.col("fips"))
    )
    merged_df = merged_df.drop(["fips_right"])
    print("Merged df shape after merge:", merged_df.shape)
    return merged_df


def add_pop_variables(
    merged_df: pl.DataFrame, census_key: str
) -> pl.DataFrame:
    census_query = utils.CensusQuery(census_key)
    fips = merged_df["fips"].to_list()
    years_census = merged_df["year"].to_list()
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
    land_df = census_query.query_land_size(fips)
    merged_df = merged_df.join(land_df, on="fips", how="left")

    merged_df = merged_df.with_columns(
        pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
        fraction_public=(pl.col("public_schooled") / pl.col("pop_5_9")),
        fraction_private=(pl.col("private_schooled") / pl.col("pop_5_9")),
        fraction_graduate_degree=(
            pl.col("number_graduate_degree") / pl.col("pop_25_up")
        ),
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
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)

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
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)

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
    include = pl.col("cases") == 1
    include = include | (pl.col("imported") == pl.col("cases"))
    include = include | (pl.col("imported") < pl.col("cases"))
    include = include | (pl.col("cases") >= params["inclusion_threshold"])
    merged_df = merged_df.filter(include)

    merged_df = merged_df.with_columns(
        cases_outbreak=pl.when(
            pl.col("imported").is_null() | pl.col("imported") == 0
        )
        .then(pl.col("cases"))
        .otherwise(pl.col("cases") - pl.col("imported")),
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

    # ------------------------------------------------------------------
    # Write processed_vaccinations_data.csv using a state-imputed copy.
    # The original vaccinations_df is used unchanged for all downstream
    # outbreak merging and analysis.
    # ------------------------------------------------------------------
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
    vaccinations_df_imputed.write_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv")
    )

    # Merge using the original (non-imputed) vaccinations_df
    raw_merged_df = merge_outbreaks_vaccinations(outbreaks_df, vaccinations_df)
    raw_merged_df.write_csv(
        os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv")
    )
    merged_df = require_vaccine_data(raw_merged_df)
    merged_df = assemble_analysis_dataset(merged_df, params)

    # %% HOW DO THE EXCLUSION CRITERIA LIMIT THE DATA
    fig, ax = compare_distributions(
        raw_merged_df["cases"],
        merged_df["cases"],
        xlabel="Cases",
        label1="Without vaccination data",
        label2="With vaccination data",
        log_scale=True,
        bins=[1, 2, 3, 4, 5, 10, 20, 50, 750],
    )

    fig, ax = compare_distributions(
        raw_merged_df["year"],
        merged_df["year"],
        xlabel="Year",
        label1="Without vaccination data",
        label2="With vaccination data",
        log_scale=False,
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
    for place, fips_list in special_places_dict.items():
        temp_df = special.filter(pl.col("county") == place)
        if temp_df.height == 0:
            continue
        temp_df = pl.concat(
            [temp_df] * len(fips_list), how="diagonal_relaxed"
        ).with_columns(fips=pl.Series(fips_list))
        temp_df = add_pop_variables(temp_df, census_key)
        temp_df = temp_df.with_columns(
            weight_15_up=pl.col("pop_15_up")
            / pl.sum("pop_15_up").over("year"),
            weight_25_up=pl.col("pop_25_up")
            / pl.sum("pop_25_up").over("year"),
            weight_overall=pl.col("pop") / pl.sum("pop").over("year"),
        )
        temp_df = temp_df.group_by(["year"]).agg(
            state=pl.first("state"),
            county=pl.lit(place),
            cases=pl.first("cases"),
            cases_outbreak=pl.first("cases_outbreak"),
            imported=pl.first("imported"),
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
            fraction_public=(
                pl.col("public_schooled") / pl.col("pop_under_5")
            ),
            fraction_private=(
                pl.col("private_schooled") / pl.col("pop_under_5")
            ),
            fraction_graduate_degree=(
                pl.col("number_graduate_degree") / pl.col("pop_25_up")
            ),
            pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
            fips=pl.lit(None),
        )
        merged_df = pl.concat([merged_df, temp_df], how="diagonal_relaxed")

    merged_df = merged_df.with_columns(
        total_exempt_pct=pl.col("nonmed_exempt_pct")
        + pl.col("med_exempt_pct"),
        fraction_nonmed_exempt=pl.col("nonmed_exempt_pct")
        / pl.col("total_exempt_pct").fill_nan(0.0),
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
        merged_df = merged_df.with_columns(
            estimated_mmr_cov_5yr=pl.lit(None),
            nonmed_exempt_pct_5yr=pl.lit(None),
            med_exempt_pct_5yr=pl.lit(None),
        )
        # iterate through the df
        new_df = []
        for row in merged_df.iter_rows(named=True):
            if row["fips"] is None:
                # this is one of our special places
                assert row["county"] in special_places_dict.keys()
            vaccinations_5_yrs = query_avg_vaccinations_region_years(
                vaccinations_df,
                years=list(range(row["year"] - 5, row["year"])),
                fips=[row["fips"]]
                if row["fips"] is not None
                else special_places_dict[row["county"]],
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

            # append to new df
            new_df.append(row)

        merged_df = pl.concat(new_df, how="diagonal_relaxed")
        # make total exemption columns
        merged_df = merged_df.with_columns(
            total_exempt_pct_5yr=pl.col("nonmed_exempt_pct_5yr")
            + pl.col("med_exempt_pct_5yr"),
        ).with_columns(
            # fractions of exemptions
            fraction_nonmed_exempt_5yr=pl.col("nonmed_exempt_pct_5yr")
            / pl.col("total_exempt_pct_5yr").fill_nan(0.0),
        )

    # (all remaining comparison/pairplot/synthetic-population code unchanged)
    # %% EXPORT DATA FOR DOWNSTREAM ANALYSIS
    merged_df.write_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )
