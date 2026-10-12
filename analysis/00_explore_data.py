import argparse
import os
import re

import polars as pl
import yaml
from addfips import AddFIPS
from dotenv import load_dotenv
from tqdm.auto import tqdm

from outbreak_size import synth_pop_clusters, utils, vaccination_wrangling

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

    # correct a year = 2925 to a 2025
    df_2025 = df_2025.with_columns(
        year=pl.when(pl.col("year") == 2925)
        .then(2025)
        .otherwise(pl.col("year"))
    )

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
    # same with Central Utah
    df_2025 = df_2025.filter(
        ~((pl.col("county") == "Central") & (pl.col("state") == "Utah"))
    )

    # set the fips for any county that is a higher level region to be the name
    # of the county
    # override the cases column with Lauren Case
    df_2025 = df_2025.filter(pl.col("Lauren Case").is_not_null()).with_columns(
        cases=pl.col("Lauren Case")
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
    # remove anything pre 2011 because we declare study period as 2011-2025
    outbreaks_df = outbreaks_df.filter(pl.col("year") >= 2011)
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
        # "Fulton, Gwinett": ["13121", "13135"],
        # "Forsyth, Fulton, Paulding": ['13117', '13121', '13223'],
        # "Blue Ridge": ['51023', '51067', '51161'],
        # 'District 1': ['22051', '22071', '21075', '21087'],
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
        "North, Alabama": [
            "01019",
            "01033",
            "01043",
            "01049",
            "01071",
            "01077",
            "01079",
            "01083",
            "01089",
            "01095",
            "01103",
            "01133",
        ],
        "Region 1, Louisiana": ["22051", "22071", "22075", "22087"],
        # https://www.vdh.virginia.gov/content/uploads/2020/05/Locality-to-HD-to-HPR.pdf
        "Piedmont, Virginia": [
            "51007",
            "51029",
            "51037",
            "51049",
            "51111",
            "51135",
            "51147",
        ],
        # 'Central Region, Virginia': [
        #     '51007', '51025', '51029', '51036', '51037', '51041', '51570', '51049', '51053', '51595', '51075', '51081', '51083', '51650', '51085', '51087', '51670', '51111', '51117', '51127', '51135', '51730', '51145', '51147', '51149', '51760', '51181', '51183'
        # ],
        # "Northwest Region, Virginia": [
        #     "51015",
        #     "51017",
        #     "51091",
        #     "51163",
        #     "51165",
        #     "51530",
        #     "51660",
        #     "51790",
        #     "51820",
        #     "51043",
        #     "51069",
        #     "51139",
        #     "51171",
        #     "51187",
        #     "51840",
        #     "51033",
        #     "51099",
        #     "51177",
        #     "51179",
        #     "51630",
        #     "51047",
        #     "51061",
        #     "51113",
        #     "51137",
        #     "51157",
        #     "51003",
        #     "51065",
        #     "51079",
        #     "51109",
        #     "51125",
        #     "51540",
        # ],
        # "Eastern Region, Virginia": [
        #     "51550",
        #     "51001",
        #     "51131",
        #     "51650",
        #     "51710",
        #     "51095",
        #     "51199",
        #     "51700",
        #     "51735",
        #     "51830",
        #     "51740",
        #     "51057",
        #     "51073",
        #     "51097",
        #     "51101",
        #     "51103",
        #     "51115",
        #     "51119",
        #     "51133",
        #     "51159",
        #     "51193",
        #     "51810",
        #     "51093",
        #     "51175",
        #     "51620",
        #     "51800",
        # ],
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

    # "New York City" cases were assigned to Kings above, but the 2025 row
    # still gets the case sheet's own NYC fips (Manhattan, 36061) because
    # there's no 2024 NY vaccine data for any county to join against; fix it
    # to Kings' own fips so it isn't attributed to the wrong county
    merged_df = merged_df.with_columns(
        fips=pl.when(
            (pl.col("state") == "New York") & (pl.col("county") == "Kings")
        )
        .then(pl.lit("36047"))
        .otherwise(pl.col("fips"))
    )

    print("Merged df shape after merge:", merged_df.shape)
    return merged_df


def add_pop_variables(
    merged_df: pl.DataFrame, census_key: str
) -> pl.DataFrame:
    census_query = utils.CensusQuery(census_key)
    fips = merged_df["fips"].to_list()
    fips_unique = merged_df["fips"].unique(maintain_order=True).to_list()
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
    land_df = census_query.query_land_size(fips_unique)
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


def require_vaccine_data(
    merged_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    merged_df_return = merged_df.filter(
        pl.col("estimated_mmr_cov").is_not_null()
        & pl.col("nonmed_exempt_pct").is_not_null()
        & pl.col("med_exempt_pct").is_not_null()
    )
    # also get out the opposite
    merged_df_no_vaccine_data = merged_df.filter(
        pl.col("estimated_mmr_cov").is_null()
        | pl.col("nonmed_exempt_pct").is_null()
        | pl.col("med_exempt_pct").is_null()
    )
    return merged_df_return, merged_df_no_vaccine_data


def assemble_analysis_dataset(
    merged_df: pl.DataFrame, params: dict = params
) -> pl.DataFrame:
    # include = pl.col("cases") == 1
    # include = include | (pl.col("imported") == pl.col("cases"))
    # include = include | (pl.col("imported") < pl.col("cases"))
    # include = include | (pl.col("cases") >= params["inclusion_threshold"])
    # merged_df = merged_df.filter(include)

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


def impute_from_bordering_years(
    missing_df: pl.DataFrame,
    vaccinations_df: pl.DataFrame,
    state_level_file: str,
) -> pl.DataFrame:
    """
    Impute missing vaccination data with the following priority:
    1. Same county, outbreak year (year+0)
    2. Same county, outbreak year-2
    3. State-level CDC data for outbreak year-1

    Only imputes for regular counties, not special regions.
    """
    # Load and process state-level data
    state_data = pl.read_csv(state_level_file)

    # Convert Estimate (%) to numeric, treating NR/NA as null
    state_data = state_data.with_columns(
        estimate_pct=pl.col("Estimate (%)").cast(pl.Float64, strict=False)
    )

    # Filter out null estimates
    state_data = state_data.filter(pl.col("estimate_pct").is_not_null())

    # Filter to state-level data only
    state_data = state_data.filter(pl.col("Geography Type") == "States")

    # Parse the school year to get the year
    # School Year format is like "2010-11" which represents fall 2010 data
    state_data = state_data.with_columns(
        year=pl.col("School Year").str.extract(r"^(\d{4})", 1).cast(pl.Int64)
    )

    # Create separate dataframes for coverage and exemptions
    # MMR coverage - filter to MMR rows
    state_mmr_cov = state_data.filter(
        pl.col("Vaccine/Exemption") == "MMR"
    ).select(
        [
            pl.col("Geography").alias("state"),
            "year",
            pl.col("estimate_pct").alias("mmr_cov_state"),
        ]
    )

    # Medical exemptions - these are NOT vaccine-specific, so don't filter to MMR
    state_med_exempt = state_data.filter(
        pl.col("Dose").str.starts_with("Medical")
    ).select(
        [
            pl.col("Geography").alias("state"),
            "year",
            pl.col("estimate_pct").alias("med_exempt_pct_state"),
        ]
    )

    # Nonmedical exemptions - these are NOT vaccine-specific, so don't filter to MMR
    # Prefer explicit nonmedical, fall back to "any"
    state_nonmed_exempt = state_data.filter(
        pl.col("Dose").str.contains("Nonmedical|Any")
    ).select(
        [
            pl.col("Geography").alias("state"),
            "year",
            pl.col("estimate_pct").alias("nonmed_exempt_pct_state"),
            pl.col("Vaccine/Exemption").alias("exemption_type"),
        ]
    )

    # For each state-year, prefer nonmedical over any
    state_nonmed_exempt = state_nonmed_exempt.sort(
        "exemption_type", descending=True
    )
    state_nonmed_exempt = state_nonmed_exempt.group_by(
        ["state", "year"]
    ).first()
    state_nonmed_exempt = state_nonmed_exempt.drop("exemption_type")

    # Merge all state-level data using full outer joins
    state_level_data = state_mmr_cov

    if state_med_exempt.height > 0:
        state_level_data = state_level_data.join(
            state_med_exempt, on=["state", "year"], how="full", coalesce=True
        )

    if state_nonmed_exempt.height > 0:
        state_level_data = state_level_data.join(
            state_nonmed_exempt,
            on=["state", "year"],
            how="full",
            coalesce=True,
        )

    # Filter out special regions
    special_places = list(higher_level_regions().keys())
    missing_regular = missing_df.filter(
        ~pl.col("county").is_in(special_places)
    )

    imputed_rows = []

    for row in missing_regular.iter_rows(named=True):
        outbreak_year = row["year"]  # The year the outbreak occurred
        year_relevant_vaccination = row[
            "year_relevant_vaccination"
        ]  # outbreak_year - 1
        state = row["state"]
        county = row["county"]

        # Track what we imputed and from where
        imputation_source_mmr = None
        imputation_source_nonmed = None
        imputation_source_med = None

        # Determine which fields need imputation
        needs_mmr = row["estimated_mmr_cov"] is None
        needs_nonmed = row["nonmed_exempt_pct"] is None
        needs_med = row["med_exempt_pct"] is None

        if not (needs_mmr or needs_nonmed or needs_med):
            continue  # Nothing to impute

        # Try outbreak year (year+0), then outbreak year-2
        # outbreak_year = vaccination year we're looking for
        # outbreak_year - 2 = go back 2 years from outbreak
        for offset, label in [(0, "year+0"), (-2, "year-2")]:
            if not (needs_mmr or needs_nonmed or needs_med):
                break  # All fields filled

            lookup_year = outbreak_year + offset

            bordering_data = vaccinations_df.filter(
                (pl.col("year") == lookup_year)
                & (pl.col("state") == state)
                & (pl.col("county") == county)
            )

            if bordering_data.height > 0:
                bordering_row = bordering_data.row(0, named=True)

                if (
                    needs_mmr
                    and bordering_row["estimated_mmr_cov"] is not None
                ):
                    row["estimated_mmr_cov"] = bordering_row[
                        "estimated_mmr_cov"
                    ]
                    imputation_source_mmr = label
                    needs_mmr = False

                if (
                    needs_nonmed
                    and bordering_row["nonmed_exempt_pct"] is not None
                ):
                    row["nonmed_exempt_pct"] = bordering_row[
                        "nonmed_exempt_pct"
                    ]
                    imputation_source_nonmed = label
                    needs_nonmed = False

                if needs_med and bordering_row["med_exempt_pct"] is not None:
                    row["med_exempt_pct"] = bordering_row["med_exempt_pct"]
                    imputation_source_med = label
                    needs_med = False

        # If still missing data, try state-level data for outbreak_year - 1
        if needs_mmr or needs_nonmed or needs_med:
            state_year_data = state_level_data.filter(
                (pl.col("state") == state)
                & (pl.col("year") == year_relevant_vaccination)
            )

            if state_year_data.height > 0:
                state_row = state_year_data.row(0, named=True)

                if needs_mmr and state_row.get("mmr_cov_state") is not None:
                    row["estimated_mmr_cov"] = state_row["mmr_cov_state"]
                    imputation_source_mmr = "state_cdc"
                    needs_mmr = False

                if (
                    needs_nonmed
                    and state_row.get("nonmed_exempt_pct_state") is not None
                ):
                    row["nonmed_exempt_pct"] = state_row[
                        "nonmed_exempt_pct_state"
                    ]
                    imputation_source_nonmed = "state_cdc"
                    needs_nonmed = False

                    # If we used "Any" exemption for nonmedical and don't have medical
                    # exemption data from state, impute medical as 0
                    if (
                        needs_med
                        and state_row.get("med_exempt_pct_state") is None
                    ):
                        row["med_exempt_pct"] = 0.0
                        imputation_source_med = "state_cdc_zero"
                        needs_med = False

                if (
                    needs_med
                    and state_row.get("med_exempt_pct_state") is not None
                ):
                    row["med_exempt_pct"] = state_row["med_exempt_pct_state"]
                    imputation_source_med = "state_cdc"
                    needs_med = False

        # Only add row if we successfully imputed at least one field
        if (
            imputation_source_mmr
            or imputation_source_nonmed
            or imputation_source_med
        ):
            row_df = pl.DataFrame([row])

            # Create imputation source column
            imputation_sources = []
            if imputation_source_mmr:
                imputation_sources.append(f"mmr:{imputation_source_mmr}")
            if imputation_source_nonmed:
                imputation_sources.append(f"nonmed:{imputation_source_nonmed}")
            if imputation_source_med:
                imputation_sources.append(f"med:{imputation_source_med}")

            row_df = row_df.with_columns(
                imputation_source=pl.lit("; ".join(imputation_sources))
            )

            imputed_rows.append(row_df)

    if len(imputed_rows) > 0:
        imputed_df = pl.concat(imputed_rows, how="diagonal_relaxed")

        # Print statistics
        total_imputed = imputed_df.shape[0]
        from_year_0 = imputed_df.filter(
            pl.col("imputation_source").str.contains("year\\+0")
        ).shape[0]
        from_year_minus_2 = imputed_df.filter(
            pl.col("imputation_source").str.contains("year-2")
        ).shape[0]
        from_state = imputed_df.filter(
            pl.col("imputation_source").str.contains("state_cdc")
        ).shape[0]

        print(f"Successfully imputed {total_imputed} county-years:")
        print(f"  - From outbreak year (year+0): {from_year_0}")
        print(f"  - From outbreak year-2: {from_year_minus_2}")
        print(f"  - From state-level CDC data: {from_state}")

        return imputed_df
    else:
        print("No rows could be imputed")
        return pl.DataFrame()


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
        f"exemptions): {100 * grid['all_three'].mean():.1f}% "
        f"({grid['all_three'].sum()})"
    )
    print(
        f"  MMR/all-vaccine coverage alone: {100 * grid['mmr'].mean():.1f}% "
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
        missing.group_by("pattern")
        .agg(
            rows=pl.len(),
            completed_by_imputation=pl.col("complete").fill_null(False).sum(),
        )
        .with_columns(
            pct_of_missing=(100 * pl.col("rows") / n_missing).round(1),
            pct_of_all=(100 * pl.col("rows") / n_all).round(1),
        )
        .sort("rows", descending=True)
    )

    print("\nB. what imputation does with them:")
    print(
        missing.group_by("outcome")
        .agg(rows=pl.len())
        .with_columns(
            pct_of_missing=(100 * pl.col("rows") / n_missing).round(1),
            pct_of_all=(100 * pl.col("rows") / n_all).round(1),
        )
        .sort("rows", descending=True)
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
        missing.group_by("pattern", "outcome")
        .agg(rows=pl.len())
        .pivot(on="outcome", index="pattern", values="rows")
        .fill_null(0)
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

    vaccinations_df = read_vaccinations(data_dir)
    vaccinations_df = process_vaccinations_df(vaccinations_df)
    print_vaccine_data_availability(vaccinations_df, data_dir)

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
    merged_df, merged_df_no_vaccine_data = require_vaccine_data(raw_merged_df)
    merged_df = assemble_analysis_dataset(merged_df, params)
    merged_df_no_vaccine_data = assemble_analysis_dataset(
        merged_df_no_vaccine_data, params
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
    merged_df_no_vaccine_data.write_csv(
        os.path.join(data_dir, "missing_counties.csv")
    )

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
    # we have to remove special places that ar eimputed
    # remove fulton, gwinett
    merged_df = merged_df.filter(
        ~(
            (pl.col("county") == "Fulton, Gwinett")
            & (pl.col("state") == "Georgia")
        )
    )
    # remove forsyth, fulton, paulding
    merged_df = merged_df.filter(
        ~(
            (pl.col("county") == "Forsyth, Fulton, Paulding")
            & (pl.col("state") == "Georgia")
        )
    )
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
            / pl.col("total_exempt_pct_5yr").fill_nan(0.0),
        )

    # %% SYNTHETIC POPULATION CLUSTERS
    # let's make a geoid for easy merging later
    merged_df = merged_df.with_columns(
        geoid=pl.when(pl.col("fips").is_null())
        .then(pl.col("county"))
        .otherwise(pl.col("county") + pl.col("fips"))
    )
    # grab the counties and fips codes
    population_pockets = merged_df[["geoid", "county", "fips"]].unique()
    population_pockets = population_pockets.with_columns(
        population_pockets_proxy=pl.lit(None),
        avg_house_size=pl.lit(None),
        std_house_size=pl.lit(None),
        # make the fips code a list for the special places
        fips=pl.struct("county", "fips").map_elements(
            lambda cols: [cols["fips"]]
            if cols["fips"] is not None
            else special_places_dict[cols["county"]],
            return_dtype=pl.List(pl.Utf8),
        ),
    )

    population_pockets_proxy = []
    avg_house_size = []
    std_house_size = []
    for fips in tqdm(population_pockets["fips"]):
        population_pockets_proxy.append(
            synth_pop_clusters.cluster_fips(fips)[0]
        )
        avg_house_size_fips, std_house_size_fips = (
            synth_pop_clusters.summarize_household_size(fips)
        )
        avg_house_size.append(avg_house_size_fips)
        std_house_size.append(std_house_size_fips)
    # merge back to main df
    population_pockets = population_pockets.with_columns(
        population_pockets_proxy=pl.Series(population_pockets_proxy),
        avg_house_size=pl.Series(avg_house_size),
        std_house_size=pl.Series(std_house_size),
    )
    merged_df = merged_df.join(
        population_pockets[
            [
                "geoid",
                "population_pockets_proxy",
                "avg_house_size",
                "std_house_size",
            ]
        ],
        on="geoid",
        how="left",
    )

    # %% EXPORT DATA FOR DOWNSTREAM ANALYSIS
    # Save original data only (non-imputed)
    merged_df.filter(pl.col("imputation_source").is_null()).write_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )

    # Save full dataset (original + imputed)
    merged_df.write_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned_imputed.csv")
    )

    print(f"\nFinal datasets saved:")
    print(
        f"  - outbreaks_vaccinations_cleaned.csv: {merged_df.filter(pl.col('imputation_source').is_null()).shape[0]} rows (original only)"
    )
    print(
        f"  - outbreaks_vaccinations_cleaned_imputed.csv: {merged_df.shape[0]} rows (original + imputed)"
    )
