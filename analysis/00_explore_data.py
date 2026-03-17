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

from outbreak_size import synth_pop_clusters, utils, vaccination_wrangling

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
    df = pl.read_excel(os.path.join(data_dir, "Combined Measles Dataset.xlsx"))

    # to ensure consistent lower case column names
    df = df.rename({col: col.lower() for col in df.columns})

    df_2025 = pl.read_csv(
        os.path.join(
            data_dir, "measles_data", "measles_county_all_updates.csv"
        ),
        schema_overrides={"location_id": pl.String},
    )
    # we can't do anything with the 2026 data because we don't have 2025 vaccination
    # coverage
    df_2025 = df_2025.filter(pl.col("date").str.starts_with("2025"))
    # rename columns to match the main df
    df_2025 = df_2025.rename(
        {
            "location_id": "fips",
            "location_name": "county",
            "value": "cases",
        }
    )

    df_2025 = df_2025.with_columns(
        year=pl.lit(2025),
        state=pl.col("county").str.split(", ").list.get(1),
        county=pl.when(
            pl.col("county").is_in(list(higher_level_regions().keys()))
        )
        .then(pl.col("county"))
        .otherwise(pl.col("county").str.split(", ").list.get(0)),
    )
    # set the fips for any county that is a higher level region to be the name
    # of the county
    df_2025 = df_2025.with_columns(
        fips=pl.when(
            pl.col("county").is_in(list(higher_level_regions().keys()))
        )
        .then(pl.col("county"))
        .otherwise(pl.col("fips"))
    )

    # group by the case to get total 2025 cases
    df_2025 = df_2025.group_by(["year", "state", "county", "fips"]).agg(
        pl.sum("cases")
    )

    df_2025_with_imports = pl.read_csv(
        os.path.join(
            data_dir, "measles_data", "measles_county_all_updates_detailed.csv"
        ),
        schema_overrides={"location_id": pl.String},
    )
    df_2025_with_imports = df_2025_with_imports.filter(
        pl.col("outcome_type") == "case_imported"
    )
    # rename columns to match the main df
    df_2025_with_imports = df_2025_with_imports.rename(
        {
            "location_id": "fips",
            "location_name": "county",
            "value": "imported",
        }
    )
    # groupby
    df_2025_with_imports = df_2025_with_imports.group_by("fips").agg(
        pl.sum("imported")
    )

    # join
    df_2025 = df_2025.join(
        df_2025_with_imports.select(["fips", "imported"]),
        on="fips",
        how="left",
    )
    df_2025 = df_2025.with_columns(
        indigenous=pl.col("cases") - pl.col("imported")
    )

    # Lauren flagged some funky places that should be dropped
    # drop snohomish
    df_2025 = df_2025.filter(
        ~(
            (pl.col("county") == "Snohomish")
            & (pl.col("state") == "Washington")
        )
    )
    # drop "Western" which is really a county!
    df_2025 = df_2025.filter(
        ~((pl.col("county") == "Western") & (pl.col("state") == "Iowa"))
    )
    # unknown county Oklahoma is treated as a known county with a fips
    df_2025 = df_2025.filter(
        ~(
            (pl.col("county") == "Unknown County")
            & (pl.col("state") == "Oklahoma")
        )
    )
    # drop Orange county California
    df_2025 = df_2025.filter(
        ~((pl.col("county") == "Orange") & (pl.col("state") == "California"))
    )
    # drop Central, Utah
    df_2025 = df_2025.filter(
        ~((pl.col("county") == "Central") & (pl.col("state") == "Utah"))
    )
    # drop Central Region, Virginia
    df_2025 = df_2025.filter(
        ~(
            (pl.col("county") == "Central Region")
            & (pl.col("state") == "Virginia")
        )
    )
    # drop Unknown County, South Carolina
    df_2025 = df_2025.filter(
        ~(
            (pl.col("county") == "Unknown County")
            & (pl.col("state") == "South Carolina")
        )
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
        # East Tennessee consists of Anderson, Bledsoe, Blount, Bradley, Campbell, Carter, Claiborne,
        # Cocke, Cumberland, Grainger, Greene, Hamblem, Hamilton, Hancock, Hawkins, Jefferson, Johnson,
        # Knox, Loudon, McMinn, Meigs, Monroe, Morgan, Polk, Rhea, Roane, Scott, Sevier, Sullivan, Unicoi,
        # Union, Washington
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
        # "North, Alabama": [
        #     "01019",
        #     "01033",
        #     "01043",
        #     "01049",
        #     "01059",
        #     "01071",
        #     "01077",
        #     "01079",
        #     "01083",
        #     "01089",
        #     "01095",
        #     "01103",
        #     "01133",
        # ],
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
        # "Upstate, South Carolina": [
        #     "45045",
        #     "45083",
        #     "45007",
        #     "45077",
        #     "45073",
        #     "45047",
        #     "45059",
        #     "45021",
        #     "45087",
        #     "45001",
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
            data_dir, "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied.xlsx"
        )
    )

    # remove the funky places lauren found in her QC
    # exclude Butts, Georgia, 2024
    df = exclude_county_state_year(df, "Butts", "Georgia", 2024)
    # exclude Jasper, Georgia 2024
    df = exclude_county_state_year(df, "Jasper", "Georgia", 2024)
    # exclude Lumpkin, Georgia 2024
    df = exclude_county_state_year(df, "Lumpkin", "Georgia", 2024)
    # exclude Richmond, Georgia 2024
    df = exclude_county_state_year(df, "Richmond", "Georgia", 2024)
    # exclude schley, georgia 2024
    df = exclude_county_state_year(df, "Schley", "Georgia", 2024)
    # exclude baltimore city, maryland 2019
    df = exclude_county_state_year(df, "Baltimore City", "Maryland", 2019)
    # drop slope county 2021
    df = exclude_county_state_year(df, "Slope", "North Dakota", 2021)
    # for Cocke, Tennessee 2024, number should be corrected from 4.5 to 94.5
    df = df.with_columns(
        MMR_cov=pl.when(
            (pl.col("county") == "Cocke")
            & (pl.col("state") == "Tennessee")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(94.5))
        .otherwise(pl.col("MMR_cov"))
    )
    # fill in states that did not report medical exemptions separately
    # as having medical exemptions of 0
    # these two states are wyoming and louisiana
    # we also do this for vermont and mississippi, which censored medical exemptions
    # whenever medical exemptions for these states are null, fill them with 0
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
        # fips codes are not always 5 digits -- those with leading zeros are 4
        # so add the leading zeros back in
        # but only do this when the fips code is not two digits
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
                ),  # type: ignore
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
    temp_df = temp_df.with_columns(
        pop_5=pop_df["pop_5_9"]
        / 5  # entering kindergarteners ~ 1/5 of 5-9 pop
    )
    # calculate a weight
    temp_df = temp_df.with_columns(
        # this is a particular county's importance in the region for that year
        weight=pl.col("pop_5") / pl.sum("pop_5").over("year")
    )
    # Let's aggregate up to the year level
    temp_df = temp_df.group_by("year").agg(
        estimated_mmr_cov=(
            pl.col("estimated_mmr_cov") * pl.col("weight")
        ).sum(),
        nonmed_exempt_pct=(
            pl.col("nonmed_exempt_pct") * pl.col("weight")
        ).sum(),
        med_exempt_pct=(pl.col("med_exempt_pct") * pl.col("weight")).sum(),
        # Just sum the population
        pop_5=pl.col("pop_5").sum(),
    )
    # Now we to condense these estimates across years into one estimate over the time range
    # We need to weight by population again
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

    # Nebraska reports vaccination data by their health district regions
    # We down-aggregate here because the only county for which we have Nebraska
    # measles cases is Douglas county, which is anyways by itself as a health district
    nebraska_fips = {
        "old_nebraska": [
            *["310012"] * 3,  # Central
            "310021",  # Dakota
            "310002",  # Douglas
            *["310005"] * 4,  # East Central
            *["310008"] * 4,  # Elkhorn Logan Valley
            *["310007"] * 4,  # Four Corners
            "310003",  # Lincoln-Lancaster
            *["310015"] * 9,  # Loup Basin
            *["310020"] * 9,  # North Central
            *["310010"] * 4,  # Northeast
            *["310018"] * 12,  # Panhandle
            *["310011"] * 5,  # Public Health Solutions
            *["310006"] * 2,  # Sarpy/Cas
            *["310013"] * 4,  # South Heartland
            *["310009"] * 5,  # Southeast
            *["310017"] * 9,  # Southwest
            *["310004"] * 3,  # Three Rivers
            *["310014"] * 7,  # Two Rivers
            *["310016"] * 6,  # West Central
        ],
        "new_nebraska": [
            "31079",
            "31081",
            "31121",  # Hall, Hamilton, and Merrick
            "31043",  # Dakota
            "31055",  # Douglas
            "31011",
            "31037",
            "31125",
            "31141",  # Boone, Colfax, Nance, and Platte
            "31021",
            "31039",
            "31119",
            "31167",  # Burt, Cuming, Madison, Stanton
            "31023",
            "31143",
            "31159",
            "31185",  # Bulter, Polk, Seward, York
            "31109",  # Lancaster
            "31009",
            "31041",
            "31071",
            "31077",
            "31093",
            "31115",
            "31163",
            "31175",
            "31183",  # Blaine, Custer, Garfield, Greeley, Howard, Loup, Sherman, Valley, Wheeler
            "31003",
            "31015",
            "31017",
            "31031",
            "31089",
            "31103",
            "31107",
            "31139",
            "31149",  # Antelope, Boyd, Brown, Cherry, Holt, Keya Paha, Knox, Pierce, Rock
            "31027",
            "31051",
            "31173",
            "31179",  # Cedar, Dixon, Thurston, Wayne
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
            "31105",  # Banner, Box Butte, Cheyenne, Dawes, Deuel, Garden, Grant, Kimball, Sioux, Scotts Bluff, Morrill, Sheridan
            "31059",
            "31067",
            "31095",
            "31151",
            "31169",  # Fillmore, Gage, Jefferson, Saline, Thayer
            "31153",
            "31025",  # Sarpy, Cass
            "31001",
            "31035",
            "31129",
            "31181",  # Adams, Clay, Nuckolls, Webster
            "31097",
            "31127",
            "31131",
            "31133",
            "31147",  # Johnson, Nemaha, Otoe, Pawnee, Richardson
            "31029",
            "31057",
            "31063",
            "31065",
            "31085",
            "31087",
            "31101",
            "31135",
            "31145",  # Chase, Dundy, Frontier, Furnas, Hayes, Hitchcock, Keith, Perkins, Red Willow
            "31053",
            "31177",
            "31155",  # Dodge, Washington, Saunders
            "31019",
            "31047",
            "31061",
            "31073",
            "31083",
            "31099",
            "31137",  # Buffalo, Dawson, Franklin, Gosper, Harlan, kearney, Phelps
            "31005",
            "31091",
            "31111",
            "31113",
            "31117",
            "31171",  # Arthur, Hooker, Lincoln, Logan, McPherson, Thomas
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

    # Indiana only reports state level data and only for exemptions
    # That's too much extrapolation for what we want to do, so not going to touch
    # those data and we just drop them

    # Let's make fake columns for the places where we have measles cases at a
    # higher geographic resolution than the vaccination data so that we
    # don't accidentally drop them in the merge

    # We want to tack on our special locations too
    # These are locations that require multiple fips codes of analysis
    assignment_special_places = higher_level_regions()

    for place, fips_list in assignment_special_places.items():
        if place == "Gulf Coast":
            continue  # We have vaccination data for the region for this one

        state = vaccinations_df.filter(pl.col("fips") == fips_list[0])[
            "state"
        ][0]

        for year in vaccinations_df["year"].unique().to_list():
            temp = query_avg_vaccinations_region_years(
                vaccinations_df, years=[year], fips=fips_list
            )

            # make a temp df
            temp_df = pl.DataFrame(temp)
            # add year, state, county columns
            temp_df = temp_df.with_columns(
                year=pl.lit(year),
                state=pl.lit(state),
                county=pl.lit(place),
                fips=pl.lit(None),
            )

            # append to vaccinations_df
            vaccinations_df = pl.concat(
                [vaccinations_df, temp_df], how="diagonal_relaxed"
            )

    # in some cases, we have longer than we need fips details
    vaccinations_df = vaccinations_df.with_columns(
        # Kansas fips codes are weird because NBC team created their own effectively
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
    merged_df = outbreaks_df.join(
        vaccinations_df,
        left_on=["year_relevant_vaccination", "state", "county"],
        right_on=["year", "state", "county"],
        how="left",
    )
    # we'll have two fips codes here -- one from the vaccine data and other from 2025
    merged_df = merged_df.with_columns(
        fips=pl.when(pl.col("fips_right").is_not_null())
        .then(pl.col("fips_right"))
        .otherwise(pl.col("fips"))
    )
    merged_df = merged_df.drop(["fips_right"])
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
    # add a land size variable
    land_df = census_query.query_land_size(fips)
    merged_df = merged_df.join(land_df, on="fips", how="left")

    merged_df = merged_df.with_columns(
        pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
        # these quantities are calculated only for people aged 5-9
        fraction_public=(pl.col("public_schooled") / pl.col("pop_5_9")),
        fraction_private=(pl.col("private_schooled") / pl.col("pop_5_9")),
        # I previously checked by hand that public+private+not_enrolled = total pop
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
        cases_outbreak=pl.when(
            pl.col("imported").is_null() | pl.col("imported") == 0
        )
        .then(pl.col("cases"))
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
    vaccinations_df.write_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv")
    )
    # Merge
    raw_merged_df = merge_outbreaks_vaccinations(outbreaks_df, vaccinations_df)
    # Let's save this dataset so we have it for plotting
    raw_merged_df.write_csv(
        os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv")
    )
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
    # plt.show()

    # Compare distribution of timing of outbreaks with and without vaccination data
    fig, ax = compare_distributions(
        raw_merged_df["year"],
        merged_df["year"],
        xlabel="Year",
        label1="Without vaccination data",
        label2="With vaccination data",
        log_scale=False,
    )
    # plt.show()

    # %% VACCINATION COVERAGE AND OUTBREAK SIZE
    # add population variables to make it easy to consider population-dependent
    # variables in the exploration
    # add census variables
    assert census_key is not None, (
        "A Census API key is required to run this analysis."
    )
    # Before we do this, we need to remove the places that are special
    special_places_dict = higher_level_regions()
    special = merged_df.filter(
        pl.col("county").is_in(special_places_dict.keys())
    )
    merged_df = merged_df.filter(
        ~pl.col("county").is_in(special_places_dict.keys())
    )
    merged_df = add_pop_variables(merged_df, census_key)
    # Query population for special places and re-append
    for place, fips_list in special_places_dict.items():
        # let's extract the place from our special df
        temp_df = special.filter(pl.col("county") == place)
        if temp_df.height == 0:
            continue  # something missing in the vaccine data for this special place
        # repeat each row for all the constituent fips codes
        temp_df = pl.concat(
            [temp_df] * len(fips_list), how="diagonal_relaxed"
        ).with_columns(fips=pl.Series(fips_list))
        # add pop variables
        temp_df = add_pop_variables(temp_df, census_key)
        # aggregate back up to the special place level
        temp_df = temp_df.with_columns(
            # make a population-weighting variable
            weight_15_up=pl.col("pop_15_up")
            / pl.sum("pop_15_up").over("year"),
            weight_25_up=pl.col("pop_25_up")
            / pl.sum("pop_25_up").over("year"),
            weight_overall=pl.col("pop") / pl.sum("pop").over("year"),
        )
        temp_df = temp_df.group_by(["year"]).agg(
            # state and county are givens
            state=pl.first("state"),
            county=pl.lit(place),
            # cases have to be kept
            cases=pl.first("cases"),
            cases_outbreak=pl.first("cases_outbreak"),
            imported=pl.first("imported"),
            indigenous=pl.first("indigenous"),
            # vaccination variables are already population-weighted averages
            estimated_mmr_cov=pl.first("estimated_mmr_cov"),
            nonmed_exempt_pct=pl.first("nonmed_exempt_pct"),
            med_exempt_pct=pl.first("med_exempt_pct"),
            total_exempt_pct=pl.first("total_exempt_pct"),
            # populations are summed
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
            # other variables are population-weighted averages
            median_household_income=(
                pl.col("median_household_income") * pl.col("weight_15_up")
            ).sum(),
            fraction_white=(
                pl.col("fraction_white") * pl.col("weight_overall")
            ).sum(),
        )
        # We have to make our own fraction public/private/not enrolled
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
        # append back to merged_df
        merged_df = pl.concat([merged_df, temp_df], how="diagonal_relaxed")

    # While we're here, let's make some other composite variables we need
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

    # MMR vaccination coverage
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["estimated_mmr_cov"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["estimated_mmr_cov"],
        xlabel="Estimated MMR vaccination coverage (%)",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(75, 100, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # MMR vaccination coverage
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["estimated_mmr_cov"],
        merged_df.filter(pl.col("cases_outbreak") >= 5)["estimated_mmr_cov"],
        xlabel="Estimated MMR vaccination coverage (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 5)",
        log_scale=False,
        bins=np.arange(75, 100, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # MMR vaccination coverage
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["estimated_mmr_cov"],
        merged_df.filter(pl.col("cases_outbreak") >= 10)["estimated_mmr_cov"],
        xlabel="Estimated MMR vaccination coverage (%)",
        label1="No outbreak",
        label2="Outbreak (Cases >= 10)",
        log_scale=False,
        bins=np.arange(75, 100, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

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

        fig, ax = compare_distributions(
            merged_df.filter(pl.col("cases_outbreak") == 0)[
                "estimated_mmr_cov_5yr"
            ],
            merged_df.filter(
                pl.col("cases_outbreak") >= params["inclusion_threshold"]
            )["estimated_mmr_cov_5yr"],
            xlabel="Estimated 5 yr MMR vaccination coverage (%)",
            label1="No outbreak",
            label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
            log_scale=False,
            bins=np.arange(75, 100, 0.01),
            density=True,
            cumulative=True,
        )
        # plt.show()

        fig, ax = compare_distributions(
            merged_df.filter(pl.col("cases_outbreak") == 0)[
                "estimated_mmr_cov_5yr"
            ],
            merged_df.filter(pl.col("cases_outbreak") >= 5)[
                "estimated_mmr_cov_5yr"
            ],
            xlabel="Estimated 5 yr MMR vaccination coverage (%)",
            label1="No outbreak",
            label2="Outbreak (Cases >= 5)",
            log_scale=False,
            bins=np.arange(75, 100, 0.01),
            density=True,
            cumulative=True,
        )
        # plt.show()

        fig, ax = compare_distributions(
            merged_df.filter(pl.col("cases_outbreak") == 0)[
                "estimated_mmr_cov_5yr"
            ],
            merged_df.filter(pl.col("cases_outbreak") >= 10)[
                "estimated_mmr_cov_5yr"
            ],
            xlabel="Estimated 5 yr MMR vaccination coverage (%)",
            label1="No outbreak",
            label2="Outbreak (Cases >= 10)",
            log_scale=False,
            bins=np.arange(75, 100, 0.01),
            density=True,
            cumulative=True,
        )
        # plt.show()

    fig, ax, _ = pairplot(
        merged_df["estimated_mmr_cov"],
        merged_df["cases_outbreak"],
        ylabel="Outbreak size (cases)",
        alpha=0.5,
        log=True,
    )
    # plt.show()

    if five_year:
        fig, ax, _ = pairplot(
            merged_df["estimated_mmr_cov_5yr"],
            merged_df["cases_outbreak"],
            xlabel="Estimated 5 yr MMR vaccination coverage (%)",
            ylabel="Outbreak size (cases)",
            alpha=0.5,
            log=True,
        )
        # plt.show()

    # %% NONMEDICAL VACCINATION EXEMPTIONS AND OUTBREAK SIZE
    # Nonmedical vaccination exemptions
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["nonmed_exempt_pct"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["nonmed_exempt_pct"],
        xlabel="Nonmedical vaccination exemptions (%)",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(0, 20, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # 5 year nonmedical vaccination exemptions
    if five_year:
        fig, ax = compare_distributions(
            merged_df.filter(pl.col("cases_outbreak") == 0)[
                "nonmed_exempt_pct_5yr"
            ],
            merged_df.filter(
                pl.col("cases_outbreak") >= params["inclusion_threshold"]
            )["nonmed_exempt_pct_5yr"],
            xlabel="Estimated 5 yr nonmedical vaccination exemptions (%)",
            label1="No outbreak",
            label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
            log_scale=False,
            bins=np.arange(0, 20, 0.01),
            density=True,
            cumulative=True,
        )
        # plt.show()

    # %% TOTAL VACCINATION EXEMPTIONS AND OUTBREAK SIZE
    # Total vaccination exemptions
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["total_exempt_pct"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["total_exempt_pct"],
        xlabel="Total vaccination exemptions (%)",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(0, 20, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # 5 year total vaccination exemptions
    if five_year:
        fig, ax = compare_distributions(
            merged_df.filter(pl.col("cases_outbreak") == 0)[
                "total_exempt_pct_5yr"
            ],
            merged_df.filter(
                pl.col("cases_outbreak") >= params["inclusion_threshold"]
            )["total_exempt_pct_5yr"],
            xlabel="Estimated 5 yr total vaccination exemptions (%)",
            label1="No outbreak",
            label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
            log_scale=False,
            bins=np.arange(0, 20, 0.01),
            density=True,
            cumulative=True,
        )
        # plt.show()

    # %% POPULATION SIZE AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["pop_under_5"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["pop_under_5"],
        xlabel="Population size (ages 0-4)",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(100, 650000, 100),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # %% POPULATION DENSITY AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["pop_density"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["pop_density"],
        xlabel=r"Population density (ages 5-9 / km$^2$)",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(0.1, 1000, 0.1),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # %% FRACTION PUBLIC SCHOOL AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["fraction_public"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["fraction_public"],
        xlabel="Fraction of students in public schools",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(0.45, 1.0, 0.001),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # %% FRACTION PRIVATE SCHOOL AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["fraction_private"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["fraction_private"],
        xlabel="Fraction of students in private schools",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(0.0, 0.5, 0.001),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # %% INTERACTION BETWEEN VACCINATION COVERAGE AND PUBLIC SCHOOL
    merged_df = merged_df.with_columns(
        vaccine_coverage_school_interaction=pl.col("estimated_mmr_cov")
        * pl.col("fraction_public")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "vaccine_coverage_school_interaction"
        ],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["vaccine_coverage_school_interaction"],
        xlabel="Vaccination coverage * fraction public school",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(45, 100, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # %% INTERACTION BETWEEN VACCINATION EXEMPTIONS AND PUBLIC SCHOOL
    merged_df = merged_df.with_columns(
        exempt_school_interaction=pl.col("total_exempt_pct")
        * pl.col("fraction_private")
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "exempt_school_interaction"
        ],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["exempt_school_interaction"],
        xlabel="Vaccination exemptions * fraction private school",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        log_scale=False,
        bins=np.arange(0, 3, 0.01),
        density=True,
        cumulative=True,
    )
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

    # %% INCOME AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "median_household_income"
        ],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["median_household_income"],
        xlabel="Median household income (2024 USD)",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        bins=np.arange(40000, 185000, 100),
        density=True,
        cumulative=True,
    )
    # plt.show()

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
    # plt.show()

    # income-exemptions interaction?
    merged_df = merged_df.with_columns(
        income_exempt_interaction=pl.col("median_household_income")
        * (100.0 - pl.col("nonmed_exempt_pct"))
    )
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "income_exempt_interaction"
        ],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["income_exempt_interaction"],
        xlabel="Income * Nonmedical exemptions interaction",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        bins=np.arange(4000000, 18000000, 10000),
        density=True,
        cumulative=True,
    )
    # plt.show()

    # %% EDUCATION AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)[
            "fraction_graduate_degree"
        ],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["fraction_graduate_degree"],
        xlabel="Fraction with graduate degree",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        bins=np.arange(0, 0.33, 0.001),
        density=True,
        cumulative=True,
    )
    # plt.show()

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
    # plt.show()

    # %% RACE AND OUTBREAK SIZE
    fig, ax = compare_distributions(
        merged_df.filter(pl.col("cases_outbreak") == 0)["fraction_white"],
        merged_df.filter(
            pl.col("cases_outbreak") >= params["inclusion_threshold"]
        )["fraction_white"],
        xlabel="Fraction white",
        label1="No outbreak",
        label2=f"Outbreak (Cases >= {params['inclusion_threshold']})",
        bins=np.arange(0.18, 1.0, 0.001),
        density=True,
        cumulative=True,
    )
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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

    # plot
    fig, ax, scatter = pairplot(
        merged_df["population_pockets_proxy"],
        merged_df["cases_outbreak"],
        xlabel="Population pockets proxy",
        ylabel="Outbreak size (cases)",
        log=True,
        c=merged_df["estimated_mmr_cov"],
    )
    # plt.show()

    # %% EXPORT DATA FOR DOWNSTREAM ANALYSIS
    # save the cleaned dataset
    merged_df.write_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )
