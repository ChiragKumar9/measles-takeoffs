import os

import polars as pl
from census import Census
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

# we need the california fips codes and county names
# these exist in the vaccination data

df = pl.read_excel(
    os.path.join(
        "data",
        "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied_AK_CA_Updated.xlsx",
    ),
    schema_overrides={"FIPS": pl.String},
)

ca = (
    df.filter(pl.col("state") == "California")
    .select(["year", "state", "county", "FIPS", "county_type", "state_ap"])
    .filter(pl.col("year") == 2010)
)

# duplicate this CSV for all years of interest
years_of_interest = range(2000, 2023)

ca = pl.concat(
    [ca.with_columns(pl.lit(year).alias("year")) for year in years_of_interest]
)

# break up into state and county fips codes
ca = (
    ca.with_columns(fips=pl.col("FIPS").str.zfill(5))
    .drop("FIPS")
    .with_columns(
        state_fips=pl.col("fips").str.slice(0, 2),
        county_fips=pl.col("fips").str.slice(2, 5),
    )
)

# for each county and year, query the census population
# recall that census variable "B01001_003E" is under 5 male population
# and "B01001_027E" is under 5 female population


def query(census, fip, y):
    try:
        fields = [
            "B01001_003E",
            "B01001_027E",
        ]
        fcn = census.acs5.state_county
        args = {"state_fips": fip[:2], "county_fips": fip[2:], "year": y}
        return fcn(
            fields=fields,
            **args,
        )[0]  # census queries return a list of one dict
    except Exception as e:
        return None


c = Census(os.getenv("CENSUS_API_KEY"))

result = [
    query(c, fip, y)
    for fip, y in tqdm(zip(ca["fips"], ca["year"]), total=ca.height)
]
# result is a list of dicts, but we just care about the fields
male_pop = [r["B01001_003E"] if r is not None else None for r in result]
female_pop = [r["B01001_027E"] if r is not None else None for r in result]

ca = ca.with_columns(
    under_5_males=pl.Series(male_pop).cast(pl.Float64),
    under_5_females=pl.Series(female_pop).cast(pl.Float64),
).with_columns(
    under_5_total=pl.col("under_5_males") + pl.col("under_5_females")
)

# export
ca.write_csv(os.path.join("data", "california_population_under_5.csv"))
