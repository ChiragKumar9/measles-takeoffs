import zipfile
from io import BytesIO

import polars as pl
import requests
from census import Census
from tqdm.auto import tqdm


def browse_df(df, page_size=10, auto_exit=False):
    """Browse dataframe page by page"""
    total_rows = df.height
    total_pages = (total_rows + page_size - 1) // page_size

    current_page = 0

    while True:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_rows)

        print(f"\n--- Page {current_page + 1} of {total_pages} ---")
        print(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
        print(df.slice(start_idx, page_size))

        if current_page == total_pages - 1:
            print("\n[Last page reached]")
            if auto_exit:
                break

        command = (
            input(
                "\nEnter 'n' for next, 'p' for previous, 'q' to quit, or row number: "
            )
            .strip()
            .lower()
        )

        if command == "q":
            break
        elif command == "n" and current_page < total_pages - 1:
            current_page += 1
        elif command == "p" and current_page > 0:
            current_page -= 1
        elif command.isdigit():
            row_num = int(command) - 1
            if 0 <= row_num < total_rows:
                page = row_num // page_size
                current_page = page
        else:
            print("Invalid command. Please try again.")


def census_variables(year: int = 2023, survey: str = "acs5") -> pl.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/{survey}/variables.json"
    response = requests.get(url)
    data = response.json()

    # Convert to searchable DataFrame
    variables_df = pl.DataFrame(
        [
            {
                "code": code,
                "label": info.get("label", ""),
                "concept": info.get("concept", ""),
            }
            for code, info in data["variables"].items()
            if code not in ["for", "in", "ucgid"]
        ]
    )

    return variables_df


def fips_land_sizes() -> pl.DataFrame:
    # Download and process in memory
    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"
    response = requests.get(url)

    # Create a ZipFile object from the downloaded content
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        # We know the one file in here
        with zip_ref.open("2020_Gaz_counties_national.txt") as f:
            df = pl.read_csv(
                f, separator="\t", schema_overrides={"GEOID": pl.String}
            )

    return df


class CensusQuery:
    def __init__(self, key: str):
        self.census = Census(key)
        self.land_size = None

    def query_5_9_pop(self, year: list[int], fips: list[str]) -> pl.DataFrame:
        """Get population of children aged 5-9 for given FIPS codes and year."""

        assert len(fips) == len(year), (
            "FIPS and year lists must be the same length."
        )
        pop_data = [
            self.census.acs5.state_county(
                fields=[
                    "B01001_004E",
                    "B01001_028E",
                    "B14003_005E",
                    "B14003_014E",
                    "B14003_033E",
                    "B14003_023E",
                    "B14003_051E",
                    "B14003_042E",
                ],
                state_fips=fip[:2],
                county_fips=fip[2:],
                year=y,
            )
            for fip, y in tqdm(zip(fips, year), total=len(fips))
        ]
        pop_df = [pl.DataFrame(pop_datum) for pop_datum in pop_data]
        pop_df = pl.concat(pop_df)
        pop_df = pop_df.with_columns(
            fips=pl.col("state") + pl.col("county"),
            pop_5_9=pl.col("B01001_004E") + pl.col("B01001_028E"),
            public_schooled_5_9=pl.col("B14003_005E") + pl.col("B14003_033E"),
            private_schooled_5_9=pl.col("B14003_014E") + pl.col("B14003_042E"),
            not_enrolled_5_9=pl.col("B14003_023E") + pl.col("B14003_051E"),
        ).select(
            [
                "fips",
                "pop_5_9",
                "public_schooled_5_9",
                "private_schooled_5_9",
                "not_enrolled_5_9",
            ]
        )
        return pop_df

    def query_land_size(self, fips: list[str]) -> pl.DataFrame:
        """Get land size in square km for given FIPS codes."""
        if self.land_size is None:
            self.land_size = fips_land_sizes()

        land_df = self.land_size
        land_df = land_df.with_columns(
            fips=pl.col("GEOID"),
            # land area is in square meters, convert to square km
            land_size_sq_km=(
                pl.col("ALAND").cast(pl.Float64, strict=False) / (1000 * 1000)
            ),
        ).select(["fips", "land_size_sq_km"])
        land_df = land_df.filter(pl.col("fips").is_in(fips))
        return land_df
