"""
Kindergarten membership of the public schools and school districts of the
2024-25 Common Core of Data (CCD, NCES), as two small files in data/ for the
weighted school SD of panel d of the county-level outbreak figure:

  data/ccd_2024_25_schools.csv    one row per public school: NCES school id
      (ncessch), district id (leaid), name, state, county FIPS, latitude and
      longitude (EDGE school locations), whether the school offers
      kindergarten (kg_offered: Yes, No or Not reported) and its kindergarten
      count (kg_count; empty when the school has no count: the count is not
      reported, suppressed or missing) with the CCD flag of the count.
  data/ccd_2024_25_districts.csv  the same for each district (leaid, name,
      kg_count, kg_flag).

The kindergarten count is the STUDENT_COUNT of the CCD membership rows with
GRADE "Kindergarten" and TOTAL_INDICATOR "Subtotal 4 - By Grade" (the count
of the grade over all races and sexes).

Raw files (NCES; download, unzip into one folder and give the folder with
--raw-dir; the membership file is 2.3 GB unzipped and is read in streaming
mode):
  https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2425_w_1a_073025.zip  school directory
  https://nces.ed.gov/ccd/Data/zip/ccd_sch_052_2425_l_1a_073025.zip  school membership
  https://nces.ed.gov/ccd/Data/zip/ccd_lea_029_2425_w_1a_073025.zip  district directory
  https://nces.ed.gov/ccd/Data/zip/ccd_lea_052_2425_l_1a_073025.zip  district membership
  https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_2425.zip  school locations
(the CSV files of the first four and EDGE_GEOCODE_PUBLICSCH_2425.TXT, which
has no header row and is pipe-delimited; the file list of the 2024-25 release
is at https://nces.ed.gov/ccd/files.asp).

usage: uv run analysis/read_ccd_enrollment.py --raw-dir DIR
"""

import argparse
import os

import polars as pl

root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root, "data")

EDGE_COLUMNS = [  # the layout of EDGE_GEOCODE_PUBLICSCH_2425.TXT
    "NCESSCH",
    "LEAID",
    "NAME",
    "OPSTFIPS",
    "STREET",
    "CITY",
    "STATE",
    "ZIP",
    "STFIP",
    "CNTY",
    "NMCNTY",
    "LOCALE",
    "LAT",
    "LON",
    "CBSA",
    "NMCBSA",
    "CBSATYPE",
    "CSA",
    "NMCSA",
    "CD",
    "SLDL",
    "SLDU",
    "SCHOOLYEAR",
]
COUNTED = ["Reported", "Manual adjustment"]  # flags of a usable count


def kindergarten_counts(path, key):
    """The kindergarten count and its flag of each school or district."""
    return (
        pl.scan_csv(path, infer_schema=False)
        .filter(
            (pl.col("GRADE") == "Kindergarten")
            & (pl.col("TOTAL_INDICATOR") == "Subtotal 4 - By Grade")
        )
        .select(
            key,
            kg_count=pl.when(pl.col("DMS_FLAG").is_in(COUNTED))
            .then(pl.col("STUDENT_COUNT").cast(pl.Int64, strict=False))
            .otherwise(None),
            kg_flag="DMS_FLAG",
        )
        .with_columns(
            kg_count=pl.when(pl.col("kg_count") >= 0).then(pl.col("kg_count"))
        )
        .collect(engine="streaming")
    )


def main():
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0]
    )
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()
    raw = args.raw_dir

    edge = pl.read_csv(
        os.path.join(raw, "EDGE_GEOCODE_PUBLICSCH_2425.TXT"),
        separator="|",
        has_header=False,
        new_columns=EDGE_COLUMNS,
        infer_schema=False,
        encoding="utf8-lossy",
    ).select(
        ncessch="NCESSCH",
        leaid="LEAID",
        school_name="NAME",
        state="STATE",
        county_fips="CNTY",
        lat=pl.col("LAT").cast(pl.Float64, strict=False),
        lon=pl.col("LON").cast(pl.Float64, strict=False),
    )
    directory = pl.read_csv(
        os.path.join(raw, "ccd_sch_029_2425_w_1a_073025.csv"),
        infer_schema=False,
        encoding="utf8-lossy",
    ).select(ncessch="NCESSCH", kg_offered="G_KG_OFFERED")
    schools = (
        edge.join(directory, on="ncessch", how="left")
        .join(
            kindergarten_counts(
                os.path.join(raw, "ccd_sch_052_2425_l_1a_073025.csv"),
                pl.col("NCESSCH").alias("ncessch"),
            ),
            on="ncessch",
            how="left",
        )
        .sort("ncessch")
    )
    assert not schools["ncessch"].is_duplicated().any()
    districts = (
        pl.read_csv(
            os.path.join(raw, "ccd_lea_029_2425_w_1a_073025.csv"),
            infer_schema=False,
            encoding="utf8-lossy",
        )
        .select(leaid="LEAID", lea_name="LEA_NAME")
        .join(
            kindergarten_counts(
                os.path.join(raw, "ccd_lea_052_2425_l_1a_073025.csv"),
                pl.col("LEAID").alias("leaid"),
            ),
            on="leaid",
            how="left",
        )
        .sort("leaid")
    )
    assert not districts["leaid"].is_duplicated().any()
    schools.write_csv(os.path.join(data_dir, "ccd_2024_25_schools.csv"))
    districts.write_csv(os.path.join(data_dir, "ccd_2024_25_districts.csv"))
    print(
        f"{schools.height} schools ({schools['kg_count'].is_not_null().sum()} "
        f"with a kindergarten count, {(schools['kg_offered'] == 'No').sum()} "
        f"without kindergarten), {districts.height} districts "
        f"({districts['kg_count'].is_not_null().sum()} with a count)"
    )


if __name__ == "__main__":
    main()
