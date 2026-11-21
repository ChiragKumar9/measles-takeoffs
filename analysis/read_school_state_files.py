"""
Read the school-level files published by states (the "state files" of the
school-data zip in data/, drive-download-*.zip) into one table in the layout of
school_vaccination.standardize, for analysis/assemble_school_vaccination.py.

Fifteen states have a reader (outbreak_size.school_state_files): California,
Colorado, Indiana, Iowa, Maine, Maryland, Massachusetts, Minnesota, New York,
North Carolina, Ohio, Pennsylvania, South Carolina, Texas and Wisconsin. Iowa,
Maine and South Carolina are read from PDFs with `pdftotext` (poppler), which
must be installed. The county of each school is the
county its file names; a school whose label names two counties has no county
and is left out (printed). Row ids are unique across states.

Run from the repo root, before assemble_school_vaccination.py:
    uv run analysis/read_school_state_files.py

Writes data/schools/state_files_schools.csv, and in outputs/school_vaccination/
the QC log of the readers (state_files_qc_log.csv).
"""

import glob
import io
import os
import re
import subprocess
import zipfile

import fastexcel
import geopandas as gpd
import polars as pl

from outbreak_size import school_state_files as ssf
from outbreak_size import school_vaccination as sv

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "school_vaccination"
)
os.makedirs(output_dir, exist_ok=True)

(zip_path,) = glob.glob(os.path.join(data_dir, "drive-download-*.zip"))
ZIP = zipfile.ZipFile(zip_path)
counties = pl.from_pandas(
    gpd.read_file(
        os.path.join(
            data_dir, "cb_2020_us_county_500k", "cb_2020_us_county_500k.shp"
        )
    ).rename(
        columns={
            "GEOID": "county_fips",
            "NAME": "county_name",
            "STUSPS": "county_state",
        }
    )[["county_fips", "county_name", "county_state"]]
)
log = sv.QCLog()
frames = []  # one frame per state and file, with the columns of its reader


def files(prefix, suffix=(".xlsx", ".xls", ".csv", ".pdf")):
    return sorted(
        n
        for n in ZIP.namelist()
        if n.startswith(prefix) and n.endswith(suffix)
    )


def year_of(name):
    """The first year of the school year in a file name ("2023-24", "2023_2024")."""
    return int(re.search(r"(20\d\d)[-_ ]*(\d{2,4})", name).group(1))


# Maryland
for name in files("Maryland/"):
    y = year_of(name)
    df, _ = ssf.parse_maryland(
        ssf.read_sheet(ZIP.read(name), "Kindergarten"), y, log
    )
    frames.append(
        ssf.attach_counties(df, "MD", counties).with_columns(
            state=pl.lit("MD"), rate_source=pl.lit("mmr")
        )
    )

# New York: the two original files (combined.csv is their stack)
ny_tables = [
    pl.read_csv(
        io.BytesIO(ZIP.read(n)), infer_schema_length=0, encoding="utf8-lossy"
    )
    for n in files("New York/")
    if "combined" not in n
]
df, _ = ssf.parse_new_york(pl.concat(ny_tables, how="diagonal_relaxed"), log)
frames.append(
    ssf.attach_counties(df, "NY", counties).with_columns(
        state=pl.lit("NY"), rate_source=pl.lit("mmr")
    )
)

# California
for name in files("California/"):
    df, _ = ssf.parse_california(
        ssf.read_sheet(ZIP.read(name), "Enrollment 20 or More"),
        year_of(name),
        log,
    )
    frames.append(
        ssf.attach_counties(df, "CA", counties).with_columns(
            state=pl.lit("CA")
        )
    )

# South Carolina (45-day report PDFs)
for name in files("South Carolina/"):
    y = int(re.search(r"_(\d\d)-(\d\d)_", name).group(1)) + 2000
    text = subprocess.run(
        ["pdftotext", "-layout", "-", "-"],
        input=ZIP.read(name),
        capture_output=True,
        check=True,
    ).stdout.decode("utf8")
    df, _ = ssf.parse_south_carolina(text, log)
    frames.append(
        ssf.attach_counties(df, "SC", counties).with_columns(
            state=pl.lit("SC"), year_start=pl.lit(y, dtype=pl.Int64)
        )
    )

# Iowa: one PDF a year (all grades, the rate of all required vaccines: not
# kindergarten MMR; the Chen & Bento and Washington Post Iowa values are the
# same measure). The school year comes from the title of the PDF.
for name in files("Iowa/", (".pdf",)):
    text = subprocess.run(
        ["pdftotext", "-layout", "-", "-"],
        input=ZIP.read(name),
        capture_output=True,
        check=True,
    ).stdout.decode("utf8")
    df, _ = ssf.parse_iowa(text, None, log)
    frames.append(
        ssf.attach_counties(df, "IA", counties).with_columns(
            state=pl.lit("IA")
        )
    )

# Maine: Excel files from 2018-19 and PDFs for 2014-15 to 2017-18 (read from
# the word coordinates of `pdftotext -bbox`); each parser checks that the
# students of the parsed rows add up to the printed statewide total
for name in files("Maine/", (".xlsx", ".pdf")):
    y = int(re.search(r"(20\d\d)", name).group(1))
    if name.lower().endswith(".xlsx"):
        df, _ = ssf.parse_me_excel(
            ssf.read_sheet(ZIP.read(name), "Sheet1"), y, log
        )
    else:
        bbox = subprocess.run(
            ["pdftotext", "-bbox", "-", "-"],
            input=ZIP.read(name),
            capture_output=True,
            check=True,
        ).stdout.decode("utf8")
        df, _ = ssf.parse_me_pdf(bbox, y, log)
    frames.append(
        ssf.attach_counties(df, "ME", counties).with_columns(
            state=pl.lit("ME")
        )
    )

# Wisconsin: one file with every school year (2018-19 on) in the "By School"
# sheet; the kindergarten MMR column exists from 2024-25 on, and before that
# the all-vaccine column is used (rate_source says which)
for name in files("Wisconsin/", (".xlsx",)):
    df, _ = ssf.parse_wisconsin(
        ssf.read_sheet(ZIP.read(name), "By School"), None, log
    )
    frames.append(
        ssf.attach_counties(df, "WI", counties).with_columns(
            state=pl.lit("WI")
        )
    )

# Indiana: one Excel file a year (school years 2023-24 on)
for name in files("Indiana/", (".xlsx",)):
    y = int(re.search(r"school-year-(20\d\d)-(20\d\d)", name).group(1))
    sheet = pl.read_excel(
        io.BytesIO(ZIP.read(name)),
        read_options={"header_row": None},
        infer_schema_length=0,
    )
    df, _ = ssf.parse_indiana(sheet, y, log)
    frames.append(
        ssf.attach_counties(df, "IN", counties).with_columns(
            state=pl.lit("IN")
        )
    )

# North Carolina: one Excel file in the folder of each school year; the files
# do not say "kindergarten" (the enrollment adds up to one cohort); the rate
# is all-vaccine
nc_files = {}
for name in files("North Carolina/", (".xlsx",)):
    m = re.fullmatch(r"North Carolina/(20\d\d)-(20\d\d)/[^/]+\.xlsx", name)
    if m:
        assert int(m.group(2)) == int(m.group(1)) + 1, name
        assert int(m.group(1)) not in nc_files, f"two xlsx for {m.group(1)}"
        nc_files[int(m.group(1))] = name
for y, name in sorted(nc_files.items()):
    sheet = pl.read_excel(
        io.BytesIO(ZIP.read(name)),
        read_options={"header_row": None},
        infer_schema_length=0,
    )
    df, _ = ssf.parse_north_carolina(sheet, y, log)
    frames.append(
        ssf.attach_counties(df, "NC", counties).with_columns(
            state=pl.lit("NC")
        )
    )

# Colorado: two long csv files (70-140 MB); one at a time, only the
# kindergarten MMR rows. A school year is in one file only.
CO_COLUMNS = [
    "ID",
    "Site_Name",
    "District_Name",
    "Facility_Type",
    "County",
    "Survey_Type",
    "Vaccine",
    "Metric",
    "Value_Percent",
    "Enrollment",
    "Year",
]
co_year_file = {}
for name in files("Colorado/", (".csv",)):
    full = pl.read_csv(
        io.BytesIO(ZIP.read(name)), infer_schema_length=0, columns=CO_COLUMNS
    )
    small = full.filter(
        (pl.col("Survey_Type") == "Kindergarten")
        & (pl.col("Vaccine") == "MMR")
    )
    del full
    for label in sorted(small["Year"].unique().to_list()):
        y = int(label[:4])
        assert label == f"{y}/{y + 1}", label
        assert y not in co_year_file, f"school year {y} is in two files"
        co_year_file[y] = name
        df, _ = ssf.parse_colorado(small, y, log)
        frames.append(
            ssf.attach_counties(df, "CO", counties).with_columns(
                state=pl.lit("CO")
            )
        )
    del small

# Massachusetts: one workbook a year, the sheet with "kinder" in its name
# (the undated kinder-rates-by-school.xls is left out)
for name in files("Massachusetts/", (".xls", ".xlsx")):
    y = ssf.file_year_massachusetts(name)
    if y is None:
        print("Massachusetts file without a school year, left out:", name)
        continue
    data = ZIP.read(name)
    (sheet_name,) = [
        n
        for n in fastexcel.read_excel(data).sheet_names
        if "kinder" in n.lower()
    ]
    sheet = pl.read_excel(
        io.BytesIO(data),
        sheet_name=sheet_name,
        read_options={"header_row": None},
        infer_schema_length=0,
    )
    df, _ = ssf.parse_massachusetts(sheet, y, log)
    frames.append(
        ssf.attach_counties(df, "MA", counties).with_columns(
            state=pl.lit("MA")
        )
    )

# Minnesota: one workbook a year, sheet "K_School"; the years without a
# County column (2013-2018 and 2020) are not read (MNOldLayout)
for name in files("Minnesota/", (".xlsx",)):
    y = ssf.file_year_minnesota(name)
    sheet = pl.read_excel(
        io.BytesIO(ZIP.read(name)),
        sheet_name="K_School",
        read_options={"header_row": None},
        infer_schema_length=0,
    )
    try:
        df, _ = ssf.parse_minnesota(sheet, y, log)
    except ssf.MNOldLayout as error:
        print(f"Minnesota {y} not read: {error}")
        continue
    frames.append(
        ssf.attach_counties(df, "MN", counties).with_columns(
            state=pl.lit("MN")
        )
    )

# Pennsylvania: one csv (2025-26), kindergarten rows only
for name in files("Pennsylvania/", (".csv",)):
    sheet = pl.read_csv(
        io.BytesIO(ZIP.read(name)), has_header=False, infer_schema_length=0
    )
    df, _ = ssf.parse_pennsylvania(
        sheet, ssf.file_year_pennsylvania(name), log
    )
    frames.append(
        ssf.attach_counties(df, "PA", counties).with_columns(
            state=pl.lit("PA")
        )
    )

# Texas: one workbook a year, sheet "Coverage by District"; public school
# rows are districts (unit "district"), private school rows are schools
for name in files("Texas/", (".xls", ".xlsx")):
    sheet = pl.read_excel(
        io.BytesIO(ZIP.read(name)),
        sheet_name="Coverage by District",
        read_options={"header_row": None},
        infer_schema_length=0,
    )
    df, _ = ssf.parse_texas(sheet, ssf.file_year_texas(name), log)
    frames.append(
        ssf.attach_counties(df, "TX", counties).with_columns(
            state=pl.lit("TX")
        )
    )

# Ohio: one Excel file a year; the sheet of the kindergarten schools is named differently
OHIO_SHEETS = {
    2015: "2015-16 School Iz Report_Final",
    2016: "Kindergarten, 2016-2017",
    2017: "Kindergarten_School",
    2018: "Kindergarten_School",
    2019: "K_School Level Single Antigen",
    2020: "Kindergarten_School",
    **{y: "Kindergarten_School Level" for y in range(2021, 2026)},
}
for name in files("Ohio/", (".xlsx",)):
    y = year_of(name)
    df, _ = ssf.parse_ohio(
        ssf.read_sheet(ZIP.read(name), OHIO_SHEETS[y]), y, log
    )
    frames.append(
        ssf.attach_counties(df, "OH", counties).with_columns(
            state=pl.lit("OH")
        )
    )

# the common layout; a row id and a rid that are unique across states
table = pl.concat(
    [ssf.finish(df, df["state"][0]) for df in frames], how="diagonal_relaxed"
)
table = (
    table.drop("rid", "row_id")
    .with_row_index("rid")
    .with_columns(
        row_id=pl.format(
            "state_file:{}:{}:{}",
            pl.col("state"),
            pl.col("year_start"),
            pl.col("rid"),
        ),
        rid=pl.col("rid").cast(pl.UInt32),
    )
)
no_county = table.filter(pl.col("county_fips").is_null())
print(
    f"{no_county.height} schools without a single county (a label of two "
    "counties, or the county 'Unknown') are left out:",
    no_county.group_by("state").len().sort("state").rows(),
)
table = table.filter(pl.col("county_fips").is_not_null())
os.makedirs(os.path.join(data_dir, "schools"), exist_ok=True)
table.write_csv(os.path.join(data_dir, "schools", "state_files_schools.csv"))
log.rows_frame().write_csv(os.path.join(output_dir, "state_files_qc_log.csv"))
pl.Config.set_tbl_rows(100)
print(
    table.group_by("state", "year_start")
    .agg(
        schools=pl.len(),
        with_rate=pl.col("rate").is_not_null().sum(),
        median_rate=pl.col("rate").median(),
    )
    .sort("state", "year_start")
)
