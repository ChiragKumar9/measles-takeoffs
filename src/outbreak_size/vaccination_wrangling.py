import glob
import os
import re

import geopandas as gpd
import polars as pl
from addfips import AddFIPS

from outbreak_size.data_year import outbreak_year_of_county_data
from outbreak_size.regions import (
    higher_level_regions,
    members_with_own_row,
    members_without_own_row,
)

# Malformed FIPS codes in the compiled vaccination spreadsheet and what they
# should be: two Virginia codes with a stray trailing digit (Fairfax,
# Greensville) and the District of Columbia's state code
_FIPS_TYPO_FIXES = {"510591": "51059", "510811": "51081", "11": "11001"}

# Valid-looking but wrong codes in the spreadsheet: (state, county, code in
# the spreadsheet) -> the county's real FIPS. 06067 is Sacramento and 30057 is
# Madison County
_FIPS_OVERRIDES = {
    ("California", "San Benito", "06067"): "06069",
    ("Montana", "Lincoln", "30057"): "30053",
}


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# the columns of the compiled vaccination spreadsheet that hold each field a
# QC exclusion can name
_QC_FIELD_COLUMNS = {"coverage": ["MMR_cov", "all_vacc_cov"]}


def apply_qc_exclusions(
    df: pl.DataFrame, exclusions: list[dict]
) -> pl.DataFrame:
    """
    Apply the data QC exclusions (params.yml: qc_exclusions) to the compiled
    vaccination spreadsheet `df` (columns state, county and year, and the
    columns of _QC_FIELD_COLUMNS).

    An exclusion names a state, a county and a year. Without `fields` the
    whole row of that county-year is removed. With `fields` (a list of the
    names of _QC_FIELD_COLUMNS) only those fields are blanked, so the county
    keeps its other values and takes its state's series for the blanked ones.
    An exclusion that matches no row is an error: a misspelled county would
    silently exclude nothing.
    """
    for exclusion in exclusions:
        is_row = (
            (pl.col("state") == exclusion["state"])
            & (pl.col("county") == exclusion["county"])
            & (pl.col("year") == exclusion["year"])
        )
        if df.filter(is_row).height == 0:
            raise ValueError(f"QC exclusion matches no row: {exclusion}")
        fields = exclusion.get("fields")
        if fields is None:
            df = df.filter(~is_row)
            continue
        columns = [c for field in fields for c in _QC_FIELD_COLUMNS[field]]
        df = df.with_columns(
            pl.when(~is_row).then(pl.col(c)).alias(c) for c in columns
        )
    return df


def fix_wrong_fips(df: pl.DataFrame) -> pl.DataFrame:
    """
    Give the rows listed in _FIPS_OVERRIDES their county's real FIPS. A row is
    matched on its state, county and the wrong code, so the county that really
    has that code keeps it.

    df needs the string columns state, county and fips.
    """
    overrides = pl.DataFrame(
        {
            "state": [k[0] for k in _FIPS_OVERRIDES],
            "county": [k[1] for k in _FIPS_OVERRIDES],
            "fips": [k[2] for k in _FIPS_OVERRIDES],
            "fips_override": list(_FIPS_OVERRIDES.values()),
        }
    )
    return (
        df.join(
            overrides,
            on=["state", "county", "fips"],
            how="left",
            maintain_order="left",
        )
        .with_columns(fips=pl.coalesce("fips_override", "fips"))
        .drop("fips_override")
    )


# The old Nebraska health-district codes (six digits). Their rows are not
# counties, but process_vaccinations_df gives each of them to the counties of
# the district (the table nebraska_fips there), so resolve_vaccination_fips
# keeps these codes instead of leaving the FIPS null.
NEBRASKA_DISTRICT_CODES = frozenset(
    {
        "310002",
        "310003",
        "310004",
        "310005",
        "310006",
        "310007",
        "310008",
        "310009",
        "310010",
        "310011",
        "310012",
        "310013",
        "310014",
        "310015",
        "310016",
        "310017",
        "310018",
        "310020",
        "310021",
    }
)

# Rows that pool several counties: (state, county name in the data) -> the
# member counties (name, fips). expand_pooled_rows gives the pooled values to
# each member.
POOLED_ROWS = {
    ("Mississippi", "Issaquena/Sharkey"): (
        ("Issaquena", "28055"),
        ("Sharkey", "28125"),
    ),
}


def resolve_vaccination_fips(df: pl.DataFrame, data_dir: str) -> pl.DataFrame:
    """
    Settle the FIPS code of every row of the compiled vaccination data.

    df needs the string columns state, county and fips (fips already
    zero-padded to five digits where it has more than two characters). The
    rule, in order:
      1. fix the known malformed codes (_FIPS_TYPO_FIXES) and the two known
         wrong codes (_FIPS_OVERRIDES);
      2. keep the spreadsheet's FIPS when it is a county in the 2020 shapefile;
      3. otherwise (missing or not a county) look the county up by name, with
         addfips first and then a case-insensitive match on the shapefile
         county name within the state ("Lasalle" -> "LaSalle");
      4. places that are not a single county (regions, school districts,
         "State" rows, and rows that pool counties, written "A/B") stay null,
         unless their spreadsheet code is a county in the shapefile (step 2
         keeps it; rows that pool counties are always null). The exception
         is an old Nebraska health-district code (NEBRASKA_DISTRICT_CODES),
         which stays so that the district's counties get its values.
    Delaware is left exactly as it is in the spreadsheet (analysis/
    00_explore_data.py pools its three counties).

    A name lookup alone (without keeping valid FIPS) turns counties that share
    a name with an independent city (Baltimore, St. Louis, Bedford, Fairfax,
    Franklin, Richmond, Roanoke) into the city.
    """
    gdf = gpd.read_file(os.path.join(data_dir, "cb_2020_us_county_20m.shp"))
    valid = set(gdf["GEOID"])
    # (state fips, name) -> county, for the names that belong to one county
    # (a county and an independent city can share a name: Baltimore, say)
    names_to_geoids = {}
    for geoid, name in zip(gdf["GEOID"], gdf["NAME"]):
        names_to_geoids.setdefault((geoid[:2], _name_key(name)), []).append(
            geoid
        )
    shapefile_names = {
        key: geoids[0]
        for key, geoids in names_to_geoids.items()
        if len(geoids) == 1
    }
    addfips = AddFIPS()

    original = df["fips"]
    df = df.with_columns(fips=pl.col("fips").replace(_FIPS_TYPO_FIXES))
    df = fix_wrong_fips(df)

    # look up, by name, only the distinct places whose FIPS is not a county
    need = (
        df.filter(
            ~pl.col("fips").is_in(list(valid)).fill_null(False)
            & (pl.col("state") != "Delaware")
        )
        .select("state", "county")
        .unique()
    )

    def lookup(row):
        state, county = row["state"], row["county"]
        if county is None:
            return None
        code = addfips.get_county_fips(county, state)
        # a code that is not a county of the 2020 shapefile is not used
        if code not in valid:
            code = None
        if code is None:
            state_fips = addfips.get_state_fips(state)
            if state_fips is not None:
                code = shapefile_names.get((state_fips, _name_key(county)))
        return code

    need = need.with_columns(
        fips_lookup=pl.struct(["state", "county"]).map_elements(
            lookup, return_dtype=pl.String
        )
    )
    df = df.join(
        need, on=["state", "county"], how="left", maintain_order="left"
    )
    df = df.with_columns(
        fips=pl.when(pl.col("state") == "Delaware")
        .then(original)
        # an old Nebraska health-district code stays: the row is given to the
        # counties of the district later (process_vaccinations_df). This
        # includes "Sarpy/Cass", which pools two counties.
        .when(pl.col("fips").is_in(list(NEBRASKA_DISTRICT_CODES)))
        .then(pl.col("fips"))
        # any other row pooling several counties ("Issaquena/Sharkey") is not
        # one county, even if the spreadsheet lists one county's code; it is
        # given to its members later (expand_pooled_rows)
        .when(pl.col("county").str.contains("/"))
        .then(pl.lit(None, dtype=pl.String))
        .when(pl.col("fips").is_in(list(valid)).fill_null(False))
        .then(pl.col("fips"))
        .otherwise(pl.col("fips_lookup"))
    ).drop("fips_lookup")
    return df


# kansas data is special because it's usually at the school district level,
# but we need it at the county level
# So we hda it queried separately at the county level
def read_kansas_data(data_dir):
    # get all MMR coverage data
    # (sorted: the listing of a folder has no fixed order)
    mmr_files = sorted(glob.glob(os.path.join(data_dir, "MMR", "**", "*.csv")))
    kansas_mmr_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in mmr_files
    ]
    kansas_mmr = pl.concat(kansas_mmr_dfs)
    # do the same for exemption data
    nme_files = sorted(
        glob.glob(os.path.join(data_dir, "Religious Exemption", "**", "*.csv"))
    )
    kansas_nme_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in nme_files
    ]
    kansas_nme = pl.concat(kansas_nme_dfs)
    # do the same for medical exemption data
    me_files = sorted(
        glob.glob(os.path.join(data_dir, "Medical Exemption", "**", "*.csv"))
    )
    kansas_me_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in me_files
    ]
    kansas_me = pl.concat(kansas_me_dfs)

    # now we have to clean and merge
    kansas_mmr = kansas_mmr.with_columns(
        mmr_cov=pl.col("County Coverage")
        .str.replace("%", "")
        .cast(pl.Float64, strict=False),
        year=pl.col("Academic Year")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False),
    ).select(["County", "mmr_cov", "year"])

    kansas_nme = kansas_nme.with_columns(
        nonmed_exempt_pct=pl.col("Exemption String")
        .str.replace("%", "")
        .cast(pl.Float64, strict=False),
        year=pl.col("Academic Year")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False),
    ).select(["County", "nonmed_exempt_pct", "year"])

    kansas_me = kansas_me.with_columns(
        med_exempt_pct=pl.col("Exemption String")
        .str.replace("%", "")
        .cast(pl.Float64, strict=False),
        year=pl.col("Academic Year")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False),
    ).select(["County", "med_exempt_pct", "year"])

    kansas_vacc = kansas_mmr.join(
        kansas_nme, on=["County", "year"], how="inner", maintain_order="left"
    ).join(
        kansas_me, on=["County", "year"], how="inner", maintain_order="left"
    )

    kansas_vacc = kansas_vacc.with_columns(state=pl.lit("Kansas"))

    # we need to add fips codes into this data
    addfips = AddFIPS()
    kansas_vacc = kansas_vacc.with_columns(
        fips=pl.col("County").map_elements(
            lambda county: addfips.get_county_fips(county, "Kansas")
        )
    )

    # rename "County" to be "county" for merging purposes
    kansas_vacc = kansas_vacc.rename({"County": "county"})

    return kansas_vacc


def read_montana_data(data_dir):
    montana = pl.read_excel(
        os.path.join(data_dir, "Montana.xlsx"), read_options={"skip_rows": 1}
    )

    # make the column names actually useful
    # first of all, the columns are:
    columns = [
        "year",
        "state",
        "county",
        "Race",
        "nonmed_exempt_pct",
        "nonmed_exempt_num",
        "med_exempt_pct",
        "MMR_cov",
        "all_vacc_cov",
        "number",
    ]
    # set the column names
    montana.columns = columns
    # drop things we don't need
    montana = montana.drop(["Race", "nonmed_exempt_num", "number"])

    # sometimes the county names are in all caps, sometimes they're not, so let's
    # make them normal case for merging purposes. "Lewis & Clark" is spelled
    # "Lewis And Clark" in the compiled vaccination spreadsheet, so spell the
    # ampersand out to make the merge work
    montana = montana.with_columns(
        county=pl.col("county").str.replace_all("&", "And").str.to_titlecase()
    )

    # we only have coverage data for 2019, and exemption data (2015-2018)
    # before that. The 2019 exemptions are NOT filled from 2018: a copy would
    # be taken for a value of 2019 (the baselines use the county's own 2018
    # exemptions, from the compiled spreadsheet, as its latest values)
    montana = montana.sort(["county", "year"], maintain_order=True)

    # case mmr cov and all vacc cov as floats
    montana = montana.with_columns(
        MMR_cov=pl.col("MMR_cov").cast(pl.Float64, strict=False),
        all_vacc_cov=pl.col("all_vacc_cov").cast(pl.Float64, strict=False),
    )

    # since we can't really QC this data, we want to remove any places that have
    # mmr cov equal to 0, since that's probably a data error
    montana = montana.filter(pl.col("MMR_cov") > 0)
    montana = montana.filter(pl.col("all_vacc_cov") > 0)

    # since we do the merging of estimated_mmr_cov post adding in the montana data,
    # do it here
    montana = montana.with_columns(
        # use mmr cov when available or not zero, otherwise use all vacc cov
        estimated_mmr_cov=pl.when(
            pl.col("MMR_cov").is_not_null() & (pl.col("MMR_cov") > 0)
        )
        .then(pl.col("MMR_cov"))
        .otherwise(pl.col("all_vacc_cov"))
    )

    return montana


def read_oklahoma_data(data_dir):
    # Oklahoma's sheet has the header row first and the description row second
    # (Montana's sheet has these reversed), but both have exactly 2 non-data
    # rows on top, so skip_rows=1 still lands us on the first data row.
    oklahoma = pl.read_excel(
        os.path.join(data_dir, "Oklahoma.xlsx"), read_options={"skip_rows": 1}
    )

    # the columns are:
    columns = [
        "year",
        "state",
        "county",
        "Race",
        "nonmed_exempt_pct",
        "nonmed_exempt_num",
        "med_exempt_pct",
        "MMR_cov",
        "all_vacc_cov",
        "measles_cases",
        "junk",
    ]
    oklahoma.columns = columns
    # drop things we don't need (junk is a trailing stray column in the sheet)
    oklahoma = oklahoma.drop(
        ["Race", "nonmed_exempt_num", "measles_cases", "junk"]
    )

    # drop trailing empty rows past the actual data
    oklahoma = oklahoma.filter(pl.col("year").is_not_null())
    oklahoma = oklahoma.with_columns(year=pl.col("year").cast(pl.Int64))

    # county names have inconsistent casing, so normalize for merging purposes
    oklahoma = oklahoma.with_columns(
        county=pl.col("county").str.to_titlecase()
    )

    # cast to floats. Note: unlike Montana.xlsx (which stores 0-100 values),
    # Oklahoma.xlsx stores these columns as fractions (e.g. 0.95) with Excel's
    # built-in percent display format applied, so they show as "95.00%" when
    # opened in Excel despite the underlying stored value being 0.95. The
    # sheet's own header text ("Percentage (0-100%)") is misleading here.
    # Multiply by 100 so this matches the 0-100 scale used everywhere else.
    oklahoma = oklahoma.with_columns(
        [
            (pl.col(col).cast(pl.Float64, strict=False) * 100).alias(col)
            for col in [
                "nonmed_exempt_pct",
                "med_exempt_pct",
                "MMR_cov",
                "all_vacc_cov",
            ]
        ]
    )

    # since we can't really QC this data, remove rows with no MMR coverage
    # (unlike Montana, don't also filter on all_vacc_cov, since Oklahoma has
    # valid MMR coverage for some rows where all_vacc_cov is 0)
    oklahoma = oklahoma.filter(pl.col("MMR_cov") > 0)

    # since we do the merging of estimated_mmr_cov post adding in the oklahoma
    # data, do it here
    oklahoma = oklahoma.with_columns(
        # use mmr cov when available or not zero, otherwise use all vacc cov
        estimated_mmr_cov=pl.when(
            pl.col("MMR_cov").is_not_null() & (pl.col("MMR_cov") > 0)
        )
        .then(pl.col("MMR_cov"))
        .otherwise(pl.col("all_vacc_cov"))
    )

    return oklahoma


def read_new_york_data(data_dir):
    """
    New York's county vaccination data (New York.xlsx), in the columns of the
    Montana and Oklahoma readers (0-100 scale, county in title case, fips from
    the county name). It replaces the compiled spreadsheet's New York rows
    (see replace_state_rows).
    """
    ny = pl.read_excel(
        os.path.join(data_dir, "New York.xlsx"), has_header=False
    )
    # a row of column codes and three rows of descriptions come first: none of
    # them has a numeric year, and every data row does
    ny.columns = [
        "year",
        "state",
        "county",
        "race",
        "nonmed_exempt_pct",
        "nonmed_exempt_num",
        "med_exempt_pct",
        "MMR_cov",
        "all_vacc_cov",
        "measles_cases",
        "district",
    ]
    ny = ny.with_columns(
        year=pl.col("year").cast(pl.String).cast(pl.Float64, strict=False)
    ).filter(pl.col("year").is_not_null())
    ny = ny.select(
        pl.col("year").cast(pl.Int64),
        "state",
        pl.col("county").str.to_titlecase(),
        pl.col(
            "nonmed_exempt_pct", "med_exempt_pct", "MMR_cov", "all_vacc_cov"
        )
        .cast(pl.String)
        .cast(pl.Float64, strict=False),
    )

    # New York banned nonmedical exemptions in 2019, so their rate is 0 from
    # 2019 on: the sheet leaves it blank (and only from 2019) outside NYC
    ny = ny.with_columns(pl.col("nonmed_exempt_pct").fill_null(0.0))

    addfips = AddFIPS()
    ny = ny.with_columns(
        fips=pl.col("county").map_elements(
            lambda county: addfips.get_county_fips(county, "New York"),
            return_dtype=pl.String,
        ),
        estimated_mmr_cov=pl.when(pl.col("MMR_cov") > 0)
        .then(pl.col("MMR_cov"))
        .otherwise(pl.col("all_vacc_cov")),
    )
    return ny


def replace_state_rows(df, state, new_rows):
    """
    df with every row of `state` replaced by new_rows (columns df lacks are
    null, columns new_rows lacks are dropped from its rows). Unlike the
    Montana and Oklahoma fills, which only fill nulls, nothing of the state's
    old rows is kept.
    """
    new_rows = new_rows.select(
        [col for col in new_rows.columns if col in df.columns]
    )
    return pl.concat(
        [df.filter(pl.col("state") != state), new_rows],
        how="diagonal_relaxed",
    )


# The 2025-26 school year ("year" 2025) of the counties the compiled
# spreadsheet does not have, from the data owner's file of 2 Nov 2026
VACCINATION_2025_FILE = "vaxexempt2026 - forprod [Nov 2].csv"


def read_2025_vaccination_rows(data_dir):
    """
    The 2025 rows of VACCINATION_2025_FILE in the columns of the compiled
    spreadsheet (year, state, county, nonmed_exempt_pct, med_exempt_pct,
    MMR_cov, all_vacc_cov, FIPS, county_type, state_ap), as numbers. A value
    the file marks as suppressed or undefined ("*", "N/A*", "#DIV/0!", ".")
    is missing. The censoring flags (GT, GTE) are not used: a flagged value
    is the bound itself (8 MMR values in 2025).
    """
    raw = pl.read_csv(
        os.path.join(data_dir, VACCINATION_2025_FILE),
        schema_overrides={"FIPS": pl.String},
        infer_schema_length=None,
        glob=False,  # the name has brackets
    ).filter(pl.col("Year") == 2025)

    def number(column):
        return pl.col(column).cast(pl.Float64, strict=False)

    # a state-level row has the county "State" in the spreadsheet
    is_state = (pl.col("county_type") == "State") | (
        pl.col("County") == "State"
    )
    return raw.select(
        year=pl.col("Year"),
        state=pl.col("State"),
        county=pl.when(is_state).then(pl.lit("State")).otherwise("County"),
        nonmed_exempt_pct=number("Non-medical exemptions (%)"),
        med_exempt_pct=number("Medical exemptions (%)"),
        MMR_cov=number("MMR vaccine coverage"),
        all_vacc_cov=number("Up-to-date vaccination (all vaccines)"),
        # Alaska's economic regions have a leading zero here that the
        # spreadsheet's codes (which the Alaska mapping uses) do not
        FIPS=pl.when(
            (pl.col("State") == "Alaska")
            & (pl.col("FIPS").str.len_chars() == 6)
            & pl.col("FIPS").str.starts_with("0")
        )
        .then(pl.col("FIPS").str.slice(1))
        .otherwise("FIPS"),
        county_type=pl.when(is_state)
        .then(pl.lit("State"))
        .otherwise("county_type"),
        state_ap=pl.col("state_ap"),
    )


# The data owner's Tennessee file (the sheet "data", years 2017-2025). Only its
# 2025 rows (school year 2025-26) are used: the spreadsheet has 2010-2024 (the
# file has the same values for 2017-2024, but Knox 2020 has its exemptions
# shifted by a column in the file)
VACCINATION_TENNESSEE_FILE = "Tennessee_Nov9_5pm.xlsx"


def tennessee_2025_rows(raw, spreadsheet):
    """
    The 2025 rows of `raw` (the sheet "data" of VACCINATION_TENNESSEE_FILE,
    every column text) in the columns of `read_2025_vaccination_rows`. The
    file has no FIPS: each county takes the name spelling, FIPS, county_type
    and state_ap of its rows in `spreadsheet` (the compiled spreadsheet as
    read), matched ignoring case (the file has DeKalb, McMinn and McNairy
    where the spreadsheet has Dekalb, Mcminn and Mcnairy). A county without a
    value stays missing.
    """

    def number(column):
        return pl.col(column).cast(pl.Float64, strict=False)

    def lowercase_letters(column):
        return pl.col(column).str.to_lowercase().str.replace_all("[^a-z]", "")

    rows = raw.filter(number("year") == 2025).select(
        key=lowercase_letters("county"),
        nonmed_exempt_pct=number("nme_pct"),
        med_exempt_pct=number("me_pct"),
        MMR_cov=number("mmr_pct"),
        all_vacc_cov=number("utd_pct"),
    )
    counties = (
        spreadsheet.filter(
            (pl.col("state") == "Tennessee") & pl.col("county").is_not_null()
        )
        .with_columns(key=lowercase_letters("county"))
        # the first value of each that is not missing (the early years of
        # the spreadsheet have no FIPS)
        .group_by("key", maintain_order=True)
        .agg(
            pl.col("county", "FIPS", "county_type", "state_ap")
            .drop_nulls()
            .first()
        )
    )
    # a county that is not in the spreadsheet would have no FIPS
    unmatched = rows.join(counties, on="key", how="anti")["key"].to_list()
    if unmatched or rows.height != rows["key"].n_unique() or not rows.height:
        raise ValueError(
            f"{VACCINATION_TENNESSEE_FILE}: {rows.height} rows for 2025, "
            f"counties not in the spreadsheet: {unmatched}"
        )
    return rows.join(
        counties, on="key", how="left", maintain_order="left"
    ).select(
        year=pl.lit(2025),
        state=pl.lit("Tennessee"),
        county="county",
        nonmed_exempt_pct="nonmed_exempt_pct",
        med_exempt_pct="med_exempt_pct",
        MMR_cov="MMR_cov",
        all_vacc_cov="all_vacc_cov",
        FIPS="FIPS",
        county_type="county_type",
        state_ap="state_ap",
    )


def add_tennessee_2025_rows(df, data_dir):
    """
    `df`, the compiled spreadsheet with its 2025 rows (add_2025_vaccination_rows),
    with Tennessee's 2025 rows of VACCINATION_TENNESSEE_FILE added, in the
    spreadsheet's column types. Tennessee must have no 2025 rows yet (two
    sources would duplicate its counties).
    """
    assert (
        df.filter(
            (pl.col("state") == "Tennessee") & (pl.col("year") == 2025)
        ).height
        == 0
    ), "Tennessee already has 2025 rows"
    raw = pl.read_excel(
        os.path.join(data_dir, VACCINATION_TENNESSEE_FILE),
        sheet_name="data",
        infer_schema_length=0,
    )
    new = tennessee_2025_rows(raw, df)
    new = new.select(
        [pl.col(c).cast(df.schema[c], strict=False) for c in new.columns]
    )
    return pl.concat([df, new], how="diagonal_relaxed")


def add_2025_vaccination_rows(df, data_dir):
    """
    `df`, the compiled spreadsheet as read (before any cleaning), with the
    2025 rows it lacks added from VACCINATION_2025_FILE:
      - New York is left out: its rows are replaced by New York.xlsx
        (replace_state_rows), which has 2025;
      - a state the spreadsheet already has 2025 rows for (West Virginia, with
        exemptions but no MMR) keeps them, and only the file's state row is
        added: its MMR coverage reaches the counties through the "State" row
        fill, as the state's earlier years do (it is not a county's own value);
      - every other state's 2025 rows are added.
    The new rows get the spreadsheet's column types.
    """
    new = read_2025_vaccination_rows(data_dir).filter(
        pl.col("state") != "New York"
    )
    new = new.select(
        [pl.col(c).cast(df.schema[c], strict=False) for c in new.columns]
    )
    held = df.filter(pl.col("year") == 2025)["state"].unique()
    return pl.concat(
        [
            df,
            new.filter(
                ~pl.col("state").is_in(held.implode())
                | (pl.col("county") == "State")
            ),
        ],
        how="diagonal_relaxed",
    )


# The data owner's file of 9 Nov 2026 (the same layout as VACCINATION_2025_FILE)
VACCINATION_NOV9_FILE = "vaxexempt2026 - forprod_Nov9_2026.csv"


def nov9_county_rows(data_dir, state):
    """
    The county rows of `state` of VACCINATION_NOV9_FILE: year (as the file
    labels it), state, county and the four values as numbers (nonmed, med,
    mmr, allv).
    """
    raw = pl.read_csv(
        os.path.join(data_dir, VACCINATION_NOV9_FILE),
        schema_overrides={"FIPS": pl.String},
        infer_schema_length=None,
        glob=False,
    )

    def number(column):
        return pl.col(column).cast(pl.Float64, strict=False)

    return raw.filter(
        (pl.col("State") == state)
        & pl.col("County").is_not_null()
        & (pl.col("County") != "State")
        & (pl.col("county_type").fill_null("") != "State")
    ).select(
        year=pl.col("Year"),
        state=pl.col("State"),
        county=pl.col("County"),
        nonmed=number("Non-medical exemptions (%)"),
        med=number("Medical exemptions (%)"),
        mmr=number("MMR vaccine coverage"),
        allv=number("Up-to-date vaccination (all vaccines)"),
    )


def fix_wyoming_year_labels(
    df, data_dir, mmr_column="mmr_cov", fips_column="fips"
):
    """
    Wyoming's data are labelled by the year they were recorded, one after the
    school year they describe (a school year is named by the year it starts
    in, as everywhere here):
      - the compiled spreadsheet's nonmedical exemption of Wyoming moves one
        year earlier (its value labelled 2017 is that of school year 2016);
        a year with no value after the move is blank;
      - the MMR coverage of the data owner's file of 9 Nov 2026 is labelled
        the same way: its value labelled Y + 1 is that of school year Y. It
        fills the Wyoming MMR the spreadsheet lacks (school years 2022 and
        2023); the spreadsheet's own values (labelled by the school year)
        are kept.
    """
    wyoming = pl.col("state") == "Wyoming"
    shifted = df.filter(wyoming).select(
        "state",
        "county",
        (pl.col("year") - 1).alias("year"),
        pl.col("nonmed_exempt_pct")
        .cast(pl.Float64, strict=False)
        .alias("_nonmed_shifted"),
    )
    df = df.join(
        shifted,
        on=["state", "county", "year"],
        how="left",
        maintain_order="left",
    ).with_columns(
        nonmed_exempt_pct=pl.when(wyoming)
        .then(pl.col("_nonmed_shifted"))
        .otherwise(pl.col("nonmed_exempt_pct").cast(pl.Float64, strict=False))
    )
    nov9 = nov9_county_rows(data_dir, "Wyoming").select(
        "state",
        "county",
        (pl.col("year") - 1).alias("year"),
        pl.col("mmr").alias("_mmr_nov9"),
    )
    out = (
        df.drop("_nonmed_shifted")
        .join(
            nov9,
            on=["state", "county", "year"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.coalesce(
                pl.col(mmr_column).cast(pl.Float64, strict=False),
                pl.col("_mmr_nov9"),
            ).alias(mmr_column)
        )
        .drop("_mmr_nov9")
    )
    if fips_column in out.columns:
        # the spreadsheet's blank rows (2022-2024) have no fips: a row with a
        # value needs its county's
        out = out.with_columns(
            pl.when(wyoming & pl.col(fips_column).is_null())
            .then(
                pl.col(fips_column)
                .drop_nulls()
                .first()
                .over("state", "county")
            )
            .otherwise(pl.col(fips_column))
            .alias(fips_column)
        )
    return out


def fill_hawaii_2024_exemptions(df, data_dir):
    """
    Hawaii's county nonmedical and medical exemptions of 2024, from the data
    owner's file of 9 Nov 2026 (the compiled spreadsheet has the 2024
    coverage but no 2024 exemptions). Only blanks are filled.
    """
    nov9 = (
        nov9_county_rows(data_dir, "Hawaii")
        .filter(pl.col("year") == 2024)
        .select(
            "state",
            "county",
            "year",
            pl.col("nonmed").alias("_nonmed_nov9"),
            pl.col("med").alias("_med_nov9"),
        )
    )
    return (
        df.join(
            nov9,
            on=["state", "county", "year"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.coalesce(
                pl.col("nonmed_exempt_pct").cast(pl.Float64, strict=False),
                pl.col("_nonmed_nov9"),
            ).alias("nonmed_exempt_pct"),
            pl.coalesce(
                pl.col("med_exempt_pct").cast(pl.Float64, strict=False),
                pl.col("_med_nov9"),
            ).alias("med_exempt_pct"),
        )
        .drop("_nonmed_nov9", "_med_nov9")
    )


# Cells of the compiled spreadsheet that the data owner's file of 9 Nov 2026
# has right (checked against the raw state files): (state, counties or None
# for every county of the state, year, spreadsheet columns).
#  - Virginia 2024 MMR: the spreadsheet's values sit one county row too high in
#    three stretches (59 counties; Nov 9 equals the raw data in 125 of 126);
#  - Florida: the values of Santa Rosa, Sarasota, Seminole, St. Johns and
#    St. Lucie are in the wrong county order in the spreadsheet (2021
#    exemptions, 2020 all-vaccines coverage);
#  - Washington 2024 exemptions: the spreadsheet has the exemptions for any
#    vaccine, Nov 9 the measles-specific ones of the other years;
#  - Michigan Oakland 2024 MMR: blank in the spreadsheet.
_FLORIDA_SWAPPED = [
    "Santa Rosa",
    "Sarasota",
    "Seminole",
    "St. Johns",
    "St. Lucie",
]
NOV9_REPLACEMENTS = [
    ("Virginia", None, 2024, ["MMR_cov"]),
    (
        "Florida",
        _FLORIDA_SWAPPED,
        2021,
        ["nonmed_exempt_pct", "med_exempt_pct"],
    ),
    ("Florida", _FLORIDA_SWAPPED, 2020, ["all_vacc_cov"]),
    ("Washington", None, 2024, ["nonmed_exempt_pct", "med_exempt_pct"]),
    ("Michigan", ["Oakland"], 2024, ["MMR_cov"]),
]
_NOV9_COLUMN = {
    "MMR_cov": "mmr",
    "nonmed_exempt_pct": "nonmed",
    "med_exempt_pct": "med",
    "all_vacc_cov": "allv",
}

# Values set by hand: (state, county, year, spreadsheet column, value).
# Washington 2024, measles-specific exemptions of the raw county counts
# (measles exempt - measles medical, and measles medical, over the
# kindergarten enrolment): Nov 9 has them slightly off for these three.
# South Dakota Lyman 2017: the raw data say 102% MMR coverage, with no
# exemptions: capped at 100.
KNOWN_VALUES = [
    (
        "Washington",
        "King",
        2024,
        "nonmed_exempt_pct",
        (553 - 39) / 21712 * 100,
    ),
    ("Washington", "King", 2024, "med_exempt_pct", 39 / 21712 * 100),
    (
        "Washington",
        "Pierce",
        2024,
        "nonmed_exempt_pct",
        (383 - 17) / 8458 * 100,
    ),
    ("Washington", "Pierce", 2024, "med_exempt_pct", 17 / 8458 * 100),
    (
        "Washington",
        "Snohomish",
        2024,
        "nonmed_exempt_pct",
        (396 - 26) / 8813 * 100,
    ),
    ("Washington", "Snohomish", 2024, "med_exempt_pct", 26 / 8813 * 100),
    ("South Dakota", "Lyman", 2017, "MMR_cov", 100.0),
]


def fix_known_values(
    df,
    data_dir,
    replacements=NOV9_REPLACEMENTS,
    known_values=KNOWN_VALUES,
):
    """
    `df`, the compiled spreadsheet (its own column names, after the 2025 rows
    are added), with the cells of NOV9_REPLACEMENTS taken from the data owner's
    file of 9 Nov 2026 (for the counties the file has, blank or not) and those
    of KNOWN_VALUES set. A county named in either list that is not in the spreadsheet is an
    error: a misspelled county would silently fix nothing. Run it before
    apply_qc_exclusions, so that an excluded county-year stays excluded.
    """

    def key(column):
        return (
            pl.col(column).str.to_lowercase().str.replace_all("[^a-z0-9]", "")
        )

    height = df.height
    df = df.with_columns(_key=key("county"))
    for state, counties, year, columns in replacements:
        nov9 = (
            nov9_county_rows(data_dir, state)
            .filter(pl.col("year") == year)
            .with_columns(_key=key("county"))
            .unique("_key", keep="first", maintain_order=True)
        )
        in_rows = (pl.col("state") == state) & (pl.col("year") == year)
        wanted = None if counties is None else {_name_key(c) for c in counties}
        if wanted is not None:
            missing = wanted - set(nov9["_key"])
            missing |= wanted - set(df.filter(in_rows)["_key"])
            if missing:
                raise ValueError(
                    f"{state} {year}: counties not in the spreadsheet or in "
                    f"the file of 9 Nov 2026: {sorted(missing)}"
                )
            nov9 = nov9.filter(pl.col("_key").is_in(sorted(wanted)))
        elif nov9.height == 0 or df.filter(in_rows).height == 0:
            raise ValueError(f"{state} {year}: no rows to fix")
        renamed = {_NOV9_COLUMN[c]: f"_{c}_nov9" for c in columns}
        df = df.join(
            nov9.select(
                pl.lit(state).alias("state"),
                pl.lit(year, dtype=pl.Int64).alias("year"),
                "_key",
                pl.lit(True).alias("_in_nov9"),
                *[pl.col(old).alias(new) for old, new in renamed.items()],
            ),
            on=["state", "year", "_key"],
            how="left",
            maintain_order="left",
        )
        # the file's value, blank or not, for every county it has: Virginia's
        # Lunenburg has no 2024 MMR in the file, and the spreadsheet's value
        # there is its neighbour's (one row off)
        df = df.with_columns(
            pl.when(in_rows & pl.col("_in_nov9").fill_null(False))
            .then(pl.col(f"_{c}_nov9").cast(df.schema[c]))
            .otherwise(pl.col(c))
            .alias(c)
            for c in columns
        ).drop(["_in_nov9", *[f"_{c}_nov9" for c in columns]])
    for state, county, year, column, value in known_values:
        is_row = (
            (pl.col("state") == state)
            & (pl.col("year") == year)
            & (pl.col("_key") == _name_key(county))
        )
        if df.filter(is_row).height == 0:
            raise ValueError(f"known value matches no row: {state} {county}")
        df = df.with_columns(
            pl.when(is_row)
            .then(pl.lit(value).cast(df.schema[column]))
            .otherwise(pl.col(column))
            .alias(column)
        )
    assert df.height == height, "a fix duplicated or lost rows"
    return df.drop("_key")


def add_montana_data(df, data_dir, mmr_column="mmr_cov"):
    """
    Montana.xlsx (read_montana_data: the 2019 coverage of 50 counties) given
    to `df`: only the values `df` lacks are filled, as the burden figure does
    for its own data.
    """
    montana = read_montana_data(data_dir).rename({"MMR_cov": mmr_column})
    montana = montana.drop("estimated_mmr_cov")
    df = df.join(
        montana,
        on=["state", "county", "year"],
        how="left",
        suffix="_montana",
        maintain_order="left",
    )
    is_montana = pl.col("state") == "Montana"
    columns = [
        "nonmed_exempt_pct",
        "med_exempt_pct",
        mmr_column,
        "all_vacc_cov",
    ]
    df = df.with_columns(
        [
            pl.when(
                is_montana & pl.col(c).cast(pl.Float64, strict=False).is_null()
            )
            .then(pl.col(f"{c}_montana"))
            .otherwise(pl.col(c).cast(pl.Float64, strict=False))
            .alias(c)
            for c in columns
        ]
    )
    return df.drop([f"{c}_montana" for c in columns])


def expand_pooled_rows(df, value_columns):
    """
    The rows that pool several counties (POOLED_ROWS) given to each member
    county, year by year: a member's own row keeps its values and takes the
    pooled row's where it has none; a member with no row of its own that year
    gets one (its name and fips, the pooled values). The pooled rows are
    dropped. `value_columns` are the columns that hold values (the others
    keep the member's own row, or the pooled row's for a new row).
    """
    for (state, name), members in POOLED_ROWS.items():
        is_pooled = (pl.col("state") == state) & (pl.col("county") == name)
        pooled = df.filter(is_pooled)
        if pooled.is_empty():
            continue
        member_fips = [fips for _, fips in members]
        is_member = (pl.col("state") == state) & pl.col("fips").is_in(
            member_fips
        )
        member_rows = pl.concat(
            [
                pooled.with_columns(
                    county=pl.lit(member_name), fips=pl.lit(member_fips_)
                )
                for member_name, member_fips_ in members
            ]
        )
        own = df.filter(is_member & ~is_pooled)
        # the member's own rows come first, so that their values win
        both = pl.concat([own, member_rows], how="diagonal_relaxed")
        columns = [
            c for c in both.columns if c not in ("state", "fips", "year")
        ]
        merged = both.group_by(
            ["state", "fips", "year"], maintain_order=True
        ).agg(
            [
                pl.col(c).drop_nulls().first()
                if c in value_columns
                else pl.col(c).first()
                for c in columns
            ]
        )
        df = pl.concat(
            [df.filter(~is_pooled & ~is_member), merged.select(df.columns)],
            how="diagonal_relaxed",
        )
    return df


# Mapping of 2-digit state FIPS codes to full state names for the target
# states: the 13 whose state values are in state_level_coverages.csv, then
# Hawaii and Mississippi, whose state values are the CDC's
_TARGET_STATE_FIPS_TO_NAME = {
    "41": "Oregon",
    "32": "Nevada",
    "49": "Utah",
    "30": "Montana",
    "31": "Nebraska",
    "48": "Texas",
    "22": "Louisiana",
    "46": "South Dakota",
    "38": "North Dakota",
    "51": "Virginia",
    "50": "Vermont",
    "02": "Alaska",
    "29": "Missouri",
    "15": "Hawaii",
    "28": "Mississippi",
}

_TARGET_STATES = set(_TARGET_STATE_FIPS_TO_NAME.values())


def cdc_state_level_data(state_level_file: str) -> pl.DataFrame:
    """
    The CDC's state-level kindergarten vaccination data, one row per state
    and school year (year is the first year of the school year): mmr_cov_state,
    med_exempt_pct_state, nonmed_exempt_pct_state and nonmed_source (which
    CDC series the nonmedical exemption comes from).

    For the nonmedical exemption the CDC's "Non-Medical Exemption" is used;
    where a state-year only has "Any Exemption" (medical and nonmedical
    together), Any minus the state's medical exemption (at least 0) is used
    when that exists, and Any itself otherwise.
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
    state_med_exempt = (
        state_data.filter(pl.col("Dose").str.starts_with("Medical"))
        .select(
            [
                pl.col("Geography").alias("state"),
                "year",
                pl.col("estimate_pct").alias("med_exempt_pct_state"),
            ]
        )
        .unique(["state", "year"], keep="first", maintain_order=True)
    )

    # Nonmedical exemptions - these are NOT vaccine-specific, so don't filter to MMR
    # The CDC file has "Non-Medical Exemption" and "Any Exemption" (medical and
    # nonmedical together). Use the Non-Medical rows; where a state-year only
    # has "Any Exemption", use Any minus the state's medical exemption when
    # that exists (so medical exemptions are not counted as nonmedical), and
    # otherwise Any itself
    nonmed_only = (
        state_data.filter(pl.col("Dose") == "Non-Medical Exemption")
        .select(
            pl.col("Geography").alias("state"),
            "year",
            pl.col("estimate_pct").alias("nonmed_only_pct"),
        )
        .unique(["state", "year"], keep="first", maintain_order=True)
    )
    any_exempt = (
        state_data.filter(pl.col("Dose") == "Any Exemption")
        .select(
            pl.col("Geography").alias("state"),
            "year",
            pl.col("estimate_pct").alias("any_pct"),
        )
        .unique(["state", "year"], keep="first", maintain_order=True)
    )
    state_nonmed_exempt = (
        nonmed_only.join(
            any_exempt,
            on=["state", "year"],
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )
        .join(
            state_med_exempt,
            on=["state", "year"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            nonmed_exempt_pct_state=pl.coalesce(
                "nonmed_only_pct",
                pl.when(pl.col("med_exempt_pct_state").is_not_null())
                .then(
                    pl.max_horizontal(
                        pl.col("any_pct") - pl.col("med_exempt_pct_state"),
                        pl.lit(0.0),
                    )
                )
                .otherwise(pl.col("any_pct")),
            ),
            nonmed_source=pl.when(pl.col("nonmed_only_pct").is_not_null())
            .then(pl.lit("Non-Medical Exemption"))
            .otherwise(pl.lit("Any Exemption")),
        )
        .filter(pl.col("nonmed_exempt_pct_state").is_not_null())
        .select("state", "year", "nonmed_exempt_pct_state", "nonmed_source")
    )

    # Merge all state-level data using full outer joins
    state_level_data = state_mmr_cov

    if state_med_exempt.height > 0:
        state_level_data = state_level_data.join(
            state_med_exempt,
            on=["state", "year"],
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )

    if state_nonmed_exempt.height > 0:
        state_level_data = state_level_data.join(
            state_nonmed_exempt,
            on=["state", "year"],
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )
    return state_level_data


def state_level_series(data_dir: str) -> pl.DataFrame:
    """
    State values by school year from the CDC (cdc_state_level_data, every
    state), with series_source "cdc". The values the CDC file lacks for a
    school year (Montana's series ends in 2020) are not made up here: the
    imputation takes the state's nearest year, and the baselines of the
    projections trend the state's last value.
    """
    cdc = cdc_state_level_data(
        os.path.join(
            data_dir,
            "Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260526.csv",
        )
    )
    return cdc.with_columns(series_source=pl.lit("cdc"))


def flag_imputed_rows(
    imputed: pl.DataFrame, original: pl.DataFrame
) -> pl.DataFrame:
    """
    Add the boolean column _is_imputed to the result of
    impute_from_state_level_coverages: True for a county-year that has no row
    in `original`, and for a row whose coverage, NME rate or ME rate is blank
    in `original` but has a state value now.
    """
    columns = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
    # as in the imputation, the first row of a county-year is the real one
    reported = (
        original.filter(pl.col("fips").is_not_null())
        .unique(["fips", "year"], keep="first", maintain_order=True)
        .select(
            "fips",
            "year",
            *(pl.col(c).is_not_null().alias(f"{c}_reported") for c in columns),
        )
    )
    return (
        imputed.with_row_index("_row")
        .join(reported, on=["fips", "year"], how="left")
        .sort("_row")
        .with_columns(
            # a null-FIPS row (a region, a district) is never imputed
            _is_imputed=pl.col("fips").is_not_null()
            & pl.any_horizontal(
                pl.col(c).is_not_null()
                & ~pl.col(f"{c}_reported").fill_null(False)
                for c in columns
            )
        )
        .drop("_row", *(f"{c}_reported" for c in columns))
    )


def impute_from_state_level_coverages(
    vaccination_data: pl.DataFrame,
    data_dir: str,
) -> pl.DataFrame:
    """
    For the target states, ensure every county-level FIPS code has data for
    ALL years present in the dataset. Fills missing counties and years with
    the CDC's state-level values for the year, else those of the state's
    nearest year (the earlier one at a tie). A CDC field that is missing in a
    year is filled from the nearest year that has it. The CDC's state series
    is the only source of state values (the earlier curated file of 13 states
    is no longer used).

    This operates on and returns a COPY of vaccination_data. The original is
    not modified.
    """
    vaccination_data = vaccination_data.clone()

    # 1. State-level fallback values
    # Create lookup: {(state, year): {coverage, nme, me}}
    state_lookup: dict[tuple[str, int], dict] = {}

    # the CDC's values of every year it has, a field that is missing in a year
    # filled from the nearest year that has it
    cdc_fields = {
        "mmr_cov_state": "estimated_mmr_cov",
        "nonmed_exempt_pct_state": "nonmed_exempt_pct",
        "med_exempt_pct_state": "med_exempt_pct",
    }
    cdc_states = (
        state_level_series(data_dir)
        .filter(
            (pl.col("series_source") == "cdc")
            & pl.col("state").is_in(list(_TARGET_STATES))
        )
        .sort("state", "year", maintain_order=True)
        .with_columns(
            pl.col(field)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .over("state")
            for field in cdc_fields
        )
    )
    for row in cdc_states.iter_rows(named=True):
        state_lookup[(row["state"], row["year"])] = {
            key: float(row[field]) for field, key in cdc_fields.items()
        }

    # a year the state has no values for takes those of its nearest year (the
    # earlier one at a tie): the first year's for the years before it, the
    # last year's for the years after it
    state_years: dict[str, list[int]] = {}
    for state_name, year in state_lookup:
        state_years.setdefault(state_name, []).append(year)

    def state_values(state: str, year: int) -> dict:
        years = state_years[state]
        nearest = min(years, key=lambda y: (abs(y - year), y))
        return state_lookup[(state, nearest)]

    # ------------------------------------------------------------------
    # 2. Load shapefile to get ALL county FIPS for target states
    # ------------------------------------------------------------------
    shapefile_path = os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    counties_gdf = gpd.read_file(shapefile_path)

    target_statefps = set(_TARGET_STATE_FIPS_TO_NAME.keys())
    target_counties = counties_gdf[
        counties_gdf["STATEFP"].isin(target_statefps)  # type: ignore
    ][["STATEFP", "GEOID", "NAME"]].copy()
    target_counties["state"] = target_counties["STATEFP"].map(  # type: ignore
        _TARGET_STATE_FIPS_TO_NAME  # type: ignore
    )

    # ------------------------------------------------------------------
    # 3. Get all years in the dataset
    # ------------------------------------------------------------------
    all_years = sorted(
        vaccination_data.filter(pl.col("year").is_not_null())["year"]
        .unique()
        .to_list()
    )
    print(f"Years in vaccination_data: {min(all_years)} to {max(all_years)}")

    # ------------------------------------------------------------------
    # 4. Create complete skeleton: all target counties × all years
    #    Using year-specific state values when available
    # ------------------------------------------------------------------
    complete_rows = []
    for _, row in target_counties.iterrows():  # type: ignore
        fips = row["GEOID"]
        state_name = row["state"]
        county_name = row["NAME"]

        if state_name not in state_years:
            print(
                f"WARNING: No state-level coverage for '{state_name}' - skipping"
            )
            continue

        for year in all_years:
            vals = state_values(state_name, year)  # type: ignore

            complete_rows.append(
                {
                    "fips": fips,
                    "state": state_name,
                    "county": county_name,
                    "year": year,
                    "estimated_mmr_cov": vals["estimated_mmr_cov"],
                    "nonmed_exempt_pct": vals["nonmed_exempt_pct"],
                    "med_exempt_pct": vals["med_exempt_pct"],
                    "_is_imputed": True,
                }
            )

    print(
        f"Created {len(complete_rows)} skeleton rows for "
        f"{len(target_counties)} counties × {len(all_years)} years"
    )

    skeleton_df = pl.DataFrame(
        complete_rows,
        schema_overrides={
            "fips": pl.String,
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "_is_imputed": pl.Boolean,
        },
    )

    # ------------------------------------------------------------------
    # 5. Mark existing data and combine
    # ------------------------------------------------------------------
    vaccination_data = vaccination_data.with_columns(_is_imputed=pl.lit(False))

    combined = pl.concat(
        [vaccination_data, skeleton_df], how="diagonal_relaxed"
    )

    # ------------------------------------------------------------------
    # 6. Prioritize real data over imputed: group by fips+year, take first non-imputed
    # ------------------------------------------------------------------
    # rows with a null FIPS (regions, districts, ...) all share one group key,
    # which would collapse them to one row per year, so only deduplicate the
    # rows that have a FIPS and keep the null-FIPS rows as they are
    null_fips_rows = combined.filter(pl.col("fips").is_null())
    combined = (
        combined.filter(pl.col("fips").is_not_null())
        .sort("_is_imputed", maintain_order=True)  # False (real) comes first
        .group_by(["fips", "year"], maintain_order=True)
        .agg(pl.all().first())  # Takes first row = real data preferred
    )
    combined = pl.concat(
        [combined, null_fips_rows.select(combined.columns)],
        how="vertical_relaxed",
    )

    # ------------------------------------------------------------------
    # 7. Fill any remaining nulls with state-level values (year-specific)
    # ------------------------------------------------------------------
    def get_state_value(state: str, year: int, key: str) -> float:
        """The state-level value of a year, or of the state's nearest year"""
        if state in state_years:
            return state_values(state, year)[key]  # type: ignore
        return None  # type: ignore

    for col, lookup_key in [
        ("estimated_mmr_cov", "estimated_mmr_cov"),
        ("nonmed_exempt_pct", "nonmed_exempt_pct"),
        ("med_exempt_pct", "med_exempt_pct"),
    ]:
        combined = combined.with_columns(
            pl.when(pl.col(col).is_null())
            .then(
                pl.struct(["state", "year"]).map_elements(
                    lambda x: get_state_value(
                        x["state"], x["year"], lookup_key
                    ),
                    return_dtype=pl.Float64,
                )
            )
            .otherwise(pl.col(col))
            .alias(col)
        )

    combined = combined.drop("_is_imputed")

    # ------------------------------------------------------------------
    # 8. Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IMPUTATION SUMMARY BY STATE")
    print("=" * 80)
    for state_name in sorted(_TARGET_STATES):
        state_rows = combined.filter(
            (pl.col("state") == state_name)
            & (pl.col("fips").str.len_chars() == 5)
        )
        n_fips = state_rows["fips"].n_unique()
        n_nulls = state_rows.filter(
            pl.col("estimated_mmr_cov").is_null()
            | pl.col("nonmed_exempt_pct").is_null()
            | pl.col("med_exempt_pct").is_null()
        ).height

        # Show year coverage
        years_with_data = sorted(state_rows["year"].unique().to_list())
        year_range = (
            f"{min(years_with_data)}-{max(years_with_data)}"
            if years_with_data
            else "none"
        )

        print(
            f"  {state_name:20s}: {n_fips:3d} counties, "
            f"years {year_range}, {n_nulls:4d} rows with remaining nulls"
        )
    print("=" * 80 + "\n")

    return combined


def query_avg_vaccinations_region_years(
    vaccinations_df: pl.DataFrame,
    years: list[int],
    fips: list[str],
    population_query=None,
) -> dict:
    """
    The population-weighted average coverage and exemptions of the counties
    `fips` over `years`, as {field: [average]} (None for a field that no
    county reports, and for every field when no county has data).

    population_query(years, fips) returns a frame with the 5-9 population
    (pop_5_9) of each county-year, in order; by default the census is asked.
    """
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
    if population_query is None:
        # imported here: utils needs the census package
        from outbreak_size import utils

        # with the cache of the Census answers and its requests in parallel:
        # script 00 calls this once per row for the five school years of
        # --five-year, so the same county-year is asked for many times
        population_query = utils.CensusQuery(
            os.getenv("CENSUS_API_KEY"),  # type: ignore
            cache_path=utils.CENSUS_CACHE_PATH,
            workers=utils.CENSUS_WORKERS,
        ).query_5_9_pop
    pop_df = population_query(
        temp_df["year"].to_list(), temp_df["fips"].to_list()
    )
    temp_df = temp_df.with_columns(pop_5=pop_df["pop_5_9"] / 5)

    def weighted_average(field: str):
        """
        Population-weighted average of `field` over the rows that report it.

        Rows with a null value are left out of both the numerator and the
        weights, so a county with no data does not pull the average toward
        zero. The average is taken within each year first and then across
        years, weighting each year by its population. None if no row reports
        the field (or the rows that do have no population).
        """
        rows = temp_df.filter(pl.col(field).is_not_null())
        if rows.height == 0:
            return None
        by_year = rows.group_by("year", maintain_order=True).agg(
            value=(pl.col(field) * pl.col("pop_5")).sum()
            / pl.col("pop_5").sum(),
            pop_5=pl.col("pop_5").sum(),
        )
        total_pop = by_year["pop_5"].sum()
        # no population to weight by: no average
        if not total_pop:
            return None
        return (by_year["value"] * by_year["pop_5"]).sum() / total_pop

    return {
        field: [weighted_average(field)]
        for field in [
            "estimated_mmr_cov",
            "nonmed_exempt_pct",
            "med_exempt_pct",
        ]
    }


def process_vaccinations_df(
    vaccinations_df: pl.DataFrame,
    regions=None,
    region_averages=None,
    case_places=None,
) -> pl.DataFrame:
    """
    Clean the compiled vaccination data and add the multi-county places.

    Coverage falls back to the all-vaccine coverage where the MMR coverage is
    missing or 0; county names are spelled as in the case data; the District of
    Columbia and the old Nebraska codes are mapped to county codes; and each
    region (all but the Gulf Coast) gets, for every year, a row with the
    averages of its counties.

    regions is a {name: county fips} dict (by default the regions of
    outbreak_size.regions) and region_averages(vaccinations_df, years, fips)
    returns a region's averages (by default
    query_avg_vaccinations_region_years).

    case_places (the columns year, state, county and fips of the case data) says
    which counties have a case row of their own: such a county is not among
    the people of a region that year (see regions.members_with_own_row), so the
    region's row of school year y, which the cases of year y + 1 take (see
    analysis_dataset.merge_outbreaks_vaccinations), averages the members that
    have no case row of their own in y + 1. Without case_places every member
    is averaged.
    """
    # when we don't have mmr coverage, use all coverage
    vaccinations_df = vaccinations_df.with_columns(
        estimated_mmr_cov=pl.when(
            pl.col("mmr_cov").is_null() | (pl.col("mmr_cov") == 0.0)
        )
        .then(pl.col("all_vacc_cov"))
        .otherwise(pl.col("mmr_cov")),
        # which of the two the coverage above is (null when it is missing)
        coverage_source=pl.when(
            pl.col("mmr_cov").is_null() | (pl.col("mmr_cov") == 0.0)
        )
        .then(
            pl.when(pl.col("all_vacc_cov").is_not_null()).then(
                pl.lit("all_vaccine")
            )
        )
        .otherwise(pl.lit("mmr")),
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

    # the rows that pool counties (Issaquena/Sharkey) go to their members
    vaccinations_df = expand_pooled_rows(
        vaccinations_df,
        [
            c
            for c in (
                "nonmed_exempt_pct",
                "med_exempt_pct",
                "mmr_cov",
                "all_vacc_cov",
                "estimated_mmr_cov",
                "coverage_source",
            )
            if c in vaccinations_df.columns
        ],
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

    # the counties' names, in the order of new_nebraska (from the 2020
    # county shapefile), so that a district's rows are named for the counties
    # the case data name
    nebraska_fips["new_nebraska_county"] = [
        "Hall",
        "Hamilton",
        "Merrick",
        "Dakota",
        "Douglas",
        "Boone",
        "Colfax",
        "Nance",
        "Platte",
        "Burt",
        "Cuming",
        "Madison",
        "Stanton",
        "Butler",
        "Polk",
        "Seward",
        "York",
        "Lancaster",
        "Blaine",
        "Custer",
        "Garfield",
        "Greeley",
        "Howard",
        "Loup",
        "Sherman",
        "Valley",
        "Wheeler",
        "Antelope",
        "Boyd",
        "Brown",
        "Cherry",
        "Holt",
        "Keya Paha",
        "Knox",
        "Pierce",
        "Rock",
        "Cedar",
        "Dixon",
        "Thurston",
        "Wayne",
        "Banner",
        "Box Butte",
        "Cheyenne",
        "Dawes",
        "Deuel",
        "Garden",
        "Grant",
        "Sioux",
        "Scotts Bluff",
        "Morrill",
        "Sheridan",
        "Kimball",
        "Fillmore",
        "Gage",
        "Jefferson",
        "Saline",
        "Thayer",
        "Sarpy",
        "Cass",
        "Adams",
        "Clay",
        "Nuckolls",
        "Webster",
        "Johnson",
        "Nemaha",
        "Otoe",
        "Pawnee",
        "Richardson",
        "Chase",
        "Dundy",
        "Frontier",
        "Furnas",
        "Hayes",
        "Hitchcock",
        "Keith",
        "Perkins",
        "Red Willow",
        "Dodge",
        "Washington",
        "Saunders",
        "Buffalo",
        "Dawson",
        "Franklin",
        "Gosper",
        "Harlan",
        "Kearney",
        "Phelps",
        "Arthur",
        "Hooker",
        "Lincoln",
        "Logan",
        "McPherson",
        "Thomas",
    ]
    nebraska_fips = pl.DataFrame(nebraska_fips)
    vaccinations_df = vaccinations_df.join(
        nebraska_fips,
        left_on="fips",
        right_on="old_nebraska",
        how="left",
        maintain_order="left_right",
    )
    # a district row becomes one row per county of the district, with the
    # county's fips (new_nebraska keeps the county's code as well)
    vaccinations_df = vaccinations_df.with_columns(
        fips=pl.when(pl.col("new_nebraska").is_not_null())
        .then(pl.col("new_nebraska"))
        .otherwise(pl.col("fips")),
        county=pl.when(pl.col("new_nebraska").is_not_null())
        .then(pl.col("new_nebraska_county"))
        .otherwise(pl.col("county")),
    ).drop("new_nebraska_county")
    vaccinations_df = vaccinations_df.with_columns(
        fips=pl.when(pl.col("fips") == "11")
        .then(pl.lit("11001"))
        .otherwise(pl.col("fips"))
    )

    assignment_special_places = (
        higher_level_regions() if regions is None else regions
    )
    if region_averages is None:
        region_averages = query_avg_vaccinations_region_years
    own_rows = (
        None
        if case_places is None
        else members_with_own_row(
            case_places, assignment_special_places, vaccinations_df
        )
    )

    for place, fips_list in assignment_special_places.items():
        if place == "Gulf Coast":
            continue

        state = vaccinations_df.filter(pl.col("fips") == fips_list[0])[
            "state"
        ][0]

        for year in (
            vaccinations_df["year"].unique(maintain_order=True).to_list()
        ):
            # the counties this row stands for: the row of school year
            # `year` is the one the cases of the next year take
            members = (
                fips_list
                if own_rows is None
                else members_without_own_row(
                    assignment_special_places,
                    own_rows,
                    place,
                    outbreak_year_of_county_data(year),
                )
            )
            temp = region_averages(vaccinations_df, [year], members)

            # the region's coverage averages its counties' coverage, so it is
            # of one source only if all of them are; "mixed" otherwise
            county_sources = vaccinations_df.filter(
                (pl.col("year") == year)
                & pl.col("fips").is_in(members)
                & pl.col("coverage_source").is_not_null()
            )["coverage_source"].unique()
            region_source = (
                county_sources[0]
                if county_sources.len() == 1
                else ("mixed" if county_sources.len() > 1 else None)
            )

            temp_df = pl.DataFrame(temp)
            temp_df = temp_df.with_columns(
                coverage_source=pl.lit(region_source, dtype=pl.String),
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


def impute_from_bordering_years(
    missing_df: pl.DataFrame,
    vaccinations_df: pl.DataFrame,
    state_level_file: str,
    regions=None,
) -> pl.DataFrame:
    """
    Impute missing vaccination data with the following priority:
    1. Same county, outbreak year (year+0)
    2. Same county, outbreak year-2
    3. State-level CDC data for outbreak year-1

    For the state-level nonmedical exemption the CDC's "Non-Medical Exemption"
    is used; where a state-year only has "Any Exemption" (medical and
    nonmedical together), Any minus the state's medical exemption (at least
    0) is used when the medical exemption exists, and Any itself (with
    medical set to 0) otherwise.

    Only imputes for regular counties, not special regions (the names of
    `regions`, by default those of outbreak_size.regions).
    """
    state_level_data = cdc_state_level_data(state_level_file)

    # Filter out special regions
    special_places = list(
        (higher_level_regions() if regions is None else regions).keys()
    )
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

                    # If we had to use the CDC's "Any Exemption" for nonmedical
                    # (no Non-Medical row) and don't have medical exemption
                    # data from state, impute medical as 0
                    if (
                        needs_med
                        and state_row.get("med_exempt_pct_state") is None
                        and state_row.get("nonmed_source") == "Any Exemption"
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
