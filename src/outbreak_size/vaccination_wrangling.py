import glob
import os

import polars as pl
from addfips import AddFIPS


# kansas data is special because it's usually at the school district level,
# but we need it at the county level
# So we hda it queried separately at the county level
def read_kansas_data(data_dir):
    # get all MMR coverage data
    mmr_files = glob.glob(os.path.join(data_dir, "MMR", "**", "*.csv"))
    kansas_mmr_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in mmr_files
    ]
    kansas_mmr = pl.concat(kansas_mmr_dfs)
    # do the same for exemption data
    nme_files = glob.glob(
        os.path.join(data_dir, "Religious Exemption", "**", "*.csv")
    )
    kansas_nme_dfs = [
        pl.read_csv(file, separator="\t", encoding="utf16")
        for file in nme_files
    ]
    kansas_nme = pl.concat(kansas_nme_dfs)
    # do the same for medical exemption data
    me_files = glob.glob(
        os.path.join(data_dir, "Medical Exemption", "**", "*.csv")
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
        kansas_nme, on=["County", "year"], how="inner"
    ).join(kansas_me, on=["County", "year"], how="inner")

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
