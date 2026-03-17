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
    # make them normal case for merging purposes
    montana = montana.with_columns(county=pl.col("county").str.to_titlecase())

    # we notice that we only have coverage data for 2019, and we only have exemption
    # data prior to that
    # so let's just fill in the missing exemption data for 2019 with the exemption
    # data for 2018

    # this is ok because we are only using this data for risk projections
    montana = montana.sort(["county", "year"]).with_columns(
        pl.col("nonmed_exempt_pct").forward_fill().over("county"),
        pl.col("med_exempt_pct").forward_fill().over("county"),
    )

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
