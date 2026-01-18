import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


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
        "North, Alabama": [
            "01019",
            "01033",
            "01043",
            "01049",
            "01059",
            "01071",
            "01077",
            "01079",
            "01083",
            "01089",
            "01095",
            "01103",
            "01133",
        ],
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
        "Upstate, South Carolina": [
            "45045",
            "45083",
            "45007",
            "45077",
            "45073",
            "45047",
            "45059",
            "45021",
            "45087",
            "45001",
        ],
    }
    return assignment_special_places


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 7.5))
    if nrows * ncols == 1:
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
        return ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


# plot for the vaccination coverage in county-years with >= 10 cases
def plot_coverage_distribution(
    df,
    cases_threshold,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    # want to make violin plots by outbreak size category and then subfaceted by year
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 1, 4, cases_threshold, 1000000000],
            labels=["-1", "0", "1", "2", "3", "4"],
            left_closed=True,
        ),
    )
    df = df.group_by(["outbreak_size_buckets"]).agg(pl.col(vaccination_metric))
    df = df.sort(by=["outbreak_size_buckets"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25)
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75)
        ),
    )

    data_post = df.to_dict()[vaccination_metric].to_list()
    medians_post = df.to_dict()["median_cov"].to_numpy()
    lowers_post = df.to_dict()["lower_cov"].to_numpy()
    uppers_post = df.to_dict()["upper_cov"].to_numpy()
    means_post = df.to_dict()["avg_cov"].to_list()

    # boxplot and swarmplot of coverage distributions

    violinparts2 = ax.violinplot(
        data_post,
        positions=range(len(data_post)),
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor="green")

    ax.errorbar(
        x=range(len(medians_post)),
        y=medians_post,
        yerr=[medians_post - lowers_post, uppers_post - medians_post],
        fmt="o",
        capsize=5,
        color="green",
    )

    # add means as stars
    ax.scatter(
        range(len(means_post)),
        means_post,
        marker="*",
        s=90,
        color="green",
    )

    ax.set_xticks(range(len(data_post)))
    ax.set_xticklabels(
        [
            "0",
            "1-3",
            "4-9",
            f"≥{cases_threshold}",
        ]
    )

    ax.legend([violinparts2["bodies"][0]], ["2021-2025"], loc=(0.45, 0.02))
    ax.axhline(y=vaccination_threshold, color="k", linestyle="--")
    ax.set_ylabel("MMR vaccine coverage (%)")
    ax.set_xlabel("Outbreak size (cases)")

    return fig, ax


if __name__ == "__main__":
    fig, ax = setup_plot(1, 3)  # type: ignore

    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    mega_regions = df.filter(pl.col("mega_region_2025") == True)
    mega_regions_cleaned = mega_regions.select(
        ["county", "state", "cases_outbreak", "estimated_mmr_cov"]
    )
    mega_regions_cleaned = mega_regions_cleaned.with_columns(
        county=pl.col("county").str.split(", ").list.first()
    )
    # get the range of coverage in these mega regions
    special_places = higher_level_regions()
    vaccine_data = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={"fips": pl.String, "estimated_mmr_cov": pl.Float64},
    )

    mins = np.array([])
    maxs = np.array([])
    for region_name in mega_regions["county"].to_list():
        fips_list = special_places[region_name]
        region_counties = vaccine_data.filter(
            (pl.col("fips").is_in(fips_list)) & (pl.col("year") == 2024)
        )
        mins = np.append(
            mins,
            region_counties["estimated_mmr_cov"].min(),  # type: ignore
        )
        maxs = np.append(
            maxs,
            region_counties["estimated_mmr_cov"].max(),  # type: ignore
        )

    mega_regions_cleaned = mega_regions_cleaned.with_columns(
        minimum_coverage=pl.lit(mins),
        maximum_coverage=pl.lit(maxs),
    )

    plot_coverage_distribution(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])).filter(
            pl.col("mega_region_2025") == False
        ),
        cases_threshold=10,
        vaccination_threshold=95,
        fig=fig,
        ax=ax[1],  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_2025_mega_regions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_2025_mega_regions.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
