"""
The clusters of 2025 outbreaks that are left out of the share of cases in
multi-county outbreaks (panel d of figure_predicting_county_level_outbreaks.py)
and out of the fits of script 02 (its leave-one-cluster-out runs).

A cluster is a group of outbreak places (3 or more outbreak cases in the year)
joined by borders; it is left out when it holds one of the seeds that official
sources name as part of an outbreak. The places are found from the case data
of all the counties (outbreak_cases_all_counties), whether or not they have
vaccine data, so that the figure and the fits leave out the same places.
"""

import numpy as np
import polars as pl
from addfips import AddFIPS
from scipy.sparse.csgraph import connected_components

from outbreak_size import adjacency
from outbreak_size.analysis_dataset import assemble_analysis_dataset
from outbreak_size.regions import higher_level_regions

# the clusters, in the order of the bars of the figure, and the tag of each in
# the names of the files of the fits that leave it out
CLUSTER_NAMES = ("West Texas-NM", "Upstate SC", "Utah-Arizona")
CLUSTER_TAGS = {
    "West Texas-NM": "west_texas_nm",
    "Upstate SC": "upstate_sc",
    "Utah-Arizona": "utah_arizona",
}


def region_members():
    """
    The counties of each place that is several counties (a region or health
    district), by the name the place has in the case data.
    """
    members = higher_level_regions()
    # the commented list of "District 1" has 21075 and 21087, which are
    # Kentucky counties, in place of 22075 and 22087: with those corrected it
    # is the list of "Region 1, Louisiana"
    members["District 1"] = members["Region 1, Louisiana"]
    return members


def outbreak_cases_all_counties(raw_merged, regions=None):
    """
    The county-years of the case data with their outbreak cases (cases_outbreak,
    defined as in analysis_dataset.csv), whether or not the county has vaccine
    data, for the share of cases in multi-county outbreaks (panel c). A county
    whose fips is missing in the data (it did not match the vaccine data) gets
    its fips from its name. Places that are several counties (a region, a
    health district, a whole state) have no single county fips. They are left
    out of the share, both its numerator and its denominator, unless
    `regions` (place name -> its county fips, see region_members) says which
    counties a place is made of; the place is then kept, with its name in the
    fips column.
    """
    regions = {} if regions is None else regions
    fip_getter = AddFIPS()
    df = assemble_analysis_dataset(raw_merged).with_columns(
        fips=pl.struct("state", "county", "fips").map_elements(
            lambda row: (
                row["fips"]
                if row["fips"] is not None
                else fip_getter.get_county_fips(row["county"], row["state"])
            ),
            return_dtype=pl.String,
        )
    )
    is_county = pl.col("fips").str.len_chars() == 5
    is_region = pl.col("county").is_in(list(regions)) & ~is_county.fill_null(
        False
    )
    df = df.with_columns(
        fips=pl.when(is_region)
        .then(pl.col("county"))
        .otherwise(pl.col("fips"))
    )
    kept = is_county.fill_null(False) | is_region
    left_out = df.filter(~kept)
    print(
        f"{left_out.height} multi-county places left out "
        f"({left_out['cases_outbreak'].sum()} outbreak cases of "
        f"{df['cases_outbreak'].sum()})"
    )
    places = df.filter(kept).select("year", "fips", "cases_outbreak")
    assert places.select("year", "fips").is_duplicated().sum() == 0
    return places


def place_adjacency_matrix(adjacency_data, places, regions, steps=1):
    """
    1 where two of the places (county fips, or the name of a region in
    `regions`) are neighbors, as a matrix in the order of `places`. With
    steps=1 places are neighbors if they border each other: a region borders a
    place if any of its counties borders any of the place's counties. With
    steps=2 they are also neighbors if some county (of any place or none)
    borders both. Places that share a county are not neighbors.
    """
    assert steps in (1, 2)
    adjacency_data = adjacency.add_connecticut_counties(adjacency_data)
    neighbors = {}
    for county, neighbor in adjacency_data.select(
        "County GEOID", "Neighbor GEOID"
    ).iter_rows():
        neighbors.setdefault(county, set()).add(neighbor)
    counties = [set(regions.get(place, [place])) for place in places]
    touching = [
        set().union(*(neighbors.get(c, set()) for c in cs)) for cs in counties
    ]
    if steps == 2:
        touching = [
            reach.union(*(neighbors.get(c, set()) for c in reach))
            for reach in touching
        ]
    matrix = np.zeros((len(places), len(places)))
    for i in range(len(places)):
        for j in range(len(places)):
            if (
                i != j
                and counties[j] & touching[i]
                and not counties[i] & counties[j]
            ):
                matrix[i, j] = 1
    return matrix


def outbreak_place_clusters(
    places, adjacency_data, regions, year, min_cases=3
):
    """
    The outbreak places of `year` (those with `min_cases` or more outbreak
    cases; `places` has year, fips and cases_outbreak, see
    outbreak_cases_all_counties) and their clusters. Returns the fips (or
    region names) of the places, their outbreak cases, the cluster of each
    (an integer) and the matrix of the places that border each other.
    """
    outbreaks = places.filter(
        (pl.col("year") == year) & (pl.col("cases_outbreak") >= min_cases)
    )
    fips = outbreaks["fips"].to_list()
    cases = outbreaks["cases_outbreak"].to_numpy()
    matrix = place_adjacency_matrix(adjacency_data, fips, regions)
    _, cluster = connected_components(matrix, directed=False)
    return fips, cases, cluster, matrix


def clusters_holding(fips, cluster, seeds):
    """The clusters (integers) that hold one of `seeds`; a seed that is not an
    outbreak place (not in `fips`) holds none."""
    return {cluster[fips.index(seed)] for seed in seeds if seed in fips}


def places_in_cluster(places, adjacency_data, regions, year, seeds):
    """
    The outbreak places of `year` in the clusters that hold one of `seeds`,
    in the order of `places`.
    """
    fips, _, cluster, _ = outbreak_place_clusters(
        places, adjacency_data, regions, year
    )
    held = clusters_holding(fips, cluster, seeds)
    return [place for place, k in zip(fips, cluster) if k in held]


def is_leave_out_fit(
    *,
    model_name,
    years,
    default_covariate_scale,
    sample_subset,
    me_cutoff,
    use_neighbors,
    five_year,
    exclude_clustered,
):
    """
    Whether a combination of the main loop of script 02 is one that gets a
    leave-one-cluster-out fit: the plain baseline of the years with 2025
    (All covariates on the covariate scale of params.yml, all the county-years,
    no cutoff of the medical exemption, no neighbors, no five-year coverage).
    """
    return (
        model_name == "All covariates"
        and 2025 in years
        and default_covariate_scale
        and sample_subset is None
        and me_cutoff is None
        and not use_neighbors
        and not five_year
        and not exclude_clustered
    )


def drop_places_of_year(df, places, year):
    """`df` without the rows of `year` of `places` (fips or region name): the
    other years of those places stay."""
    return df.filter(
        ~((pl.col("year") == year) & pl.col("fips").is_in(places))
    )


def cluster_seeds_2025(adjacency_data, regions):
    """
    The seeds of the clusters of 2025 outbreaks that panel d of
    figure_predicting_county_level_outbreaks.py leaves out one at a time,
    {name: [county fips and names of regions]}. A seed is a county or a
    region that official sources name as part of the outbreak, whatever its
    number of cases; only the seeds that are outbreak places (3+ cases that
    year) hold a cluster (see clusters_holding). `adjacency_data` is
    county_adjacencies.txt; `regions` is region_members().
    """
    fip_getter = AddFIPS()
    #
    # West Texas-NM
    #   Texas: every county in the "2025 Texas Outbreak" column of
    #   the Texas DSHS 2025 Measles Data Report (data as of 1/12/2026; the
    #   12/23/25 report, data as of 12/17/2025, has the same table)
    #   https://www.dshs.texas.gov/sites/default/files/Admin-Meales/doc/2025-measles-outbreak-data-report-011226.pdf
    #   One outbreak in New Mexico, Oklahoma and Texas: MMWR 74(14):232, PMID
    #   40273019, https://doi.org/10.15585/mmwr.mm7414a1 (no Oklahoma county has
    #   3+ cases in 2025, so there is nothing to leave out there)
    #   New Mexico: the outbreak began in west Texas and spread to New Mexico, two
    #   thirds of its 99 cases in Lea County: MMWR 75(9):111, PMID 41818136,
    #   https://doi.org/10.15585/mmwr.mm7509a1 ; Eddy County: NMDOH release of
    #   April 4, 2025, https://www.nmhealth.org/news/vaccine/2025/4/?view=2198
    #   Not left out: Kansas (the KDHE outbreak pages,
    #   https://www.kdhe.ks.gov/2314/Measles-Outbreak-Data, do not link it to the
    #   Texas outbreak); Sandoval and Santa Fe, NM (the MMWR does not name the
    #   eight New Mexico counties of the outbreak and no official source says
    #   their cases are outbreak cases).
    dshs_outbreak_counties = [  # the 38 rows of the column (they add up to 763; DSHS prints 762)
        "Andrews", "Atascosa", "Bailey", "Bell", "Bexar", "Borden", "Brewster",
        "Brown", "Carson", "Cochran", "Collin", "Dallam", "Dawson", "Eastland",
        "Ector", "El Paso", "Erath", "Fannin", "Gaines", "Garza", "Hale",
        "Hardeman", "Hockley", "Lamar", "Lamb", "Lubbock", "Lynn", "Martin",
        "McLennan", "Midland", "Parmer", "Potter", "Randall", "Reeves",
        "Rockwall", "Terry", "Upshur", "Yoakum",
    ]  # fmt: skip
    west_texas_nm = [
        fip_getter.get_county_fips(county, "Texas")
        for county in dshs_outbreak_counties
    ] + [
        "35025",  # Lea, NM
        "35015",  # Eddy, NM
    ]
    assert None not in west_texas_nm
    # Upstate SC
    #   The counties of the 997 outbreak cases (Spartanburg 940, Greenville 36,
    #   Anderson 8, Pickens 6, Lancaster 4, Cherokee 2, Sumter 1): SC DPH
    #   release of April 27, 2026
    #   https://dph.sc.gov/news/dph-announces-end-measles-outbreak-upstate-997-cases
    #   The outbreak (997 cases, October 2025-March 2026) as reported by SC DPH:
    #   MMWR 75(33):516, PMID 42658733, https://doi.org/10.15585/mmwr.mm7533a1
    #   Note: the SC counts of the case data add up to 847, the DPH total of
    #   January 30, 2026; DPH reported 176 cases by December 30, 2025.
    upstate_sc = [
        fip_getter.get_county_fips(county, "South Carolina")
        for county in [
            "Spartanburg", "Greenville", "Anderson", "Pickens", "Lancaster",
            "Cherokee", "Sumter",
        ]
    ]  # fmt: skip
    assert None not in upstate_sc
    # Utah-Arizona
    #   Utah DHHS counts the Utah residents diagnosed "in this outbreak" for the
    #   whole state (724 as of 10/27/2026, 197 of them in 2025), so every Utah
    #   place is left out: https://epi.utah.gov/measles-response/
    #   Mohave County, AZ: the outbreak of the Arizona-Utah border communities,
    #   managed by Mohave County Public Health since August 2025, declared over
    #   on June 11, 2026:
    #   https://www.mohave.gov/news-notices/posts/adhs-and-mohave-county-announce-end-of-measles-outbreak/
    #   The Utah-Arizona border outbreak (Mohave County, AZ and Washington County,
    #   UT, from August 2025; "measles has since spread throughout Utah" after
    #   February 28, 2026; linked transmission also in Idaho and Colorado, no
    #   counties named): Marye et al., medRxiv preprint, PMID 42528501,
    #   https://doi.org/10.64898/2026.07.09.26357695
    #   Utah's whole 2025-2026 outbreak, 602 cases June 2025-April 2026 (Utah
    #   DHHS authors): NEJM Evidence, PMID 42446927,
    #   https://doi.org/10.1056/EVIDpha2600141
    #   Only Mohave is a seed in Arizona: the sources name no other Arizona county
    #   (and no other Arizona place has 3+ cases in 2025).
    #   Open: there is no MMWR; the linkage evidence covers the border core
    #   (Washington County and Mohave), and the rest of Utah is left out because
    #   Utah DHHS counts the whole state as one outbreak. Utah DHHS gives the
    #   cases by health district in a map and a dashboard
    #   (https://files.epi.utah.gov/Utah%20measles%20dashboard.html) that were
    #   not read. Mesa County, CO is left out only because it borders Grand
    #   County, UT (no official source links it to this outbreak).
    # every Utah county, and every region (health district) made of Utah counties
    utah_places = sorted(
        set(
            adjacency_data["County GEOID"].filter(
                adjacency_data["County GEOID"].str.starts_with("49")
            )
        )
    ) + [
        region
        for region, members in regions.items()
        if all(member.startswith("49") for member in members)
    ]
    return {
        "West Texas-NM": west_texas_nm,
        "Upstate SC": upstate_sc,
        "Utah-Arizona": ["04015", *utah_places],
    }
