"""The clusters of 2025 outbreaks that are left out of panel d of
figure_predicting_county_level_outbreaks.py and of the leave-one-cluster-out
fits of script 02 (outbreak_size.outbreak_clusters)."""

import re

import numpy as np
import polars as pl
import pytest

from outbreak_size.outbreak_clusters import (
    CLUSTER_NAMES,
    CLUSTER_TAGS,
    cluster_seeds_2025,
    clusters_holding,
    drop_places_of_year,
    outbreak_cases_all_counties,
    outbreak_place_clusters,
    place_adjacency_matrix,
    places_in_cluster,
)

# counties that border as A - B - C, D - E, and F with no neighbor
A, B, C, D, E, F = "01001", "01003", "01005", "01007", "01009", "01011"


def borders(*pairs):
    """The adjacency table with each pair listed in both directions."""
    both = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    return pl.DataFrame(
        {
            "County GEOID": [a for a, _ in both],
            "Neighbor GEOID": [b for _, b in both],
        }
    )


@pytest.fixture
def adjacency_data():
    # "09001" is there so that the Connecticut counties are not added
    return borders((A, B), (B, C), (D, E), ("09001", "09003"))


def outbreak_cases(rows):
    """(year, fips, cases_outbreak) rows as the frame the cluster code takes."""
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "fips": pl.String,
            "cases_outbreak": pl.Int64,
        },
        orient="row",
    )


@pytest.fixture
def places():
    return outbreak_cases(
        [
            (2025, A, 5),
            (2025, B, 3),
            (2025, C, 1),  # borders B, but is no outbreak place (1 case)
            (2025, D, 4),
            (2025, E, 6),
            (2025, F, 3),  # an outbreak place with no neighbor
            (2024, A, 9),  # another year
        ]
    )


# --- the places that border each other ----------------------------------------


def test_places_that_border_are_neighbors_and_the_matrix_is_symmetric(
    adjacency_data,
):
    matrix = place_adjacency_matrix(adjacency_data, [A, B, C, D], {})
    assert matrix.tolist() == [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert np.array_equal(matrix, matrix.T)


def test_two_steps_also_joins_places_with_a_county_between_them(
    adjacency_data,
):
    matrix = place_adjacency_matrix(adjacency_data, [A, C], {}, steps=2)
    assert matrix[0, 1] == 1
    assert place_adjacency_matrix(adjacency_data, [A, C], {})[0, 1] == 0


def test_a_region_borders_a_place_if_any_of_its_counties_does(adjacency_data):
    regions = {"Region R": [C, "01099"]}
    matrix = place_adjacency_matrix(
        adjacency_data, [A, B, "Region R"], regions
    )
    # R holds C, which borders B but not A
    assert matrix.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


def test_places_that_share_a_county_are_not_neighbors(adjacency_data):
    regions = {"R1": [B, C], "R2": [C, D]}
    matrix = place_adjacency_matrix(adjacency_data, ["R1", "R2"], regions)
    assert matrix.sum() == 0


# --- the clusters of the outbreak places --------------------------------------


def test_the_outbreak_places_of_the_year_are_joined_by_borders(
    adjacency_data, places
):
    fips, cases, cluster, _ = outbreak_place_clusters(
        places, adjacency_data, {}, 2025
    )
    # C has 1 case, so it is no outbreak place and does not join A to B's side
    assert fips == [A, B, D, E, F]
    assert cases.tolist() == [5, 3, 4, 6, 3]
    assert cluster[0] == cluster[1]  # A - B
    assert cluster[2] == cluster[3]  # D - E
    assert len({cluster[0], cluster[2], cluster[4]}) == 3


def test_only_the_cases_of_the_year_count(adjacency_data, places):
    fips, *_ = outbreak_place_clusters(places, adjacency_data, {}, 2024)
    assert fips == [A]


def test_a_cluster_is_left_out_whole(adjacency_data, places):
    assert places_in_cluster(places, adjacency_data, {}, 2025, [A]) == [A, B]
    # any place of the cluster as the seed gives the same cluster
    assert places_in_cluster(places, adjacency_data, {}, 2025, [B]) == [A, B]
    assert places_in_cluster(places, adjacency_data, {}, 2025, [E]) == [D, E]


def test_a_seed_that_is_not_an_outbreak_place_leaves_nothing_out(
    adjacency_data, places
):
    # C is named by the sources but has 1 case; "99999" is in no data
    assert places_in_cluster(places, adjacency_data, {}, 2025, [C]) == []
    assert places_in_cluster(places, adjacency_data, {}, 2025, ["99999"]) == []
    assert places_in_cluster(places, adjacency_data, {}, 2025, []) == []


def test_several_seeds_leave_out_each_of_their_clusters_once(
    adjacency_data, places
):
    left_out = places_in_cluster(places, adjacency_data, {}, 2025, [A, B, F])
    assert left_out == [A, B, F]


def test_a_cluster_of_one_place_is_left_out_alone(adjacency_data, places):
    assert places_in_cluster(places, adjacency_data, {}, 2025, [F]) == [F]


def test_a_region_is_part_of_the_cluster_of_the_places_it_borders(
    adjacency_data,
):
    regions = {"Region R": [C]}
    frame = outbreak_cases(
        [(2025, A, 5), (2025, B, 3), (2025, "Region R", 4), (2025, D, 4)]
    )
    left_out = places_in_cluster(frame, adjacency_data, regions, 2025, [A])
    assert left_out == [A, B, "Region R"]


def test_clusters_holding_matches_seeds_to_labels():
    fips = ["a", "b", "c"]
    cluster = np.array([0, 0, 1])
    assert clusters_holding(fips, cluster, ["a"]) == {0}
    assert clusters_holding(fips, cluster, ["c", "zz"]) == {1}
    assert clusters_holding(fips, cluster, ["zz"]) == set()


# --- the rows that a leave-one-cluster-out fit drops ------------------------


def test_only_the_rows_of_the_year_of_the_places_are_dropped():
    df = pl.DataFrame(
        {
            "year": [2024, 2025, 2025, 2025, 2025],
            "fips": [A, A, B, C, "Region R"],
            "cases_outbreak": [9, 5, 3, 1, 4],
        }
    )
    kept = drop_places_of_year(df, [A, "Region R"], 2025)
    assert kept["fips"].to_list() == [A, B, C]
    assert kept["year"].to_list() == [2024, 2025, 2025]  # A stays in 2024
    assert df.height == 5  # the input is not changed


def test_dropping_no_place_drops_nothing():
    df = pl.DataFrame({"year": [2025, 2025], "fips": [A, B], "x": [1, 2]})
    assert drop_places_of_year(df, [], 2025).equals(df)


# --- the case data of all the places ---------------------------------------


def test_the_places_come_from_the_case_data_whether_or_not_they_have_vaccine_data():
    raw_merged = pl.DataFrame(
        {
            "state": ["Texas", "Texas", "Utah", "Utah"],
            "county": [
                "Harris",
                "Dallas",
                "Bear River, Utah",
                "Some District",
            ],
            "year": [2025, 2025, 2025, 2025],
            "cases": [5, 2, 10, 7],
            "imported": [None, 1, None, None],
            "fips": [None, "48113", None, None],
            "nonmed_exempt_pct": [1.0, 1.0, 1.0, 1.0],
            "med_exempt_pct": [0.5, 0.5, 0.5, 0.5],
        },
        schema_overrides={"imported": pl.Int64, "fips": pl.String},
    )
    regions = {"Bear River, Utah": ["49003", "49005"]}
    out = outbreak_cases_all_counties(raw_merged, regions).sort("fips")
    # Harris takes its fips from its name, Dallas has 2 - 1 outbreak cases, the
    # region keeps its name, and the district that is no known region is left out
    assert out["fips"].to_list() == ["48113", "48201", "Bear River, Utah"]
    assert out["cases_outbreak"].to_list() == [1, 5, 10]


# --- the seeds ---------------------------------------------------------------


@pytest.fixture
def seeds():
    utah = borders(("49001", "49003"), ("49003", "49005"), ("08001", "49001"))
    regions = {
        "Southwest Health District, Utah": ["49053", "49021"],
        "Mixed region": ["49001", "08001"],
    }
    return cluster_seeds_2025(utah, regions)


def test_there_is_a_seed_list_for_each_cluster(seeds):
    assert list(seeds) == list(CLUSTER_NAMES)
    assert all(len(s) == len(set(s)) for s in seeds.values())


def test_west_texas_nm_seeds_are_the_40_counties_of_the_report(seeds):
    west_texas = seeds["West Texas-NM"]
    assert len(west_texas) == 40
    assert all(re.fullmatch(r"\d{5}", s) for s in west_texas)
    assert "48141" in west_texas  # El Paso
    assert {"35025", "35015"} <= set(west_texas)  # Lea and Eddy, NM
    assert sum(s.startswith("48") for s in west_texas) == 38


def test_upstate_sc_seeds_are_the_seven_counties_of_the_outbreak(seeds):
    upstate = seeds["Upstate SC"]
    assert len(upstate) == 7
    assert all(s.startswith("45") for s in upstate)
    assert {"45083", "45045"} <= set(upstate)  # Spartanburg, Greenville


def test_utah_arizona_seeds_are_mohave_every_utah_place_and_no_other(seeds):
    utah_arizona = seeds["Utah-Arizona"]
    assert utah_arizona[0] == "04015"  # Mohave County, AZ
    assert {"49001", "49003", "49005"} <= set(utah_arizona)
    assert "Southwest Health District, Utah" in utah_arizona
    # a region with a county of another state, and counties of other states
    assert "Mixed region" not in utah_arizona
    assert "08001" not in utah_arizona


def test_each_cluster_has_a_tag_for_the_names_of_the_files():
    assert list(CLUSTER_TAGS) == list(CLUSTER_NAMES)
    tags = list(CLUSTER_TAGS.values())
    assert len(set(tags)) == len(tags)
    assert all(re.fullmatch(r"[a-z_]+", tag) for tag in tags)


# --- which fits of script 02 are also fit without a cluster ------------------


def test_only_the_plain_baseline_of_the_years_with_2025_is_fit_without_a_cluster():
    """The grid of the main loop of script 02: of all its combinations one
    gets the leave-one-cluster-out fits (so 3 extra fits for 3 clusters)."""
    from itertools import product

    from outbreak_size.outbreak_clusters import is_leave_out_fit

    grid = product(
        ["MMR coverage", "MMR coverage + exemptions", "All covariates"],
        [list(range(2011, 2021)), list(range(2021, 2026))],
        [True, False],  # the covariate scale of params.yml
        [None, "imports_known", "mmr_coverage_only"],
        [None, 5.0, 10.0],
        [False, True],  # neighbors
        [True, False],  # five-year coverage
        [False, True],  # the legacy exclusion of clustered outbreaks
    )
    chosen = [
        combination
        for combination in grid
        if is_leave_out_fit(
            model_name=combination[0],
            years=combination[1],
            default_covariate_scale=combination[2],
            sample_subset=combination[3],
            me_cutoff=combination[4],
            use_neighbors=combination[5],
            five_year=combination[6],
            exclude_clustered=combination[7],
        )
    ]
    assert chosen == [
        (
            "All covariates",
            [2021, 2022, 2023, 2024, 2025],
            True,
            None,
            None,
            False,
            False,
            False,
        )
    ]
