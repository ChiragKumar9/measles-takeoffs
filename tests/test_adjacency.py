"""The county adjacency matrix (outbreak_size.adjacency)."""

import polars as pl
import pytest

from outbreak_size.adjacency import (
    CONNECTICUT_COUNTY_PAIRS,
    add_connecticut_counties,
    assemble_adjacency_matrix,
    require_county_order,
    require_same_unit_order,
    state_adjacency_matrix,
)


def adjacency_data():
    # the file lists each pair in both directions
    pairs = [("01001", "01003"), ("01003", "01005"), ("01005", "99999")]
    rows = pairs + [(b, a) for a, b in pairs]
    return pl.DataFrame(
        {
            "County GEOID": [a for a, _ in rows],
            "Neighbor GEOID": [b for _, b in rows],
        }
    )


def test_matrix_follows_the_order_of_the_fips_list():
    matrix = assemble_adjacency_matrix(
        adjacency_data(), ["01005", "01001", "01003"]
    )
    # order: 01005, 01001, 01003; 01001-01003 and 01003-01005 are neighbors
    assert matrix.tolist() == [[0, 0, 1], [0, 0, 1], [1, 1, 0]]


def test_matrix_is_symmetric_and_ignores_counties_outside_the_list():
    matrix = assemble_adjacency_matrix(
        adjacency_data(), ["01001", "01003", "01005"]
    )
    assert (matrix == matrix.T).all()
    assert matrix.shape == (3, 3)  # 99999 is not in the dataset


def test_a_county_without_neighbors_in_the_list_has_an_empty_row():
    matrix = assemble_adjacency_matrix(adjacency_data(), ["01001", "01005"])
    assert matrix.sum() == 0


def test_the_counties_of_connecticut_get_their_neighbors():
    # the file lists Connecticut by planning regions (09110...), not counties
    data = adjacency_data()
    matrix = assemble_adjacency_matrix(
        data, ["09001", "09009", "36119", "09003"]
    )
    # Fairfield borders New Haven and Westchester NY; Hartford borders New
    # Haven; Fairfield does not border Hartford, New Haven not Westchester
    assert matrix.tolist() == [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]


def test_connecticut_pairs_are_added_in_both_directions_once():
    data = add_connecticut_counties(adjacency_data())
    pairs = set(zip(data["County GEOID"], data["Neighbor GEOID"]))
    assert len(pairs) == data.height  # no pair twice
    assert ("09001", "36119") in pairs and ("36119", "09001") in pairs
    assert len(CONNECTICUT_COUNTY_PAIRS) == 32
    # added once: data that list the counties already are left as they are
    assert add_connecticut_counties(data).height == data.height


def test_the_planning_regions_of_connecticut_are_removed():
    # NY 36119 and MA 25013 both border the planning region 09190: left in,
    # it would make them neighbors of neighbors. Rows with a missing id stay.
    pairs = [("36119", "09190"), ("25013", "09190"), ("09190", "09110")]
    rows = pairs + [(b, a) for a, b in pairs] + [("01001", None)]
    data = pl.DataFrame(
        {
            "County GEOID": [a for a, _ in rows],
            "Neighbor GEOID": [b for _, b in rows],
        }
    )
    result = add_connecticut_counties(data)
    ids = set(result["County GEOID"]) | set(
        result["Neighbor GEOID"].drop_nulls()
    )
    assert not {i for i in ids if i.startswith("09") and i >= "09100"}
    assert ("01001", None) in set(
        zip(result["County GEOID"], result["Neighbor GEOID"])
    )
    # the county pairs replaced them, and Westchester no longer reaches
    # Hampden through a region
    pairs_after = set(zip(result["County GEOID"], result["Neighbor GEOID"]))
    assert ("36119", "09001") in pairs_after
    assert ("36119", "25013") not in pairs_after


def test_every_connecticut_county_has_neighbors_and_no_planning_region_is_used():
    counties = {c for pair in CONNECTICUT_COUNTY_PAIRS for c in pair}
    in_connecticut = sorted(c for c in counties if c.startswith("09"))
    assert in_connecticut == [
        "09001", "09003", "09005", "09007", "09009", "09011", "09013", "09015",
    ]  # fmt: skip
    data = add_connecticut_counties(adjacency_data())
    for county in in_connecticut:
        assert data.filter(pl.col("County GEOID") == county).height >= 3


def test_a_county_without_a_row_in_the_data_warns():
    with pytest.warns(UserWarning, match="without neighbors.*12345"):
        assemble_adjacency_matrix(adjacency_data(), ["01001", "12345"])


def test_a_county_listed_twice_raises():
    with pytest.raises(ValueError, match="more than once.*01001"):
        assemble_adjacency_matrix(
            adjacency_data(), ["01001", "01003", "01001"]
        )


def test_a_fips_that_is_no_county_may_be_listed_twice():
    # the Iowa regions share the state's fips (19), which has no neighbors
    matrix = assemble_adjacency_matrix(
        adjacency_data(), ["01001", "19", "01003", "19"]
    )
    assert matrix.tolist() == [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_a_pair_listed_in_one_direction_only_raises():
    one_way = adjacency_data().head(1)  # 01001 -> 01003 only
    with pytest.raises(ValueError, match="one direction only"):
        assemble_adjacency_matrix(one_way, ["01001", "01003"])


# --- the matrix rows are the counties of the data in np.unique order --------


def test_matrix_order_check_accepts_the_sorted_counties_of_the_rows():
    # a county in several rows counts once; the row order does not matter
    require_county_order(
        ["01001", "01003"], ["01003", "01001", "01003"], "years"
    )


def test_matrix_order_check_raises_on_another_order():
    with pytest.raises(ValueError, match="years: the adjacency matrix has 2"):
        require_county_order(["01003", "01001"], ["01001", "01003"], "years")


@pytest.mark.parametrize(
    "matrix_fips",
    [["01001"], ["01001", "01003", "01005"], ["01001", "01005"]],
)
def test_matrix_order_check_raises_on_other_counties(matrix_fips):
    with pytest.raises(ValueError, match="sorted order"):
        require_county_order(matrix_fips, ["01001", "01003"], "years")


def test_matrix_order_check_names_the_row_and_county_counts():
    with pytest.raises(ValueError, match="has 3 rows .* the 2 counties"):
        require_county_order(
            ["01001", "01003", "01005"], ["01001", "01003"], "years"
        )


# --- the units of a model are numbered like the rows of its matrix ----------


def test_unit_order_check_accepts_the_same_places_in_the_same_order():
    # repeated rows (years) of a place do not matter
    require_same_unit_order(
        ["01003", "01001", "01003", "01001"],
        ["g3", "g1", "g3", "g1"],
        "rows",
    )


def test_unit_order_check_raises_when_two_places_share_a_fips():
    with pytest.raises(ValueError, match="rows: the spatial units"):
        require_same_unit_order(["19", "19", "01001"], ["a", "b", "c"], "rows")


def test_unit_order_check_raises_when_a_place_has_two_fips():
    with pytest.raises(ValueError, match="same order"):
        require_same_unit_order(["01001", "01003"], ["a", "a"], "rows")


def test_unit_order_check_raises_when_the_order_differs():
    # the places are the same, but the numbering by first appearance of the
    # second row's place differs
    with pytest.raises(ValueError, match="same order"):
        require_same_unit_order(
            ["01001", "01003", "01001", "01005"],
            ["g1", "g3", "g5", "g5"],
            "rows",
        )


# --- states ----------------------------------------------------------------


def county_pairs(pairs):
    rows = pairs + [(b, a) for a, b in pairs]
    return pl.DataFrame(
        {
            "County GEOID": [a for a, _ in rows],
            "Neighbor GEOID": [b for _, b in rows],
        }
    )


def test_states_are_adjacent_when_any_of_their_counties_border():
    data = county_pairs(
        [
            ("01001", "01003"),  # inside state 01
            ("01003", "02005"),  # 01 borders 02
            ("02007", "03009"),  # 02 borders 03
        ]
    )
    matrix = state_adjacency_matrix(data, ["01", "02", "03", "04"])
    assert matrix.tolist() == [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]


def test_the_state_matrix_follows_the_order_of_the_prefixes():
    data = county_pairs([("01003", "02005")])
    forward = state_adjacency_matrix(data, ["01", "02", "03"])
    shuffled = state_adjacency_matrix(data, ["03", "02", "01"])
    assert forward.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert shuffled.tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0]]


def test_a_state_is_not_its_own_neighbour_and_other_states_are_ignored():
    data = county_pairs([("01001", "01003"), ("01003", "09005")])
    matrix = state_adjacency_matrix(data, ["01", "02"])
    assert matrix.tolist() == [[0, 0], [0, 0]]  # 09 is not in the list


def test_the_state_matrix_is_symmetric_even_if_the_file_lists_a_pair_once():
    one_way = pl.DataFrame(
        {"County GEOID": ["01003"], "Neighbor GEOID": ["02005"]}
    )
    matrix = state_adjacency_matrix(one_way, ["01", "02"])
    assert matrix.tolist() == [[0, 1], [1, 0]]
