"""The analysis dataset and its adjacency matrices (outbreak_size.analysis_dataset)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.analysis_dataset import (
    assemble_analysis_dataset,
    merge_outbreaks_vaccinations,
    process_dataset,
    require_vaccine_data,
)

# county-years of the cleaned records: (year, fips, county, cases_outbreak)
ROWS = [
    (2022, "01001", "Autauga", 30),
    (2023, "01001", "Autauga", 12),
    (2024, "01001", "Autauga", 40),
    (2025, "01001", "Autauga", 55),
    (2024, "01003", "Baldwin", 5),
    (2025, "01005", "Barbour", 10),  # a 2025 county-year with < 20 cases
    (2025, None, "North, Alabama", 30),  # a region without a county fips
    (2025, "01007", "Bibb", 100),
]


@pytest.fixture
def result(tmp_path):
    cleaned = pl.DataFrame(
        {
            "year": [r[0] for r in ROWS],
            "year_relevant_vaccination": [r[0] - 1 for r in ROWS],
            "fips": [r[1] for r in ROWS],
            "county": [r[2] for r in ROWS],
            "state": ["Alabama"] * len(ROWS),
            "cases_outbreak": [r[3] for r in ROWS],
            "estimated_mmr_cov": [90.0] * len(ROWS),
        }
    )
    cleaned.write_csv(tmp_path / "cleaned.csv")
    # each pair is listed in both directions
    pairs = [("01001", "01003"), ("01001", "01005"), ("01003", "01007")]
    adjacency = pl.DataFrame(
        {
            "County GEOID": [a for a, b in pairs] + [b for a, b in pairs],
            "Neighbor GEOID": [b for a, b in pairs] + [a for a, b in pairs],
        }
    )
    vaccinations = pl.DataFrame(
        {
            "fips": ["01003", "01005", "01003", "01001"],
            "year": [2023, 2023, 2022, 2023],
            "estimated_mmr_cov": [90.0, 80.0, 50.0, 70.0],
        }
    )
    process_dataset(
        str(tmp_path / "cleaned.csv"),
        str(tmp_path / "analysis.csv"),
        str(tmp_path / "adjacency.csv"),
        str(tmp_path / "adjacency_no_oddities.csv"),
        adjacency,
        vaccinations,
    )
    return {
        "df": pl.read_csv(
            tmp_path / "analysis.csv", schema_overrides={"fips": pl.String}
        ),
        "adjacency": np.loadtxt(tmp_path / "adjacency.csv", delimiter=","),
        "adjacency_no_oddities": np.loadtxt(
            tmp_path / "adjacency_no_oddities.csv", delimiter=","
        ),
    }


def test_every_row_is_kept_with_a_dummy_for_its_year(result):
    df = result["df"]
    assert df.height == len(ROWS)
    assert "year_relevant_vaccination" not in df.columns
    for year in (2022, 2023, 2024, 2025):
        assert (df[f"year_{year}"] == (df["year"] == year).cast(int)).all()


def test_years_are_paired_into_contiguous_two_year_steps(result):
    got = dict(
        zip(
            result["df"]["year"],
            result["df"]["year_every_two_contiguous"],
        )
    )
    # 2022 and 2023 -> 2022; 2024 and 2025 -> 2024; contiguous from 0
    assert got == {2022: 0, 2023: 0, 2024: 1, 2025: 1}


def test_a_region_without_a_fips_is_identified_by_its_name(result):
    df = result["df"]
    region = df.filter(pl.col("county") == "North, Alabama")
    assert region["fips"].to_list() == ["North, Alabama"]
    assert region["mega_region_2025"].to_list() == [True]
    assert (
        df.filter(pl.col("county") == "Autauga")["mega_region_2025"].sum() == 0
    )


def test_a_2025_county_year_with_under_20_cases_is_an_oddity(result):
    df = result["df"]
    oddities = df.filter(pl.col("oddities_2025"))
    assert oddities["county"].to_list() == ["Barbour"]
    # small outbreaks in other years are not oddities
    assert not df.filter(pl.col("year") == 2024)["oddities_2025"].any()


def test_neighbors_coverage_averages_the_neighbors_of_the_year_before(result):
    df = result["df"]

    def neighbors(county, year):
        return df.filter(
            (pl.col("county") == county) & (pl.col("year") == year)
        )["neighbors_coverage"][0]

    # Autauga's neighbors are Baldwin (90) and Barbour (80) in 2023 -> 2024
    assert neighbors("Autauga", 2024) == pytest.approx(85.0)
    # in 2023 the year before is 2022: only Baldwin has a value (50)
    assert neighbors("Autauga", 2023) == pytest.approx(50.0)
    # Baldwin's neighbors are Autauga (70 in 2023) and Bibb (no data)
    assert neighbors("Baldwin", 2024) == pytest.approx(70.0)
    # a county with no neighbors in the adjacency data has no value
    assert neighbors("North, Alabama", 2025) is None


def test_the_adjacency_matrix_follows_the_order_of_the_counties(result):
    # the counties in the order they first appear in the dataset
    counties = result["df"]["fips"].unique(maintain_order=True).to_list()
    matrix = result["adjacency"]
    assert matrix.shape == (5, 5)
    assert (matrix == matrix.T).all()
    index = {fips: i for i, fips in enumerate(counties)}
    pairs = [("01001", "01003"), ("01001", "01005"), ("01003", "01007")]
    for a, b in pairs:
        assert matrix[index[a], index[b]] == 1
    assert matrix.sum() == 2 * len(pairs)
    assert matrix[index["North, Alabama"]].sum() == 0  # no neighbors


def test_the_matrix_without_oddities_leaves_out_the_oddity_counties(result):
    # Barbour only has a 2025 oddity, so it is left out
    assert result["adjacency_no_oddities"].shape == (4, 4)


# --- merging the case records with the vaccination data ----------------------


def case_rows(rows):
    """rows: (year, state, county, fips, cases)"""
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
            "cases": pl.Int64,
        },
        orient="row",
    )


def vaccine_rows(rows):
    """rows: (year, state, county, fips, estimated_mmr_cov)"""
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
        },
        orient="row",
    )


def test_cases_take_the_vaccination_data_of_the_year_before():
    cases = case_rows(
        [(2020, "A", "x", "01001", 5), (2021, "A", "x", "01001", 7)]
    )
    vaccines = vaccine_rows(
        [(2019, "A", "x", "01001", 91.0), (2020, "A", "x", "01001", 92.0)]
    )
    out = merge_outbreaks_vaccinations(cases, vaccines)
    assert out["year_relevant_vaccination"].to_list() == [2019, 2020]
    assert out["estimated_mmr_cov"].to_list() == [91.0, 92.0]


def test_cases_without_vaccination_data_are_kept_with_nulls():
    cases = case_rows(
        [(2020, "A", "x", "01001", 5), (2020, "A", "y", "01003", 2)]
    )
    vaccines = vaccine_rows([(2019, "A", "x", "01001", 91.0)])
    out = merge_outbreaks_vaccinations(cases, vaccines)
    assert out["county"].to_list() == ["x", "y"]
    assert out["estimated_mmr_cov"].to_list() == [91.0, None]
    assert out["cases"].to_list() == [5, 2]


def test_the_vaccination_fips_replaces_the_case_fips_when_there_is_one():
    """The case sheet's fips is kept only where no vaccination row matches."""
    cases = case_rows(
        [(2020, "A", "x", "99999", 5), (2020, "A", "y", "01003", 2)]
    )
    vaccines = vaccine_rows([(2019, "A", "x", "01001", 91.0)])
    out = merge_outbreaks_vaccinations(cases, vaccines)
    assert out["fips"].to_list() == ["01001", "01003"]


def test_the_case_rows_keep_their_order():
    cases = case_rows(
        [(2020, "A", c, f"0100{i}", i) for i, c in enumerate("edcba")]
    )
    vaccines = vaccine_rows(
        [(2019, "A", c, f"0100{i}", 90.0) for i, c in enumerate("abcde")]
    )
    assert merge_outbreaks_vaccinations(cases, vaccines)[
        "county"
    ].to_list() == list("edcba")


def test_new_york_city_cases_get_the_fips_of_kings():
    cases = case_rows([(2025, "New York", "Kings", "36061", 9)])
    out = merge_outbreaks_vaccinations(cases, vaccine_rows([]))
    assert out["fips"].to_list() == ["36047"]


def test_a_repeated_vaccination_row_of_a_county_with_cases_is_refused():
    """It would repeat the county's case row."""
    cases = case_rows([(2020, "A", "x", "01001", 5)])
    vaccines = vaccine_rows(
        [(2019, "A", "x", "01001", 91.0), (2019, "A", "x", "01001", 92.0)]
    )
    with pytest.raises(ValueError, match="1 case rows into 2"):
        merge_outbreaks_vaccinations(cases, vaccines)


def test_a_repeated_vaccination_row_of_a_place_without_cases_is_harmless():
    """The real data repeat the row of a school district that has no cases."""
    cases = case_rows([(2020, "A", "x", "01001", 5)])
    vaccines = vaccine_rows(
        [
            (2019, "A", "x", "01001", 91.0),
            (2019, "A", "district", None, None),
            (2019, "A", "district", None, None),
        ]
    )
    out = merge_outbreaks_vaccinations(cases, vaccines)
    assert out.height == 1 and out["estimated_mmr_cov"].to_list() == [91.0]


# --- splitting by availability and counting the outbreak cases ---------------


def vaccine_columns(rows):
    return pl.DataFrame(
        rows,
        schema={
            "county": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
        orient="row",
    )


def test_rows_split_by_whether_all_three_vaccine_values_exist():
    df = vaccine_columns(
        [
            ("complete", 90.0, 1.0, 1.0),
            ("no coverage", None, 1.0, 1.0),
            ("no nonmedical", 90.0, None, 1.0),
            ("no medical", 90.0, 1.0, None),
            ("none at all", None, None, None),
        ]
    )
    have, lack = require_vaccine_data(df)
    assert have["county"].to_list() == ["complete"]
    assert lack["county"].to_list() == [
        "no coverage",
        "no nonmedical",
        "no medical",
        "none at all",
    ]


def test_every_row_is_in_exactly_one_of_the_two_parts():
    df = vaccine_columns([("a", 1.0, 1.0, 1.0), ("b", None, 1.0, 1.0)])
    have, lack = require_vaccine_data(df)
    assert have.height + lack.height == df.height


def test_outbreak_cases_are_the_cases_not_imported_and_exemptions_add_up():
    df = pl.DataFrame(
        {
            "cases": [10, 10, 10, 10, 4],
            "imported": [None, 0, 1, 3, 4],
            "nonmed_exempt_pct": [2.0] * 5,
            "med_exempt_pct": [0.5] * 5,
        }
    )
    out = assemble_analysis_dataset(df)
    # a missing or zero import count leaves the cases; otherwise the imports
    # are taken off
    assert out["cases_outbreak"].to_list() == [10, 10, 9, 7, 0]
    assert out["total_exempt_pct"].to_list() == [2.5] * 5
