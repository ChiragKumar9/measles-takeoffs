"""The cells of the compiled vaccination spreadsheet that are replaced by the
data owner's file of 9 Nov 2026 or set by hand
(vaccination_wrangling.fix_known_values), and the QC exclusion of Warren."""

import os

import polars as pl
import pytest
import yaml

from outbreak_size import vaccination_wrangling as vw

NOV9_COLUMNS = [
    "Year",
    "State",
    "County",
    "Non-medical exemptions (%)",
    "Medical exemptions (%)",
    "MMR vaccine coverage",
    "Up-to-date vaccination (all vaccines)",
    "FIPS",
    "county_type",
]


def nov9(tmp_path, rows):
    """rows: (year, state, county, nonmed, med, mmr, allv)"""
    pl.DataFrame(
        [(*r, "00000", "County") for r in rows],
        schema=NOV9_COLUMNS,
        orient="row",
    ).write_csv(tmp_path / vw.VACCINATION_NOV9_FILE)
    return str(tmp_path)


def spreadsheet(rows):
    """rows: (state, county, year, nonmed, med, mmr, allv), the values text
    as the spreadsheet has them."""
    return pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "nonmed_exempt_pct": pl.String,
            "med_exempt_pct": pl.String,
            "MMR_cov": pl.String,
            "all_vacc_cov": pl.String,
        },
        orient="row",
    )


def test_every_county_of_a_state_takes_the_files_value_blank_or_not(tmp_path):
    data = nov9(
        tmp_path,
        [
            (2024, "Virginia", "Bath", 1.0, 0.0, 95.83, 90.0),
            (2024, "Virginia", "Dekalb", 1.0, 0.0, 90.36, 90.0),
            # the file has no MMR for Lunenburg: the spreadsheet's value is
            # its neighbour's, so it goes
            (2024, "Virginia", "Lunenburg", 1.0, 0.0, None, 90.0),
        ],
    )
    df = spreadsheet(
        [
            ("Virginia", "Bath", 2024, "1", "0", "94.118", "91"),
            ("Virginia", "DeKalb", 2024, "1", "0", "95.833", "91"),
            ("Virginia", "Lunenburg", 2024, "1", "0", "92.228", "91"),
            ("Virginia", "Bath", 2023, "1", "0", "88.0", "91"),
            ("Ohio", "Bath", 2024, "1", "0", "70.0", "91"),
            # a county the file lacks keeps its value
            ("Virginia", "Orange", 2024, "1", "0", "93.0", "91"),
        ]
    )
    out = vw.fix_known_values(
        df,
        data,
        replacements=[("Virginia", None, 2024, ["MMR_cov"])],
        known_values=[],
    )
    assert out.schema == df.schema and out.height == df.height
    got = {
        (r["state"], r["county"], r["year"]): r["MMR_cov"]
        for r in out.iter_rows(named=True)
    }
    assert got[("Virginia", "Bath", 2024)] == "95.83"
    assert (
        got[("Virginia", "DeKalb", 2024)] == "90.36"
    )  # matched ignoring case
    assert got[("Virginia", "Lunenburg", 2024)] is None
    assert got[("Virginia", "Bath", 2023)] == "88.0"
    assert got[("Ohio", "Bath", 2024)] == "70.0"
    assert got[("Virginia", "Orange", 2024)] == "93.0"
    # only the named column changed
    assert out["all_vacc_cov"].to_list() == df["all_vacc_cov"].to_list()
    assert (
        out["nonmed_exempt_pct"].to_list() == df["nonmed_exempt_pct"].to_list()
    )


def test_only_the_named_counties_years_and_columns_change(tmp_path):
    data = nov9(
        tmp_path,
        [
            (2021, "Florida", "Sarasota", 10.2, 2.2, 90.0, 80.0),
            (2021, "Florida", "Polk", 9.9, 9.9, 9.9, 9.9),
            (2020, "Florida", "Sarasota", 9.9, 9.9, 9.9, 89.8),
        ],
    )
    df = spreadsheet(
        [
            ("Florida", "Sarasota", 2021, "4.8", "2.7", None, "87.4"),
            ("Florida", "Polk", 2021, "2.3", "1.9", None, "95.8"),
            ("Florida", "Sarasota", 2020, "3.5", "1.8", None, "94.4"),
        ]
    )
    out = vw.fix_known_values(
        df,
        data,
        replacements=[
            (
                "Florida",
                ["Sarasota"],
                2021,
                ["nonmed_exempt_pct", "med_exempt_pct"],
            ),
            ("Florida", ["Sarasota"], 2020, ["all_vacc_cov"]),
        ],
        known_values=[],
    )
    rows = {(r["county"], r["year"]): r for r in out.iter_rows(named=True)}
    s21, s20, polk = (
        rows[("Sarasota", 2021)],
        rows[("Sarasota", 2020)],
        rows[("Polk", 2021)],
    )
    assert (s21["nonmed_exempt_pct"], s21["med_exempt_pct"]) == ("10.2", "2.2")
    assert s21["all_vacc_cov"] == "87.4"  # not named for 2021
    assert s20["all_vacc_cov"] == "89.8"
    assert (s20["nonmed_exempt_pct"], s20["med_exempt_pct"]) == ("3.5", "1.8")
    assert (polk["nonmed_exempt_pct"], polk["all_vacc_cov"]) == ("2.3", "95.8")


def test_a_blank_is_filled_and_a_county_that_is_not_there_is_an_error(
    tmp_path,
):
    data = nov9(
        tmp_path, [(2024, "Michigan", "Oakland", 1.0, 0.1, 90.097, 88.0)]
    )
    df = spreadsheet([("Michigan", "Oakland", 2024, "1", "0", None, "88")])
    fill = [("Michigan", ["Oakland"], 2024, ["MMR_cov"])]
    out = vw.fix_known_values(df, data, replacements=fill, known_values=[])
    assert out["MMR_cov"].to_list() == ["90.097"]
    # a county the spreadsheet lacks, a county the file lacks
    for county in ("Wayne", "Oakland"):
        bad_df = spreadsheet(
            [("Michigan", county, 2024, "1", "0", None, "88")]
        )
        bad = [("Michigan", ["Oakland", "Wayne"], 2024, ["MMR_cov"])]
        with pytest.raises(ValueError, match="not in the spreadsheet or in"):
            vw.fix_known_values(
                bad_df, data, replacements=bad, known_values=[]
            )


def test_known_values_are_set_and_a_missing_county_is_an_error(tmp_path):
    df = spreadsheet(
        [
            ("South Dakota", "Lyman", 2017, "0", "0", "1.02", None),
            ("South Dakota", "Lyman", 2018, "3.3", "0", "97", None),
        ]
    )
    out = vw.fix_known_values(
        df,
        str(tmp_path),
        replacements=[],
        known_values=[("South Dakota", "Lyman", 2017, "MMR_cov", 100.0)],
    )
    assert out["MMR_cov"].to_list() == ["100.0", "97"]
    with pytest.raises(ValueError, match="matches no row"):
        vw.fix_known_values(
            df,
            str(tmp_path),
            replacements=[],
            known_values=[("South Dakota", "Lymann", 2017, "MMR_cov", 100.0)],
        )


def test_a_county_that_is_excluded_stays_excluded_after_the_fix(tmp_path):
    data = nov9(
        tmp_path, [(2024, "Virginia", "Warren", 30.769, 0.0, 69.23, 69.23)]
    )
    df = spreadsheet([("Virginia", "Warren", 2024, None, None, None, None)])
    fixed = vw.fix_known_values(
        df,
        data,
        replacements=[("Virginia", None, 2024, ["MMR_cov"])],
        known_values=[],
    )
    assert fixed["MMR_cov"].to_list() == ["69.23"]
    out = vw.apply_qc_exclusions(
        fixed, [{"state": "Virginia", "county": "Warren", "year": 2024}]
    )
    assert out.height == 0


def test_the_known_values_and_replacements_are_well_formed():
    cells = [(s, c, y, col) for s, c, y, col, _ in vw.KNOWN_VALUES]
    assert len(cells) == len(set(cells))
    values = {(c, col): v for _, c, _, col, v in vw.KNOWN_VALUES}
    # (measles exempt - measles medical) / enrolment of the raw county counts
    assert values[("Pierce", "nonmed_exempt_pct")] == pytest.approx(
        4.3273, abs=1e-4
    )
    assert values[("Pierce", "med_exempt_pct")] == pytest.approx(
        0.2010, abs=1e-4
    )
    assert values[("Lyman", "MMR_cov")] == 100.0
    for _, _, _, columns in vw.NOV9_REPLACEMENTS:
        assert set(columns) <= set(vw._NOV9_COLUMN)


def test_warren_2024_is_a_whole_row_qc_exclusion():
    path = os.path.join(os.path.dirname(__file__), "..", "params.yml")
    exclusions = yaml.safe_load(open(path))["qc_exclusions"]
    warren = [e for e in exclusions if e["county"] == "Warren"]
    assert warren == [{"state": "Virginia", "county": "Warren", "year": 2024}]
