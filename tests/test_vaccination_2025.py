"""The 2025 rows added to the compiled vaccination spreadsheet
(vaccination_wrangling.add_2025_vaccination_rows)."""

import polars as pl
import pytest

from outbreak_size import vaccination_wrangling as vw

HEADER = (
    "Year,State,County,Non-medical exemptions (%),Medical exemptions (%),"
    "MMR vaccine coverage,Up-to-date vaccination (all vaccines),FIPS,"
    "county_type,state_ap"
)
ROWS = [
    # an ordinary state: suppressed values are missing
    "2025,Ohio,Adams,1.5,*,95.5,90.0,39001,County,Ohio",
    "2025,Ohio,Allen,#DIV/0!,0.2,N/A*,,39003,County,Ohio",
    # Alaska's codes have a leading zero here, and a state row has no county
    "2025,Alaska,Aleutians,0.1,0.0,80.0,,020200,Economic region,Alaska",
    "2025,Alaska,,,,76.0,,02,State,Alaska",
    # New York is replaced by New York.xlsx, so it is not added
    "2025,New York,Albany,1.0,0.1,96.0,,36001,County,N.Y.",
    # West Virginia has exemptions in the spreadsheet, and only the state's
    # coverage here
    "2025,West Virginia,Barbour,0.9,0.9,,,54001,County,W. Va.",
    "2025,West Virginia,,,,98.9,91.0,54,State,W. Va.",
    # another year is not added
    "2024,Ohio,Adams,1.0,0.0,94.0,89.0,39001,County,Ohio",
]


@pytest.fixture
def spreadsheet_and_dir(tmp_path):
    (tmp_path / vw.VACCINATION_2025_FILE).write_text(
        HEADER + "\n" + "\n".join(ROWS) + "\n"
    )
    string = pl.String
    spreadsheet = pl.DataFrame(
        {
            "year": [2024, 2025],
            "state": ["Ohio", "West Virginia"],
            "county": ["Adams", "Barbour"],
            "nonmed_exempt_pct": ["1.0", "0.11"],
            "med_exempt_pct": ["0", "0"],
            "MMR_cov": ["94.0", None],
            "all_vacc_cov": [None, None],
            "FIPS": ["39001", "54001"],
            "county_type": ["County", "County"],
            "state_ap": ["Ohio", "W. Va."],
        },
        schema_overrides={c: string for c in ["MMR_cov", "all_vacc_cov"]},
    )
    return spreadsheet, str(tmp_path)


def test_the_new_rows_have_the_spreadsheets_columns_and_types(
    spreadsheet_and_dir,
):
    spreadsheet, data_dir = spreadsheet_and_dir
    out = vw.add_2025_vaccination_rows(spreadsheet, data_dir)
    assert out.schema == spreadsheet.schema


def test_suppressed_values_are_missing_and_new_york_is_left_out(
    spreadsheet_and_dir,
):
    spreadsheet, data_dir = spreadsheet_and_dir
    out = vw.add_2025_vaccination_rows(spreadsheet, data_dir)
    ohio = out.filter((pl.col("state") == "Ohio") & (pl.col("year") == 2025))
    assert ohio.sort("county")["med_exempt_pct"].to_list() == [None, "0.2"]
    assert ohio.sort("county")["nonmed_exempt_pct"].to_list() == ["1.5", None]
    assert ohio.sort("county")["MMR_cov"].to_list() == ["95.5", None]
    assert out.filter(pl.col("state") == "New York").height == 0
    assert out.filter(pl.col("year") == 2024).height == 1  # not re-added


def test_alaska_codes_lose_the_leading_zero_and_state_rows_are_named_state(
    spreadsheet_and_dir,
):
    spreadsheet, data_dir = spreadsheet_and_dir
    alaska = vw.add_2025_vaccination_rows(spreadsheet, data_dir).filter(
        pl.col("state") == "Alaska"
    )
    assert sorted(alaska["FIPS"].to_list()) == ["02", "20200"]
    state_row = alaska.filter(pl.col("FIPS") == "02")
    assert state_row["county"].to_list() == ["State"]
    assert state_row["county_type"].to_list() == ["State"]


def test_a_state_with_2025_rows_keeps_them_and_only_its_state_row_is_added(
    spreadsheet_and_dir,
):
    spreadsheet, data_dir = spreadsheet_and_dir
    west_virginia = vw.add_2025_vaccination_rows(spreadsheet, data_dir).filter(
        pl.col("state") == "West Virginia"
    )
    assert west_virginia.height == 2  # the spreadsheet's county, the state row
    county = west_virginia.filter(pl.col("county") == "Barbour").row(
        0, named=True
    )
    assert county["nonmed_exempt_pct"] == "0.11"  # the spreadsheet's own value
    assert county["MMR_cov"] is None  # not copied: the State row fill gives it
    state = west_virginia.filter(pl.col("county") == "State").row(
        0, named=True
    )
    assert state["MMR_cov"] == "98.9"
    assert state["FIPS"] == "54"


# Tennessee's 2025 rows (vaccination_wrangling.tennessee_2025_rows)
TN_COLUMNS = [
    "year",
    "state",
    "county",
    "utd_pct",
    "mmr_pct",
    "nme_pct",
    "me_pct",
]


def tennessee_file(rows):
    """The sheet "data" as read: every column text."""
    return pl.DataFrame(
        rows, schema={c: pl.String for c in TN_COLUMNS}, orient="row"
    )


def tennessee_spreadsheet():
    return pl.DataFrame(
        {
            "state": ["Tennessee"] * 4 + ["Ohio"],
            "county": ["Dekalb", "Knox", "Van Buren", "Knox", "Adams"],
            # an early row of a county has no FIPS
            "FIPS": ["47041", None, "47175", "47093", "39001"],
            "county_type": ["County"] * 5,
            "state_ap": ["Tenn.", "Tenn.", "Tenn.", "Tenn.", "Ohio"],
        }
    )


def test_tennessee_rows_take_the_spreadsheets_spelling_and_fips():
    raw = tennessee_file(
        [
            ["2024", "Tennessee", "Knox", "90.0", "91.0", "1.0", "0.1"],
            ["2025", "Tennessee", "DeKalb", "92.5", "94.0", "4.5", "0.2"],
            ["2025", "Tennessee", "Knox", "93.0", "95.0", "3.0", "0.0"],
            # a county without data stays missing
            ["2025", "Tennessee", "Van Buren", None, None, None, None],
        ]
    )
    out = vw.tennessee_2025_rows(raw, tennessee_spreadsheet())
    assert out.columns == [
        "year",
        "state",
        "county",
        "nonmed_exempt_pct",
        "med_exempt_pct",
        "MMR_cov",
        "all_vacc_cov",
        "FIPS",
        "county_type",
        "state_ap",
    ]
    assert out["county"].to_list() == ["Dekalb", "Knox", "Van Buren"]
    assert out["FIPS"].to_list() == ["47041", "47093", "47175"]
    assert out["year"].to_list() == [2025] * 3  # 2024 is not used
    dekalb = out.row(0, named=True)
    assert (dekalb["MMR_cov"], dekalb["nonmed_exempt_pct"]) == (94.0, 4.5)
    assert (dekalb["med_exempt_pct"], dekalb["all_vacc_cov"]) == (0.2, 92.5)
    assert out.row(2, named=True)["MMR_cov"] is None


def test_a_tennessee_county_that_is_not_in_the_spreadsheet_is_refused():
    raw = tennessee_file(
        [["2025", "Tennessee", "Atlantis", "92.5", "94.0", "4.5", "0.2"]]
    )
    with pytest.raises(ValueError, match="atlantis"):
        vw.tennessee_2025_rows(raw, tennessee_spreadsheet())


def test_a_tennessee_county_twice_in_2025_is_refused():
    raw = tennessee_file(
        [
            ["2025", "Tennessee", "Knox", "93.0", "95.0", "3.0", "0.0"],
            ["2025", "Tennessee", "KNOX", "93.0", "95.0", "3.0", "0.0"],
        ]
    )
    with pytest.raises(ValueError):
        vw.tennessee_2025_rows(raw, tennessee_spreadsheet())
