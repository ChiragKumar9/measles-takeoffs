"""Tests of the readers of the state files (outbreak_size.school_state_files)."""

import polars as pl
import pytest

from outbreak_size import school_state_files as ssf
from outbreak_size.school_vaccination import QCLog


def sheet(rows):
    """A headerless frame of text/numbers, as read_sheet returns."""
    width = max(len(r) for r in rows)
    padded = [list(r) + [None] * (width - len(r)) for r in rows]
    return pl.DataFrame(
        padded,
        orient="row",
        schema=[f"column_{i}" for i in range(width)],
        strict=False,
        infer_schema_length=None,
    )


# --- bands -----------------------------------------------------------------------


def test_band_midpoints_the_rule_of_the_request():
    # ">95" with a ">96" option: 95 to 96 -> 95.5; the highest band runs to 100
    assert ssf.band_midpoints([">95%", ">96%"]) == {">95%": 95.5, ">96%": 98.0}
    # ">95" alone: 95 to 100 -> 97.5
    assert ssf.band_midpoints([">95%"]) == {">95%": 97.5}


def test_band_midpoints_california_and_low_bands():
    assert ssf.band_midpoints(["≥95%", "≥98%", "≥99%", "≥95%"]) == {
        "≥95%": 96.5,
        "≥98%": 98.5,
        "≥99%": 99.5,
    }
    assert ssf.band_midpoints(["≤1%", "≤5%"]) == {"≤1%": 0.5, "≤5%": 3.0}
    assert ssf.band_midpoints(["≤5%"]) == {"≤5%": 2.5}
    # the same threshold written two ways is one band
    assert ssf.band_midpoints(["≥95%", ">95%", "≥98%"]) == {
        "≥95%": 96.5,
        ">95%": 96.5,
        "≥98%": 99.0,
    }
    # a text that is not a band is left out
    assert ssf.band_midpoints(["97", "**", None]) == {}


def test_parse_rates_numbers_bands_and_suppressed():
    rates, bands = ssf.parse_rates(
        ["97.3%", 94, " 22", "≥95%", "**", "--*", ".", None, ">96%"]
    )
    assert rates == [97.3, 94.0, 22.0, None, None, None, None, None, None]
    assert bands == [None, None, None, "≥95%", None, None, None, None, ">96%"]


def test_scale_factor():
    assert ssf.scale_factor([1.0, 0.97, None, 0.5]) == 100.0
    assert ssf.scale_factor([97.0, 100.0, 80.0]) == 1.0
    assert ssf.scale_factor([None, None]) == 1.0


# --- counties ----------------------------------------------------------------------


def county_frame():
    return pl.DataFrame(
        {
            "county_fips": [
                "24005",
                "24510",
                "24033",
                "24037",
                "36061",
                "36047",
                "36089",
            ],
            "county_name": [
                "Baltimore",
                "Baltimore",
                "Prince George's",
                "St. Mary's",
                "New York",
                "Kings",
                "St. Lawrence",
            ],
            "county_state": ["MD", "MD", "MD", "MD", "NY", "NY", "NY"],
        }
    )


def test_county_names_baltimore_city_and_county_are_told_apart():
    found = ssf.map_counties(
        [
            "Baltimore City",
            "Baltimore County",
            "Baltimore",
            "Prince George's",
            "Saint Mary's",
        ],
        "MD",
        county_frame(),
    )
    assert found["Baltimore City"][0] == "24510"
    assert found["Baltimore County"][0] == "24005"
    assert found["Baltimore"][0] == "24005"
    assert found["Prince George's"][0] == "24033"
    assert found["Saint Mary's"][0] == "24037"


def test_new_york_names_and_unknown_names():
    found = ssf.map_counties(
        ["NEW YORK", "Kings", "ST LAWRENCE"], "NY", county_frame()
    )
    assert [found[n][0] for n in ["NEW YORK", "Kings", "ST LAWRENCE"]] == [
        "36061",
        "36047",
        "36089",
    ]
    with pytest.raises(ValueError, match="Gotham"):
        ssf.map_counties(["Kings", "Gotham"], "NY", county_frame())


def test_a_label_of_two_counties_has_no_county():
    df = pl.DataFrame({"county_label": ["Kings", "Kings/New York"]})
    out = ssf.attach_counties(df, "NY", county_frame())
    assert out["county_fips"].to_list() == ["36047", None]


# --- Maryland ------------------------------------------------------------------------

MD_NEW = [
    [
        "Percent of Kindergarteners Vaccinated by School",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "To protect privacy, schools ...",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "School Name",
        "County",
        "Type of School",
        "Total Number of Enrolled Students",
        "% Medical Exemption",
        "% Religious Exemption",
        "% DTaP",
        "% Polio",
        "% MMR",
        "% Hep B",
        "% Varicella",
    ],
    ["Beall Elementary", "Allegany", "Public", 50, 2, 4, 94, 94, 94, 94, 94],
    [
        "Tiny Montessori",
        "Allegany",
        "Private",
        "<10",
        "**",
        "**",
        "**",
        "**",
        "**",
        "**",
        "**",
    ],
    [
        "Bel Air Elementary",
        "Allegany",
        "Public",
        34,
        0,
        2.9,
        97.1,
        97.1,
        97.1,
        97.1,
        97.1,
    ],
    ["Total", None, None, 84, None, None, None, None, None, None, None],
]
MD_OLD = [
    [
        "Percent of Kindergarteners Vaccinated by School",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "a",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "b",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "c",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "School Name",
        "County",
        "Type of School ",
        "Total K Students",
        "Total K Students WITH",
        "Total K Students WITHOUT",
        "Total K Students SURVEYED",
        "% Medical Exemption",
        "% Religious Exemption",
        "% DTaP",
        "% Polio",
        "% MMR",
        "% Hep B",
        "% Varicella",
    ],
    [
        "BEALL ELEMENTARY",
        "Allegany",
        "Public",
        74,
        74,
        0,
        74,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ],
    [
        "BEL AIR ELEMENTARY",
        "Allegany",
        "Public",
        38,
        37,
        0,
        37,
        0,
        0.026315789,
        1,
        1,
        0.973,
        1,
        1,
    ],
]


def test_maryland_percent_file_and_censoring():
    log = QCLog()
    out, info = ssf.parse_maryland(sheet(MD_NEW), 2024, log)
    assert info["scale"] == "percent" and info["dropped_totals"] == 1
    assert out["school_name"].to_list() == [
        "Beall Elementary",
        "Tiny Montessori",
        "Bel Air Elementary",
    ]
    assert out["enrollment"].to_list() == [
        50.0,
        None,
        34.0,
    ]  # "<10" is a missing enrollment
    assert out["rate"].to_list() == [
        94.0,
        None,
        97.1,
    ]  # "**" is a missing rate
    assert out["exemption_religious"].to_list() == [4.0, None, 2.9]
    assert out["school_type"].to_list() == ["PUBLIC", "PRIVATE", "PUBLIC"]
    reasons = log.rows_frame()["reason"].to_list()
    assert any("enrollment censored" in r for r in reasons) and any(
        "suppressed" in r for r in reasons
    )


def test_maryland_fraction_file_is_multiplied_and_header_may_move():
    out, info = ssf.parse_maryland(sheet(MD_OLD), 2015)
    assert info["scale"] == "fraction x100"
    assert out["rate"].to_list() == pytest.approx([100.0, 97.3])
    assert out["exemption_religious"].to_list() == pytest.approx(
        [0.0, 2.6315789]
    )
    assert out["enrollment"].to_list() == [74.0, 38.0]


# --- New York --------------------------------------------------------------------------


def ny_table():
    return pl.DataFrame(
        {
            "School ID": ["1", "2", "3", "4", "5"],
            "Report Period": ["2023-2024"] * 5,
            "Type": [
                "Public School",
                "School District",
                "Private School",
                "Public School",
                "BOCES School",
            ],
            "School Name": ["A HS", "A CSD", "B SCHOOL", None, "C BOCES"],
            "Percent Medical Exemptions": [
                "0.00%",
                "0.10%",
                "0.00%",
                None,
                "1.00%",
            ],
            "Percent Immunized Measles": [
                "100.00%",
                "99.00%",
                "0%",
                None,
                "97.5%",
            ],
            "Street": ["1 Main St", None, "2 Oak St", None, "3 Elm"],
            "County": ["Kings", "Kings", "NEW YORK", "Kings", "Kings"],
            "Location": ["POINT (-73.9 40.7)", None, None, None, None],
        }
    )


def test_new_york_drops_district_totals_and_nameless_rows():
    log = QCLog()
    out, info = ssf.parse_new_york(ny_table(), log)
    assert (
        info["district_rows_dropped"] == 1
        and info["nameless_rows_dropped"] == 1
    )
    assert out["school_name"].to_list() == ["A HS", "B SCHOOL", "C BOCES"]
    assert out["rate"].to_list() == [100.0, 0.0, 97.5]
    assert out["school_type"].to_list() == [
        "PUBLIC",
        "PRIVATE",
        None,
    ]  # BOCES has no public/private type
    assert out["lon"].to_list() == [-73.9, None, None]
    assert (
        info["rate_below_10"] == 1
    )  # the 0% private school the placeholder rule would remove
    assert (
        out["enrollment"].null_count() == out.height
    )  # the survey has no enrollment


# --- California -------------------------------------------------------------------------

CA_TWO = [
    [
        "2019-2020 IMMUNIZATION",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "Schools with 20 or more",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    [
        "SCHOOL CODE",
        "COUNTY",
        "PUB/  PRIV",
        "PUBLIC SCHOOL DISTRICT",
        "CITY",
        "SCHOOL NAME",
        "ENROLL- MENT",
        "MMR‡",
        None,
        "REPORTED¶¶",
        None,
    ],
    [None, None, None, None, None, None, None, "#", "%", None, None],
    [
        "01",
        "ALAMEDA",
        "PUBLIC",
        "D",
        "OAKLAND",
        "SCHOOL ONE",
        59,
        55,
        93,
        "Y",
        None,
    ],
    [
        "02",
        "ALAMEDA",
        "PRIVATE",
        "D",
        "OAKLAND",
        "SCHOOL TWO",
        30,
        None,
        "≥95%",
        "Y",
        None,
    ],
    [
        "03",
        "ALAMEDA",
        "PUBLIC",
        "D",
        "OAKLAND",
        "SCHOOL THREE",
        31,
        None,
        "≥98%",
        "Y",
        None,
    ],
    [
        "04",
        "ALAMEDA",
        "PUBLIC",
        "D",
        "OAKLAND",
        "SCHOOL FOUR",
        None,
        ".",
        ".",
        "N",
        None,
    ],
]
CA_ONE = [
    ["2022-2023 Immunization", None, None, None, None, None, None, None, None],
    [
        "School Code",
        "Grade",
        "County",
        "Pub/ Priv",
        "Public School District",
        "City",
        "School Name",
        "Enrollment",
        "MMR‡ Percent",
    ],
    [
        "01",
        "Kindergarten",
        "Alameda",
        "Public",
        "D",
        "Hayward",
        "School One",
        24,
        "≥95%",
    ],
    [
        "01",
        "1st Grade",
        "Alameda",
        "Public",
        "D",
        "Hayward",
        "School One",
        26,
        "92",
    ],
    [
        "02",
        "Kindergarten",
        "Alameda",
        "Public",
        "D",
        "Oakland",
        "School Two",
        22,
        "≤5%",
    ],
    [
        "03",
        "Kindergarten",
        "Alameda",
        "Private",
        "D",
        "Oakland",
        "School Three",
        25,
        "≤1%",
    ],
    [
        "04",
        "Kindergarten",
        "Alameda",
        "Public",
        "D",
        "Oakland",
        "School Four",
        40,
        "≥99%",
    ],
]


def test_california_two_row_header_numbers_bands_and_missing():
    out, info = ssf.parse_california(sheet(CA_TWO), 2019, QCLog())
    assert info["layout"] == "two header rows" and info["rows"] == 4
    assert out["rate"].to_list() == [
        93.0,
        96.5,
        99.0,
        None,
    ]  # this file has only >=95 (95 to 98) and >=98 (98 to 100)
    assert out["rate_source"].to_list() == [
        "mmr",
        "mmr",
        "mmr",
        None,
    ]
    assert out["band"].to_list() == [None, "≥95%", "≥98%", None]
    assert out["school_type"].to_list() == [
        "PUBLIC",
        "PRIVATE",
        "PUBLIC",
        "PUBLIC",
    ]


def test_california_one_row_header_keeps_kindergarten_only():
    out, info = ssf.parse_california(sheet(CA_ONE), 2022)
    assert info["other_grade_rows_dropped"] == 1 and out.height == 4
    assert out["school_name"].to_list() == [
        "School One",
        "School Two",
        "School Three",
        "School Four",
    ]
    # bands of the kindergarten rows: >=95 and >=99 (95 to 99 -> 97, 99 to 100 -> 99.5), <=1 and <=5 (0.5, 3.0)
    assert out["rate"].to_list() == [97.0, 3.0, 0.5, 99.5]


# --- South Carolina ---------------------------------------------------------------------

SC_TEXT = """South Carolina Department of Public Health 45-Day Report of Schools with Required Immunization Certification, 2024-2025 School Year
Region           County                 Type       School Name                                          City            Range        Students    Immunizations
Upstate          Abbeville              Private    Shiloh Christian School                              Due West        K5-12                 64             41%
Upstate          Abbeville              Public     John C. Calhoun Elementary                           Calhoun Falls   PK-5th               109            >96%
Midlands         Aiken                  Private    First Presbyterian Church Preschool                  Aiken           PK-K                 <10             88%
Pee Dee ChesterfieldPrivate   Faith Christian Academy                           Cheraw          3P-8th grade          78   >96%
Upstate    Spartanbur Public    Woodruff Elementary                         Woodruff          3-5th                 651   >96%
Midlands   Lexington/Richland   Public    Dutch Fork Elementary                                           Irmo              PK-5th      400    94%
Pee Dee    Chester      Public    Some School With A Strange Layout and no numbers at all
Lowcountry   Bamberg    Public    Denmark-Olar High                                    Denmark              9-12th              171   <5%
"""


def test_south_carolina_lines():
    df, unparsed = ssf.parse_south_carolina(SC_TEXT, QCLog())
    assert df["county_label"].to_list() == [
        "Abbeville",
        "Abbeville",
        "Aiken",
        "Chesterfield",
        "Spartanburg",
        "Lexington/Richland",
        "Bamberg",
    ]
    assert df["school_name"].to_list()[:3] == [
        "Shiloh Christian School",
        "John C. Calhoun Elementary",
        "First Presbyterian Church Preschool",
    ]
    assert df["enrollment"].to_list() == [
        64.0,
        109.0,
        None,
        78.0,
        651.0,
        400.0,
        171.0,
    ]  # "<10" is a missing enrollment
    # ">96%" is 98 (no higher band); "<5%" is 2.5 (no lower band); the rate of the "<10" school is kept
    assert df["rate"].to_list() == [41.0, 98.0, 88.0, 98.0, 98.0, 94.0, 2.5]
    assert df["rate_source"].unique().to_list() == ["overall"]
    assert len(unparsed) == 1 and unparsed[0].startswith("Pee Dee    Chester")


@pytest.mark.parametrize(
    "grades,expected",
    [
        ("K5-12", True), ("K-12", True), ("PK-5th", True), ("Infants - K", True), ("3K-5th grade", True), ("PK-K", True),
        ("1-12th", False), ("6-12th", False), ("9-12th", False), ("3-5th", False), ("PK", False), ("3K", False), ("Pre-K", False),
    ],
)  # fmt: skip
def test_includes_kindergarten(grades, expected):
    assert ssf.includes_kindergarten(grades) is expected


def test_maryland_negative_rate_is_missing_not_zero():
    rows = [r for r in MD_NEW]
    rows.append(
        ["Bad School", "Allegany", "Public", 12, 0, 0, 1, 1, -583.3, 1, 1]
    )
    out, info = ssf.parse_maryland(sheet(rows), 2024)
    assert info["rate_negative"] == 1
    assert out.filter(pl.col("school_name") == "Bad School")[
        "rate"
    ].to_list() == [None]


# --- Ohio ---------------------------------------------------------------------------


def test_ohio_count_layout_computes_the_rate_and_masks_text_cells():
    frame = sheet(
        [
            [
                "School Name",
                "School IRN",
                "City",
                "County",
                "Total Pupils Enrolled In Grade Reported (K)",
                "Calculated Specific Antigen Up-to-Date (UTD): MMR",
            ],
            ["Adams Elementary", "000001", "Columbus", "Franklin", 50, 45],
            [
                "Tiny School",
                "000002",
                "Dayton",
                "Montgomery",
                8,
                "All data that represents less than 10 students are masked",
            ],
            ["Over School", "000003", "Akron", "Summit", 20, 22],
            ["Ohio Totals", "Ohio", None, None, 100000, 90000],
            [None, None, None, None, None, None],
        ]
    )
    out, info = ssf.parse_ohio(frame, 2022)
    assert out["school_name"].to_list() == [
        "Adams Elementary",
        "Tiny School",
        "Over School",
    ]
    assert out["rate"].to_list() == [90.0, None, pytest.approx(110.0)]
    assert out["enrollment"].to_list() == [50.0, 8.0, 20.0]
    assert out["rate_source"].unique().to_list() == ["mmr"]
    assert (out["year_start"] == 2022).all()
    assert info["layout"] == "count" and info["dropped_totals"] == 2


def test_ohio_percent_layout_is_a_fraction_and_the_header_may_move():
    frame = sheet(
        [
            ["Table 2: Kindergarten", None, None, None, None, None],
            [None] * 6,
            [
                "School Name",
                "School IRN",
                "City",
                "County Name",
                "Total Pupils Enrolled In Grade Reported (K)",
                "% Specific Antigen Up-to-Date (UTD): MMR",
            ],
            ["A", "1", "X", "Lucas", 30, 0.95],
            ["B", "2", "Y", "Lorain", 40, "All data masked"],
        ]
    )
    out, info = ssf.parse_ohio(frame, 2019)
    assert out["rate"].to_list() == [pytest.approx(95.0), None]
    assert info["layout"] == "percent"


def test_ohio_2015_flat_file_keeps_kindergarten_rows_and_the_overall_rate():
    frame = sheet(
        [
            [
                "SchoolName",
                "SchoolIRN",
                "City",
                "County",
                "Type",
                "PupilsTotal",
                "PupilsWithReqImmu",
            ],
            ["K School", "1", "X", "Lucas", "Kindergarten", 20, 19],
            ["K School", "1", "X", "Lucas", "Seventh", 25, 24],
        ]
    )
    out, info = ssf.parse_ohio(frame, 2015)
    assert out.height == 1 and out["rate"].to_list() == [95.0]
    assert out["rate_source"].to_list() == ["overall"]
    assert info["other_grade_rows"] == 1


def test_ohio_without_a_school_name_header_is_an_error():
    with pytest.raises(ValueError, match="School Name"):
        ssf.parse_ohio(sheet([["a", "b"], [1, 2]]), 2020)


def test_a_state_file_enrollment_names_its_source():
    rows = pl.DataFrame(
        {
            "school_name": ["a", "b"],
            "enrollment": [20.0, None],
            "rate": [95.0, 90.0],
            "year_start": [2022, 2022],
            "county_label": ["x", "x"],
        }
    )
    out = ssf.finish(rows, "OH").sort("rid")
    assert out["enrollment_source"].to_list() == ["state_file", None]


# --- Iowa ----------------------------------------------------------------------------

IA_HEADER_TEXT = """\
            IOWA DEPARTMENT OF HEALTH AND HUMAN SERVICES- IMMUNIZATION PROGRAM
                    IMMUNIZATION AUDIT {title} SCHOOL YEAR
            KINDERGARTEN - 12TH GRADE SUMMARY REPORT BY SCHOOL

                                                                    Percent
County Name   District        School Name        Certificate of   Total   Vaccinated
              Number                             Immunization   Enrollment  Vaccines
                                                                    Required by
                                                                       Law*
"""
IA_FOOTER_TEXT = """
Jan 15, 2025                                                       Page 1 of 1
Percent Vaccinated with Vaccines Required by Law is calculated by dividing the number of Valid
Certificates of Immunization by Total Enrollment.
*Schools with a Total Enrollment of 100 or less are suppressed and marked with a S.
"""
IA_PAD = " " * 26  # name column


def ia_doc(rows, title="2024-25"):
    return (
        IA_HEADER_TEXT.format(title=title)
        + "\n".join(rows)
        + "\n"
        + IA_FOOTER_TEXT
    )


def ia_names(out):
    return out["school_name"].to_list()


# --- layout A: one line per school (2021-22 to 2023-24) ---------------------------


def test_one_line_rows_with_commas_and_the_year_from_the_title():
    text = ia_doc(
        [
            "Adair          2673      Nodaway Valley High School           196         205         95.6",
            "Polk            261      Ankeny High School                 1,265       1,402         90.2",
        ],
        title="2022-23",
    )
    out, info = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "Nodaway Valley High School",
        "Ankeny High School",
    ]
    assert out["county_label"].to_list() == ["Adair", "Polk"]
    assert out["enrollment"].to_list() == [205.0, 1402.0]
    assert out["rate"].to_list() == [95.6, 90.2]
    assert (out["year_start"] == 2022).all() and info["year_start"] == 2022
    assert out["rate_source"].unique().to_list() == ["overall"]
    assert out["band"].null_count() == 2 and info["scale"] == "percent"


def test_multiword_county_and_footer_lines_are_not_rows():
    out, info = ssf.parse_iowa(
        ia_doc(
            [
                "Cerro Gordo    4131      Clear Lake High School               397         417         95.2"
            ]
        )
    )
    assert out["county_label"].to_list() == ["Cerro Gordo"] and out.height == 1
    assert len(info["footnotes"]) == 3  # the three footnote lines at column 0


def test_year_from_title_must_agree_with_the_year_given():
    text = ia_doc(["Adair 2673 A School 10 11 90.0"], title="2023-24")
    assert ssf.parse_iowa(text, year_start=2023)[1]["year_start"] == 2023
    with pytest.raises(ValueError, match="title says 2023"):
        ssf.parse_iowa(text, year_start=2022)
    with pytest.raises(ValueError, match="no 'IMMUNIZATION AUDIT"):
        ssf.parse_iowa("nothing here")


# --- layout A2: first line of a long name above the data line (2021-22) -----------


def test_name_head_above_a_data_line_that_carries_the_end_of_the_name():
    text = ia_doc(
        [
            "Butler         6840      Shell Rock Elementary School          145         147         98.6",
            IA_PAD + "Manson Northwest Webster Junior High/High",
            "Calhoun        4023      School                                354         370         95.7",
            "Calhoun        5325      Prairie Valley Elementary School      179         191         93.7",
        ],
        title="2021-22",
    )
    out, info = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "Shell Rock Elementary School",
        "Manson Northwest Webster Junior High/High School",
        "Prairie Valley Elementary School",
    ]
    assert out["year_start"].unique().to_list() == [2021]


# --- layout B: the name wrapped around a nameless data line (2024-25, 2025-26) ----


def test_wrapped_name_above_and_below_the_data_line():
    text = ia_doc(
        [
            IA_PAD + "Nodaway Valley Elementary",
            "Adair           2673                                       275         285         96.5",
            IA_PAD + "School",
            "Adair           2673      Nodaway Valley High School       205         223         91.9",
            IA_PAD + "Lakeview Elementary-",
            "Appanoose       1071                                       452         499         90.6",
            IA_PAD + "Centerville",
        ]
    )
    out, info = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "Nodaway Valley Elementary School",
        "Nodaway Valley High School",
        "Lakeview Elementary-Centerville",
    ]  # no space after a hyphen
    assert out["rate"].to_list() == [96.5, 91.9, 90.6]
    assert info["wrapped_names"] == 2


def test_two_wrapped_rows_in_a_row_share_the_lines_between_them():
    text = ia_doc(
        [
            IA_PAD + "Cora B Darling Elementary",
            "Allamakee       5310                                       261         276         94.6",
            IA_PAD + "School",
            IA_PAD + "East Campus Elementary",
            "Allamakee        135                                       217         227         95.6",
            IA_PAD + "School",
        ]
    )
    out, _ = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "Cora B Darling Elementary School",
        "East Campus Elementary School",
    ]


def test_one_line_between_two_nameless_rows_is_the_tail_of_the_first():
    # Buena Vista 2025-26: "Albert City-Truesdale" above, "Elementary School" below the first row;
    # the second row (a district entry) has its name only below
    text = ia_doc(
        [
            IA_PAD + "Albert City-Truesdale",
            "Buena Vista      72                                          S          73           S",
            IA_PAD + "Elementary School",
            "Buena Vista     171                                          S           1           S",
            IA_PAD + "Alta Community School District",
            "Buena Vista     171      Alta Elementary School            195         208         93.8",
        ]
    )
    out, _ = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "Albert City-Truesdale Elementary School",
        "Alta Community School District",
        "Alta Elementary School",
    ]
    assert out["unit"].to_list() == ["school", "district", "school"]


def test_name_lines_are_attached_within_each_page():
    page = (
        lambda rows: IA_HEADER_TEXT.format(title="2024-25")
        + "\n".join(rows)
        + "\n\nJan 15, 2025        Page 1 of 2\n"
    )  # noqa: E731
    text = (
        page(
            [
                IA_PAD + "First head",
                "Adair           2673                                       10         11         90.0",
                IA_PAD + "first tail",
            ]
        )
        + "\f"
        + page(
            [
                IA_PAD + "Second head",
                "Adams           1431                                       170         175         97.1",
                IA_PAD + "second tail",
            ]
        )
    )
    out, _ = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "First head first tail",
        "Second head second tail",
    ]


# --- layout C: county, district and name on one line, the numbers on the next -----


def test_numbers_on_the_line_after_the_name():
    text = ia_doc(
        [
            "Pottawattamie    1476      Franklin Elementary School-Council Bluffs",
            "                                                                  366         383         95.6",
            "Pottawattamie    1476      Gerald W Kirn Middle School          885         936         94.6",
        ]
    )
    out, _ = ssf.parse_iowa(text)
    assert ia_names(out) == [
        "Franklin Elementary School-Council Bluffs",
        "Gerald W Kirn Middle School",
    ]
    assert out["enrollment"].to_list() == [383.0, 936.0] and out[
        "rate"
    ].to_list() == [95.6, 94.6]


# --- censoring, totals, scale, log ------------------------------------------------


def test_suppressed_cells_are_missing_rows_are_kept_and_logged():
    text = ia_doc(
        [
            "Adair          4978      Orient Elementary School               S          39           S",
            "Boone           100      Madrid Comm School District            S           S          50.0",
            "Boone           100      Boone High School                    300         320         93.8",
        ]
    )
    log = QCLog()
    out, info = ssf.parse_iowa(text, log=log)
    assert out.height == 3
    assert out["rate"].to_list() == [
        None,
        50.0,
        93.8,
    ]  # a rate given next to S counts is kept
    assert out["enrollment"].to_list() == [
        39.0,
        None,
        320.0,
    ]  # S total: enrollment missing
    assert info["rate_missing"] == 1 and info["enrollment_missing"] == 1
    reasons = (
        log.rows_frame()
        .group_by("reason")
        .len()
        .sort("reason")["reason"]
        .to_list()
    )
    assert len(reasons) == 2 and all(
        r.startswith(("rate suppressed", "total enrollment")) for r in reasons
    )
    assert log.rows_frame()["state"].unique().to_list() == ["IA"]


def test_total_rows_are_dropped_and_logged_but_schools_named_like_iowa_are_kept():
    text = ia_doc(
        [
            "Adair          4978      Iowa Valley Elementary                 120         130         92.3",
            "Adair          4978      State Center Elementary                120         130         92.3",
            "Adair          4978      Total                                  240         260         92.3",
        ]
    )
    log = QCLog()
    out, info = ssf.parse_iowa(text, log=log)
    assert ia_names(out) == [
        "Iowa Valley Elementary",
        "State Center Elementary",
    ]
    assert info["dropped_totals"] == 1
    assert "total or statewide row" in log.rows_frame()["reason"].to_list()


def test_fractions_are_scaled_to_percent():
    out, info = ssf.parse_iowa(
        ia_doc(["Adair 1 A School 90 100 0.9", "Adair 1 B School 95 100 0.95"])
    )
    assert (
        out["rate"].to_list() == pytest.approx([90.0, 95.0])
        and info["scale"] == "fraction x100"
    )


def test_html_entities_of_a_bbox_text_are_unescaped():
    out, _ = ssf.parse_iowa(
        ia_doc(
            [
                "O&apos;Brien      1       St Mary&apos;s School                    50          60         83.3"
            ]
        )
    )
    assert out["county_label"].to_list() == ["O'Brien"] and ia_names(out) == [
        "St Mary's School"
    ]


def test_unexpected_layouts_are_errors():
    with pytest.raises(ValueError, match="one header end"):
        ssf.parse_iowa(
            "IMMUNIZATION AUDIT 2024-25 SCHOOL YEAR\nAdair 1 A School 1 2 50.0\n"
        )
    with pytest.raises(ValueError, match="belong to no row"):
        ssf.parse_iowa(
            ia_doc(
                [
                    "Adair 1 A School 10 11 90.0",
                    IA_PAD
                    + "stray line below a named row at the end of the page",
                ]
            )
        )


# --- counties --------------------------------------------------------------------


def test_the_misspelt_iowa_county_maps_to_pottawattamie():
    counties = pl.DataFrame(
        {
            "county_fips": ["19155", "19141"],
            "county_name": ["Pottawattamie", "O'Brien"],
            "county_state": ["IA", "IA"],
        }
    )
    found = ssf.map_counties(["Pottawattami", "O'Brien"], "IA", counties)
    assert (
        found["Pottawattami"][0] == "19155" and found["O'Brien"][0] == "19141"
    )


# --- Maine ---------------------------------------------------------------------------

# --- Excel -----------------------------------------------------------------------------

ME_HEADER_ROW = [
    "School",
    "County",
    "Assessed",
    "TotalExempt",
    "Medical",
    "Religious",
    "Philosophical",
    "90 Day",
    "Missing",
    "4DTaP",
    "3Polio",
    "2MMR",
    "2Var",
    "Assessed",
    "TotalExempt",
]


def me_excel_rows(
    totals=(100, 50, 50),
    title="2022-2023 Maine School Immunization Assessment",
    extra=(),
):
    """Cumberland: Alpha 50 (90%); Hancock: RSU 25 (a unit) 20 (85%); York: Middle school with no kindergarten, Gamma 30."""
    blank = [None] * 15
    return [
        [None, None, title],
        [None, None, "Updated 6/16/2023"],
        [None, None, "Kindergarten"] + [None] * 10 + ["Seventh Grade", None],
        [None, None, "Total", "Exemptions"],
        ME_HEADER_ROW,
        [
            "Alpha School",
            "Cumberland",
            50,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0.9,
            1,
            "-",
            "-",
        ],
        [
            "Cumberland County Totals",
            "Cumberland",
            totals[1],
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0.9,
            1,
            99,
            0,
        ],
        ["RSU 25", "Hancock", 20, 0, 0, 0, 0, 0, 0, 1, 1, 0.85, 1, "-", "-"],
        [
            "Hancock County Totals",
            "Hancock",
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0.85,
            1,
            1,
            0,
        ],
        [
            "Bay Middle School",
            "York",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            120,
            0,
        ],
        ["Gamma School", "York", 30, 0, 0, 0, 0, 0, 0, 1, 1, 0.8, 1, "-", "-"],
        *extra,
        [
            "York County Totals",
            "York",
            30,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0.8,
            1,
            120,
            0,
        ],
        ["Statewide", None, totals[0], 0, 0, 0, 0, 0, 0, 1, 1, 0.9, 1, 220, 0],
        blank,
        ["• Rates are percentages", None],
    ]


def test_excel_layout_scale_units_and_dropped_rows():
    log = QCLog()
    out, info = ssf.parse_me_excel(sheet(me_excel_rows()), 2022, log)
    assert out["school_name"].to_list() == [
        "Alpha School",
        "RSU 25",
        "Gamma School",
    ]
    assert out["county_label"].to_list() == ["Cumberland", "Hancock", "York"]
    assert out["enrollment"].to_list() == [50.0, 20.0, 30.0]
    assert out["rate"].to_list() == pytest.approx(
        [90.0, 85.0, 80.0]
    )  # fractions x 100
    assert out["rate_source"].unique().to_list() == ["mmr"]
    assert out["unit"].to_list() == [
        "school",
        "district",
        "school",
    ]  # "RSU 25" is only a district label
    assert (out["year_start"] == 2022).all()
    assert (
        info["scale"] == "fraction x100"
        and info["dropped_totals"] == 4
        and info["dropped_no_kindergarten"] == 1
    )
    assert info["dropped_footnotes"] == 1 and info["blank_rows"] == 1
    assert (
        info["printed_total"] == 100.0
        and info["total_check"].startswith("ok")
        and "3 county totals ok" in info["total_check"]
    )
    reasons = log.rows_frame()["reason"].to_list()
    assert sum("total row" in r for r in reasons) == 4 and any(
        "no kindergarten" in r for r in reasons
    )


def test_excel_percent_file_keeps_percent_and_reads_a_bad_rate_cell_as_missing():
    rows = me_excel_rows()
    for r in rows:  # percent instead of fractions
        if r[0] in ("Alpha School", "RSU 25", "Gamma School") or str(
            r[0]
        ).endswith("Totals"):
            r[11] = r[11] * 100
    rows[10][11] = "-"  # Gamma: rate cell is text
    out, info = ssf.parse_me_excel(sheet(rows), 2022)
    assert info["scale"] == "percent"
    assert out["rate"].to_list() == pytest.approx([90.0, 85.0, None])
    assert info["rate_missing"] == 1


def test_excel_censored_count_is_a_missing_enrollment_and_skips_the_total_check():
    tiny = [
        "Tiny School",
        "York",
        "<10",
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0.95,
        1,
        "-",
        "-",
    ]
    log = QCLog()
    out, info = ssf.parse_me_excel(
        sheet(me_excel_rows(extra=[tiny])), 2022, log
    )
    assert out.filter(pl.col("school_name") == "Tiny School")[
        "enrollment"
    ].to_list() == [None]
    assert out.filter(pl.col("school_name") == "Tiny School")[
        "rate"
    ].to_list() == pytest.approx([95.0])
    assert info["enrollment_censored"] == 1 and info["total_check"].startswith(
        "skipped"
    )
    assert any("censored" in r for r in log.rows_frame()["reason"].to_list())


def test_excel_printed_total_must_equal_the_sum_of_rows():
    with pytest.raises(ValueError, match="printed statewide total is 105"):
        ssf.parse_me_excel(sheet(me_excel_rows(totals=(105, 50, 50))), 2022)
    with pytest.raises(
        ValueError,
        match="Cumberland rows sum to 50, the printed county total is 49",
    ):
        ssf.parse_me_excel(sheet(me_excel_rows(totals=(100, 49, 50))), 2022)


def test_excel_title_year_and_missing_total_row():
    with pytest.raises(ValueError, match="not the school year starting 2023"):
        ssf.parse_me_excel(sheet(me_excel_rows()), 2023)
    rows = [r for r in me_excel_rows() if r[0] != "Statewide"]
    with pytest.raises(ValueError, match="no printed statewide total"):
        ssf.parse_me_excel(sheet(rows), 2022)


def test_excel_without_a_kindergarten_block_is_an_error():
    rows = me_excel_rows()
    rows[2][2] = "Fifth Grade"
    with pytest.raises(ValueError, match="Kindergarten"):
        ssf.parse_me_excel(sheet(rows), 2022)


# --- PDF -------------------------------------------------------------------------------

# right edges of the kindergarten columns (numbers are right aligned) and of the first grade "Assessed"
X_ASSESSED, X_EXEMPT, X_MMR, X_POLIO, X_NEXT = (
    192.0,
    232.0,
    262.0,
    292.0,
    372.0,
)


def me_word(text, x0, x1, y):
    return f'    <word xMin="{x0:.2f}" yMin="{y:.2f}" xMax="{x1:.2f}" yMax="{y + 6:.2f}">{text}</word>'


def me_line(y, left, cells=()):
    """A row: `left` = [(text, x0, x1)] name/county words, `cells` = [(right edge, text)] numbers."""
    out = [me_word(t, x0, x1, y) for t, x0, x1 in left]
    out += [me_word(t, x1 - 6 * len(t), x1, y) for x1, t in cells]
    return out


def me_school(y, name, county, assessed=None, mmr=None, first_grade=None):
    left, x = [], 20.0
    for w in name.split():
        left.append((w, x, x + 4 * len(w)))
        x += 4 * len(w) + 2
    if county:
        left.append((county, 100.0, 100.0 + 6 * len(county)))
    cells = []
    if assessed is not None:
        cells += [
            (X_ASSESSED, str(assessed)),
            (X_EXEMPT, "0.0%"),
            (X_POLIO, "100.0%"),
        ]
    if mmr is not None:
        cells.append((X_MMR, mmr))
    if first_grade is not None:
        cells.append((X_NEXT, str(first_grade)))
    return me_line(y, left, cells)


def me_pdf_text(
    total="110",
    title="Maine School Age Survey Results 2016-2017",
    second_page_rows=None,
):
    header = [
        me_word(title, 400, 470, 30),
        me_word("Kindergarten", 121, 150, 40),
        me_word("School", 20, 40, 49.5),
        me_word("County", 100, 120, 49.5),
        me_word("Num", 150, 160, 50),
        me_word("Assessed", 162, 192, 50),
        me_word("TOTExempt", 200, 232, 50),
        me_word("2MMR", 240, 262, 50),
        me_word("3Polio", 270, 292, 50),
        me_word("Num", 330, 340, 50),
        me_word("Assessed", 342, 372, 50),
    ]
    page1 = header + me_school(60, "Alpha School", "Cumberland", 40, "95.0%")
    page1 += me_school(67, "Beta School", "Aroostook", 25, "88.0%")
    # a long name that runs into the county word (no space between them)
    page1 += me_line(
        74,
        [("Gamma", 20, 48), ("Elementary", 50, 98), ("SchoolYork", 100, 140)],
        [(X_ASSESSED, "35"), (X_MMR, "100.0%")],
    )
    page1 += me_school(
        81, "Delta Middle School", "Knox", first_grade=12
    )  # no kindergarten block
    page1 += me_line(
        88,
        [
            ("•", 20, 22),
            ("Total", 23, 40),
            ("exemptions", 42, 90),
            ("rose", 92, 110),
            ("York", 112, 130),
            ("to", 132, 140),
        ],
        [(300, "5.2%")],
    )
    page2 = me_school(55, "Epsilon School", "Hancock", 0, "0.0%")  # 0 assessed
    page2 += me_school(
        62, "Zeta School", "Waldo", 10, None
    )  # the MMR cell is blank
    page2 += me_line(
        69,
        [("Statewide", 20, 50), ("Totals", 52, 70)],
        [(X_ASSESSED + 2, total), (X_MMR, "93.0%"), (X_NEXT, "9999")],
    )
    pages = [page1, page2 if second_page_rows is None else second_page_rows]
    return (
        "<doc>\n"
        + "\n".join(
            '  <page width="1224.000000" height="792.000000">\n'
            + "\n".join(p)
            + "\n  </page>"
            for p in pages
        )
        + "\n</doc>"
    )


def test_pdf_rows_columns_and_dropped_rows():
    log = QCLog()
    out, info = ssf.parse_me_pdf(me_pdf_text(), 2016, log)
    assert out["school_name"].to_list() == [
        "Alpha School",
        "Beta School",
        "Gamma Elementary School",
        "Zeta School",
    ]
    assert out["county_label"].to_list() == [
        "Cumberland",
        "Aroostook",
        "York",
        "Waldo",
    ]
    assert out["enrollment"].to_list() == [40.0, 25.0, 35.0, 10.0]
    assert out["rate"].to_list() == [
        95.0,
        88.0,
        100.0,
        None,
    ]  # "%" sign: already percent; blank MMR cell: missing
    assert out["rate_source"].unique().to_list() == ["mmr"]
    assert (out["year_start"] == 2016).all()
    assert (
        info["printed_total"] == 110.0
        and info["total_check"] == "ok: 110 = printed statewide total"
    )
    assert (
        info["dropped_totals"] == 1
        and info["dropped_no_kindergarten"] == 2
        and info["dropped_other_lines"] == 1
    )
    assert "mmr='2MMR'" in info["columns"]
    reasons = log.rows_frame()["reason"].to_list()
    assert any("total row" in r for r in reasons) and any(
        "MMR rate blank" in r for r in reasons
    )


def test_pdf_printed_total_is_an_assertion():
    with pytest.raises(
        ValueError, match="sum to 110, the printed statewide total is 111"
    ):
        ssf.parse_me_pdf(me_pdf_text(total="111"), 2016)
    # a school lost (a row cut away) is caught the same way
    lost = me_pdf_text().replace(">40<", ">4<")
    with pytest.raises(ValueError, match="printed statewide total"):
        ssf.parse_me_pdf(lost, 2016)


def test_pdf_without_a_total_row_title_year_or_header():
    no_total = me_pdf_text(
        second_page_rows=me_school(55, "Epsilon School", "Hancock", 0, "0.0%")
    )
    with pytest.raises(ValueError, match="no printed statewide total"):
        ssf.parse_me_pdf(no_total, 2016)
    with pytest.raises(ValueError, match="not the school year starting 2017"):
        ssf.parse_me_pdf(me_pdf_text(), 2017)
    with pytest.raises(ValueError):
        ssf.parse_me_pdf("<doc>\n  <page>\n  </page>\n</doc>", 2016)


def test_pdf_two_numbers_in_one_column_is_an_error():
    twice = me_line(
        60,
        [("Alpha", 20, 48), ("Cumberland", 100, 150)],
        [(X_ASSESSED, "40"), (X_ASSESSED - 1, "41")],
    )
    with pytest.raises(ValueError, match="two numbers in column Assessed"):
        ssf.parse_me_pdf(me_pdf_text(second_page_rows=twice), 2016)


# --- units -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,unit",
    [
        ("MSAD #20", "district"),
        ("MSAD20", "district"),
        ("RSU 88/MSAD#24", "district"),
        ("AOS 77", "district"),
        ("Auburn School Department", "district"),
        ("Hermon School District", "district"),
        ("Van Buren District Schools- RSU88", "district"),
        ("Union Elementary School", "school"),
        ("Ashland District School", "school"),
        ("RSU 19 Etna Dixmont School", "school"),
        ("MSAD15 Memorial School", "school"),
    ],
)
def test_district_unit_is_a_name_heuristic(name, unit):
    assert ssf.me_district_unit(name) == unit


def test_excel_county_total_row_with_an_empty_county_cell_is_still_checked():
    rows = me_excel_rows(totals=(100, 49, 50))
    for r in rows:
        if r[0] == "Cumberland County Totals":
            r[1] = None
    with pytest.raises(
        ValueError,
        match="Cumberland rows sum to 50, the printed county total is 49",
    ):
        ssf.parse_me_excel(sheet(rows), 2022)


# --- Wisconsin -----------------------------------------------------------------------

WI_HEADER = [
    "County", "City", "School Name", "Comment", "School Year",
    "% Met Minimum Requirements", "% Met Minimum MMR Requirements",
    "Kindergarten: % Met Minimum Requirements", "Kindergarten: % Met Minimum MMR Requirements", "Kindergarten: % In Process",
]  # fmt: skip


def wi_row(
    county, city, name, year, k_overall, k_mmr, comment="Report received"
):
    """A By School row: the all-grades columns are filled so that they are not mistaken for the kindergarten ones."""
    return [
        county,
        city,
        name,
        comment,
        year,
        "70",
        "71",
        k_overall,
        k_mmr,
        "<5",
    ]


def wi_sheet(rows, title_rows=0):
    rows = [[None] * len(WI_HEADER)] * title_rows + [WI_HEADER] + rows
    return pl.DataFrame(
        rows,
        orient="row",
        schema=[f"column_{i}" for i in range(len(WI_HEADER))],
        strict=False,
        infer_schema_length=None,
    )


# a sheet with an all-vaccine year (2023, no MMR column values) and an MMR year (2024)
WI_ROWS = [
    wi_row("Adams", "Adams", "Adams El", "2023–2024", "81", "N/A"),
    wi_row("Adams", "Adams", "Big El", "2023–2024", " 92", "N/A"),
    wi_row("Adams", "Adams", "Banded El", "2023–2024", ">95", "N/A"),
    wi_row(
        "Adams", "Adams", "Adams High", "2023–2024", "N/A", "N/A"
    ),  # no kindergarten
    wi_row("Adams", "Adams", "Adams El", "2024– 2025", "70", "84"),
    wi_row(
        "Adams", "Adams", "Big El", "2024– 2025", "<5", "93"
    ),  # the all-vaccine cell is not used in an MMR year
    wi_row("Adams", "Adams", "Banded El", "2024– 2025", ">95", ">95"),
    wi_row(
        "Adams",
        "Adams",
        "Tiny El",
        "2024– 2025",
        "N/A",
        "N/A",
        "Report received, <10 students",
    ),
    wi_row(
        "Adams",
        "Adams",
        "Silent El",
        "2024– 2025",
        "N/A",
        "N/A",
        "No report received",
    ),
    wi_row(
        "Adams", "Adams", "Odd El", "2024– 2025", "60", "N/A"
    ),  # MMR year, MMR cell N/A: missing, not the overall value
]


def test_each_year_uses_its_column_and_says_which():
    out, info = ssf.parse_wisconsin(wi_sheet(WI_ROWS))
    assert info["rate_source_by_year"] == {2023: "overall", 2024: "mmr"}
    assert out.filter(pl.col("year_start") == 2023)[
        "rate_source"
    ].unique().to_list() == ["overall"]
    assert out.filter(pl.col("year_start") == 2024)[
        "rate_source"
    ].unique().to_list() == ["mmr"]
    by = {
        (r["school_name"], r["year_start"]): r["rate"]
        for r in out.iter_rows(named=True)
    }
    assert (
        by[("Adams El", 2023)] == 81.0 and by[("Big El", 2023)] == 92.0
    )  # " 92": padded text
    assert by[("Adams El", 2024)] == 84.0 and by[("Big El", 2024)] == 93.0
    assert by[("Odd El", 2024)] is None  # not the 60 of the all-vaccine column


def test_the_year_is_the_first_year_of_the_school_year_and_one_year_can_be_asked_for():
    out, info = ssf.parse_wisconsin(wi_sheet(WI_ROWS))
    assert sorted(out["year_start"].unique().to_list()) == [2023, 2024]
    only, info = ssf.parse_wisconsin(wi_sheet(WI_ROWS), 2024)
    assert (
        only["year_start"].unique().to_list() == [2024]
        and info["other_year_rows_skipped"] == 4
    )


def test_rows_without_kindergarten_data_are_dropped_and_logged_by_comment():
    log = QCLog()
    out, info = ssf.parse_wisconsin(wi_sheet(WI_ROWS), log=log)
    assert (
        "Adams High" not in out["school_name"].to_list()
        and "Tiny El" not in out["school_name"].to_list()
    )
    assert info["dropped_no_kindergarten"] == 3
    reasons = (
        log.rows_frame()
        .filter(pl.col("action") == "dropped")["reason"]
        .to_list()
    )
    assert (
        any("No report received" in r for r in reasons)
        and any("<10 students" in r for r in reasons)
        and any("Comment: Report received" in r for r in reasons)
    )


def test_a_band_stays_a_band_and_takes_the_midpoint():
    out, info = ssf.parse_wisconsin(wi_sheet(WI_ROWS))
    banded = out.filter(pl.col("band").is_not_null())
    assert banded["school_name"].to_list() == ["Banded El", "Banded El"]
    assert banded["band"].unique().to_list() == [">95%"] and banded[
        "rate"
    ].unique().to_list() == [97.5]
    assert info["banded"] == 2 and info["bands"] == [(">95%", 97.5)]


def test_a_less_than_5_cell_is_missing_not_2_5_and_is_logged():
    rows = [
        wi_row("Adams", "Adams", "A", "2018–2019", "<5", "N/A"),
        wi_row("Adams", "Adams", "B", "2018–2019", ">95", "N/A"),
        wi_row("Adams", "Adams", "C", "2018–2019", "88", "N/A"),
    ]
    log = QCLog()
    out, info = ssf.parse_wisconsin(wi_sheet(rows), log=log)
    assert out["band"].to_list() == [None, ">95%", None]
    assert out["rate"].to_list() == [
        None,
        97.5,
        88.0,
    ]  # ">95" alone: 97.5, whatever "<5" is
    assert info["bottom_band_missing"] == 1 and out.height == 3
    assert any("'<5'" in r for r in log.rows_frame()["reason"].to_list())


def test_suppressed_rate_is_missing_but_the_row_stays_and_is_logged():
    rows = [
        wi_row("Adams", "Adams", "A", "2018–2019", "N/A", "N/A"),
        wi_row("Adams", "Adams", "B", "2018–2019", "N/A", "75"),
        wi_row("Adams", "Adams", "C", "2018–2019", "90", "N/A"),
    ]
    log = QCLog()
    out, info = ssf.parse_wisconsin(wi_sheet(rows), log=log)
    # 2018 has an MMR value, so it is an MMR year: C (all-vaccine only) is a row with a missing rate
    assert out["school_name"].to_list() == ["B", "C"] and out[
        "rate"
    ].to_list() == [75.0, None]
    assert info["rate_missing"] == 1
    assert any(
        "N/A" in r
        for r in log.rows_frame()
        .filter(pl.col("action") == "changed")["reason"]
        .to_list()
    )


def test_fractions_are_multiplied_by_100_per_year():
    rows = [
        wi_row("Adams", "Adams", "A", "2018–2019", "0.9", "N/A"),
        wi_row("Adams", "Adams", "B", "2018–2019", "0.8", "N/A"),
        wi_row("Adams", "Adams", "A", "2019–2020", "90", "N/A"),
    ]
    out, info = ssf.parse_wisconsin(wi_sheet(rows))
    assert out["rate"].to_list() == pytest.approx([90.0, 80.0, 90.0])
    assert info["scale_by_year"] == {2018: "fraction x100", 2019: "percent"}


def test_total_rows_are_dropped_and_logged_but_schools_named_wisconsin_stay():
    rows = [
        wi_row(
            "Adams", "Adams", "Wisconsin Heights El", "2018–2019", "90", "N/A"
        ),
        wi_row("Adams", "Adams", "State Road El", "2018–2019", "91", "N/A"),
        wi_row("Adams", "Adams", "Total", "2018–2019", "80", "N/A"),
        wi_row("Adams", "Adams", "Statewide", "2018–2019", "85", "N/A"),
    ]
    log = QCLog()
    out, info = ssf.parse_wisconsin(wi_sheet(rows), log=log)
    assert (
        out["school_name"].to_list()
        == ["Wisconsin Heights El", "State Road El"]
        and info["dropped_totals"] == 2
    )
    assert "total or statewide row" in log.rows_frame()["reason"].to_list()


def test_the_header_may_move_and_the_columns_are_the_kindergarten_ones():
    out, info = ssf.parse_wisconsin(wi_sheet(WI_ROWS[:1], title_rows=3))
    assert out["rate"].to_list() == [
        81.0
    ]  # not the 70 / 71 of the all-grades columns
    assert out.columns == [
        "school_name",
        "county_label",
        "city",
        "enrollment",
        "rate",
        "band",
        "rate_source",
        "year_start",
        "unit",
    ]
    assert (
        out["unit"].to_list() == ["school"]
        and out["enrollment"].to_list() == [None]
        and out["city"].to_list() == ["Adams"]
    )


def test_same_name_schools_are_both_kept_and_counties_attach():
    rows = [
        wi_row("Fond du Lac", "X", "Same El", "2018–2019", "90", "N/A"),
        wi_row("Fond du Lac", "Y", "Same El", "2018–2019", "80", "N/A"),
        wi_row("St. Croix", "Z", "Z El", "2018–2019", "70", "N/A"),
    ]
    out, _ = ssf.parse_wisconsin(wi_sheet(rows))
    counties = pl.DataFrame(
        {
            "county_fips": ["55039", "55109"],
            "county_name": ["Fond du Lac", "St. Croix"],
            "county_state": ["WI", "WI"],
        }
    )
    out = ssf.attach_counties(out, "WI", counties)
    assert out.height == 3 and out["county_fips"].to_list() == [
        "55039",
        "55039",
        "55109",
    ]


def test_a_sheet_without_the_county_header_is_an_error():
    with pytest.raises(ValueError, match="County"):
        ssf.parse_wisconsin(
            pl.DataFrame({"column_0": ["a"], "column_1": ["b"]})
        )


# --- Colorado ------------------------------------------------------------
CO_COLUMNS = [
    "ID",
    "Site_Name",
    "District_Name",
    "Facility_Type",
    "County",
    "Survey_Type",
    "Vaccine",
    "Metric",
    "Value_Percent",
    "Enrollment",
    "Year",
]


def co_table(rows, first="﻿OBJECTID"):
    """Rows (ID, name, district, facility, county, survey, vaccine, metric, value, enrollment, year) as the text csv frame."""
    cols = {c: [r[j] for r in rows] for j, c in enumerate(CO_COLUMNS)}
    return pl.DataFrame(
        {"OBJECTID": [str(i) for i in range(len(rows))], **cols},
        schema={"OBJECTID": pl.String, **{c: pl.String for c in CO_COLUMNS}},
    )


def co_row(
    id_,
    name,
    county="Denver",
    facility="Public School",
    survey="Kindergarten",
    vaccine="MMR",
    metric="Fully Immunized",
    value="90.5",
    enrollment="80",
    year="2025/2026",
):
    return [
        id_,
        name,
        "District 1",
        facility,
        county,
        survey,
        vaccine,
        metric,
        value,
        enrollment,
        year,
    ]


def test_co_keeps_only_kindergarten_mmr_fully_immunized_of_the_year():
    t = co_table(
        [
            co_row("1", "Kept"),
            co_row("1", "Kept", metric="In Process", value="3"),
            co_row("1", "Kept", vaccine="DTaP"),
            co_row("1", "Kept", survey="School", enrollment="500"),
            co_row("1", "Kept", year="2024/2025", value="50"),
            co_row(
                "2",
                "Second",
                county="Adams",
                facility="Private School",
                value="100",
                enrollment="12",
            ),
        ]
    )
    out, info = ssf.parse_colorado(t, 2025)
    assert out["school_name"].to_list() == ["Kept", "Second"]
    assert out["rate"].to_list() == pytest.approx([90.5, 100.0])
    assert out["enrollment"].to_list() == [80.0, 12.0]
    assert out["county_label"].to_list() == ["Denver", "Adams"]
    assert (
        set(out["rate_source"]) == {"mmr"}
        and set(out["unit"]) == {"school"}
        and out["year_start"].to_list() == [2025, 2025]
    )
    assert (
        info["rows"] == 2
        and info["duplicate_ids"] == 0
        and info["scale"] == 1.0
    )


def test_co_other_year_selected_by_year_start():
    t = co_table(
        [
            co_row("1", "A", year="2023/2024", value="70"),
            co_row("1", "A", year="2024/2025", value="75"),
        ]
    )
    out, _ = ssf.parse_colorado(t, 2024)
    assert out["rate"].to_list() == [75.0]


def test_co_child_care_kindergarten_records_are_dropped_and_logged():
    t = co_table(
        [
            co_row("1", "School"),
            co_row("9", "Little Ones", facility="Child Care/Preschool"),
        ]
    )
    log = QCLog()
    out, info = ssf.parse_colorado(t, 2025, log=log)
    assert (
        out["school_name"].to_list() == ["School"]
        and info["dropped_childcare"] == 1
    )
    assert "child care" in log.rows_frame()["reason"].to_list()[0]


def test_co_unknown_county_has_no_label_and_is_reported():
    t = co_table(
        [co_row("1", "Nowhere", county="Unknown"), co_row("2", "Denver K")]
    )
    log = QCLog()
    out, info = ssf.parse_colorado(t, 2025, log=log)
    assert out["county_label"].to_list() == [None, "Denver"]
    assert info["unknown_county"] == [("Nowhere", "District 1")]
    assert any("Unknown" in r for r in log.rows_frame()["reason"].to_list())
    counties = pl.DataFrame(
        {
            "county_fips": ["08031"],
            "county_name": ["Denver"],
            "county_state": ["CO"],
        }
    )
    assert ssf.attach_counties(out, "CO", counties)[
        "county_fips"
    ].to_list() == [None, "08031"]


def test_co_fractions_are_scaled_and_non_numbers_are_missing():
    t = co_table(
        [
            co_row("1", "A", value="0.9"),
            co_row("2", "B", value="0.95"),
            co_row("3", "C", value="n/a"),
        ]
    )
    out, info = ssf.parse_colorado(t, 2025)
    assert (
        out["rate"].to_list()[:2] == pytest.approx([90.0, 95.0])
        and out["rate"].to_list()[2] is None
    )
    assert info["scale"] == 100.0 and info["rate_missing"] == 1


def test_co_missing_column_raises_and_bom_is_stripped():
    t = co_table([co_row("1", "A")]).rename({"ID": "﻿ID"})
    assert ssf.parse_colorado(t, 2025)[0].height == 1
    with pytest.raises(ValueError):
        ssf.parse_colorado(t.drop("Metric"), 2025)


# --- Indiana ------------------------------------------------------------
IN_OLD = [
    "Corporation_Name",
    "School_Name",
    "Grade",
    "Reported in SNM (CHIRP)",
    "School_Reporting_To_CHIRP",
    "MMR",
    "MMR Rate",
    "County",
]
IN_NEW = [
    "Corporation_Name",
    "School_Name",
    "County_Code",
    "County",
    "Grade",
    "Reported_in_SNM_CHIRP",
    "School_Reporting_To_CHIRP",
    "MMR",
    "MMR_Rate",
]


def in_sheet(header, rows):
    """A headerless text frame: what pl.read_excel(header_row=None, infer_schema_length=0) gives."""
    width = len(header)
    cols = {
        f"column_{j + 1}": [header[j]]
        + [r[j] if j < len(r) else None for r in rows]
        for j in range(width)
    }
    return pl.DataFrame(cols, schema={c: pl.String for c in cols})


def test_in_old_header_kindergarten_only_and_fractions_scaled():
    frame = in_sheet(
        IN_OLD,
        [
            ["Corp A", "School 1", "K", "50", "0.9", "45", "0.9", "ADAMS"],
            ["Corp A", "School 1", "1", "60", "0.9", "30", "0.5", "ADAMS"],
            ["Corp B", "School 2", "K", "20", "1", "19", "0.95", "ALLEN"],
            ["Corp B", "School 2", "12", "20", "1", "19", "0.95", "ALLEN"],
        ],
    )
    out, info = ssf.parse_indiana(frame, 2023)
    assert out["school_name"].to_list() == ["School 1", "School 2"]
    assert out["county_label"].to_list() == ["ADAMS", "ALLEN"]
    assert out["enrollment"].to_list() == [50.0, 20.0]
    assert out["rate"].to_list() == pytest.approx(
        [90.0, 95.0]
    )  # 0.9 x 100; not School_Reporting_To_CHIRP
    assert set(out["rate_source"]) == {"mmr"} and set(out["unit"]) == {
        "school"
    }
    assert out["year_start"].to_list() == [2023, 2023]
    assert info["other_grade_rows"] == 2 and info["scale"] == 100.0


def test_in_new_header_spellings_give_the_same_columns():
    frame = in_sheet(
        IN_NEW,
        [
            [
                "Corp A",
                "School 1",
                "18001",
                "ADAMS",
                "K",
                "50",
                "0.9",
                "45",
                "0.9",
            ]
        ],
    )
    out, _ = ssf.parse_indiana(frame, 2024)
    assert out.row(0)[:6] == (
        "School 1",
        "ADAMS",
        50.0,
        pytest.approx(90.0),
        "mmr",
        "school",
    )


def test_in_header_is_found_below_title_rows():
    frame = in_sheet(
        IN_NEW,
        [
            [
                "Corp A",
                "School 1",
                "18001",
                "ADAMS",
                "K",
                "50",
                "0.9",
                "45",
                "0.9",
            ]
        ],
    )
    title = pl.DataFrame({c: ["Indiana 2025", None] for c in frame.columns})
    out, _ = ssf.parse_indiana(pl.concat([title, frame]), 2025)
    assert out.height == 1


def test_in_censored_enrollment_is_missing_and_rate_kept_and_logged():
    frame = in_sheet(
        IN_NEW,
        [
            [
                "Corp A",
                "Tiny",
                "18001",
                "ADAMS",
                "K",
                "<=10",
                "0.2",
                "<=10",
                "1",
            ],
            [
                "Corp A",
                "Big",
                "18001",
                "ADAMS",
                "K",
                "40",
                "0.9",
                "30",
                "0.75",
            ],
        ],
    )
    log = QCLog()
    out, info = ssf.parse_indiana(frame, 2025, log=log)
    assert out["enrollment"].to_list() == [None, 40.0]
    assert out["rate"].to_list() == pytest.approx([100.0, 75.0])
    assert (
        info["enrollment_censored"] == 1
        and info["mmr_count_censored"] == 1
        and info["censored_count_rate_100"] == 1
    )
    reasons = log.rows_frame()["reason"].to_list()
    assert any("enrollment" in r for r in reasons) and any(
        "MMR count" in r for r in reasons
    )


def test_in_censored_count_rate_can_be_made_missing():
    frame = in_sheet(
        IN_NEW,
        [
            [
                "Corp A",
                "Tiny",
                "18001",
                "ADAMS",
                "K",
                "<=10",
                "0.2",
                "<=10",
                "0",
            ],
            [
                "Corp A",
                "Big",
                "18001",
                "ADAMS",
                "K",
                "40",
                "0.9",
                "30",
                "0.75",
            ],
        ],
    )
    out, info = ssf.parse_indiana(frame, 2025, keep_censored_count_rates=False)
    assert out["rate"].to_list()[0] is None and out["rate"].to_list()[
        1
    ] == pytest.approx(75.0)
    assert info["rate_missing"] == 1


def test_in_rate_above_100_is_kept_and_logged():
    frame = in_sheet(
        IN_NEW,
        [
            ["Corp A", "Over", "18001", "ADAMS", "K", "20", "1", "21", "1.05"],
            ["Corp A", "Ok", "18001", "ADAMS", "K", "20", "1", "10", "0.5"],
        ],
    )
    log = QCLog()
    out, info = ssf.parse_indiana(frame, 2025, log=log)
    assert (
        out["rate"].to_list() == pytest.approx([105.0, 50.0])
        and info["rate_above_100"] == 1
    )
    assert "rate above 100" in log.rows_frame()["reason"].to_list()[0]


def test_in_percent_file_is_not_scaled_and_missing_rate_stays():
    frame = in_sheet(
        IN_NEW,
        [
            ["Corp A", "A", "18001", "ADAMS", "K", "20", "1", "18", "90"],
            ["Corp A", "B", "18001", "ADAMS", "K", "20", "1", "x", None],
        ],
    )
    out, info = ssf.parse_indiana(frame, 2025)
    assert out["rate"].to_list() == [90.0, None] and info["scale"] == 1.0


def test_in_missing_columns_raise():
    frame = in_sheet(
        ["Corporation_Name", "School_Name", "Grade"], [["a", "b", "K"]]
    )
    with pytest.raises(ValueError):
        ssf.parse_indiana(frame, 2025)


def test_in_counties_attach_by_name():
    counties = pl.DataFrame(
        {
            "county_fips": ["18001", "18003"],
            "county_name": ["Adams", "Allen"],
            "county_state": ["IN", "IN"],
        }
    )
    frame = in_sheet(
        IN_NEW,
        [["Corp A", "A", "18001", "ADAMS", "K", "20", "1", "18", "0.9"]],
    )
    out, _ = ssf.parse_indiana(frame, 2025)
    assert ssf.attach_counties(out, "IN", counties)[
        "county_fips"
    ].to_list() == ["18001"]


# --- North Carolina ------------------------------------------------------------
NC_HEADER = [
    "County",
    "School Name",
    "School Type",
    "Total Enrollment",
    "Up to Date (Count)",
    "Up to Date (%)",
    "Provisionally Enrolled (Count)",
]


def nc_sheet(rows, header=NC_HEADER):
    cols = {
        f"column_{j + 1}": [header[j]] + [r[j] for r in rows]
        for j in range(len(header))
    }
    return pl.DataFrame(cols, schema={c: pl.String for c in cols})


def test_nc_xlsx_fractions_and_merged_county_cells_are_carried_forward():
    frame = nc_sheet(
        [
            ["Alamance", "School A", "Private", "20", "18", "0.9", "0"],
            [None, "School B", "Public", "100", "95", "0.95", "1"],
            ["Alexander", "School C", "Charter", "50", "40", "0.8", "0"],
            [None, "School D", "Public", "10", "10", "1", "0"],
        ]
    )
    out, info = ssf.parse_north_carolina(frame, 2025)
    assert out["county_label"].to_list() == [
        "Alamance",
        "Alamance",
        "Alexander",
        "Alexander",
    ]
    assert out["rate"].to_list() == pytest.approx([90.0, 95.0, 80.0, 100.0])
    assert out["enrollment"].to_list() == [20.0, 100.0, 50.0, 10.0]
    assert set(out["rate_source"]) == {"overall"} and set(out["unit"]) == {
        "school"
    }
    assert out["year_start"].to_list() == [2025] * 4
    assert info["county_carried_rows"] == 2 and info["enrollment_sum"] == 180.0


def test_nc_csv_percent_text_is_not_scaled():
    frame = nc_sheet(
        [
            ["Alamance", "A", "Public", "34", "29", "85.3%", "0"],
            ["Alamance", "B", "Public", "100", "82", "82.0%", "9"],
        ]
    )
    out, info = ssf.parse_north_carolina(frame, 2025)
    assert (
        out["rate"].to_list() == pytest.approx([85.3, 82.0])
        and info["scale"] == 1.0
    )


def test_nc_total_and_nameless_rows_are_dropped_and_logged():
    frame = nc_sheet(
        [
            ["Alamance", "A", "Public", "34", "29", "0.85", "0"],
            [None, "State Total", None, "34", "29", "0.85", None],
            [None, None, None, None, None, None, None],
        ]
    )
    log = QCLog()
    out, info = ssf.parse_north_carolina(frame, 2025, log=log)
    assert (
        out["school_name"].to_list() == ["A"] and info["dropped_totals"] == 2
    )
    assert "nameless or total row" in log.rows_frame()["reason"].to_list()


def test_nc_zero_enrollment_has_no_rate():
    frame = nc_sheet(
        [
            ["Alamance", "Empty", "Public", "0", "0", "0", "0"],
            ["Alamance", "A", "Public", "10", "9", "0.9", "0"],
        ]
    )
    out, info = ssf.parse_north_carolina(frame, 2025)
    assert (
        out["rate"].to_list()[0] is None
        and info["enrollment_zero"] == 1
        and info["rate_missing"] == 1
    )


def test_nc_missing_column_raises():
    with pytest.raises(ValueError):
        ssf.parse_north_carolina(
            nc_sheet(
                [["Alamance", "A", "Public", "1", "1", "1", "0"]],
                header=[
                    "County",
                    "School Name",
                    "School Type",
                    "Total Enrollment",
                    "Count",
                    "Pct",
                    "x",
                ],
            ),
            2025,
        )


def test_nc_counties_attach_by_name():
    counties = pl.DataFrame(
        {
            "county_fips": ["37001", "37003"],
            "county_name": ["Alamance", "Alexander"],
            "county_state": ["NC", "NC"],
        }
    )
    out, _ = ssf.parse_north_carolina(
        nc_sheet(
            [
                ["Alamance", "A", "Public", "10", "9", "0.9", "0"],
                [None, "B", "Public", "10", "9", "0.9", "0"],
            ]
        ),
        2025,
    )
    assert ssf.attach_counties(out, "NC", counties)[
        "county_fips"
    ].to_list() == ["37001", "37001"]


# --- Massachusetts ------------------------------------------------------------
MA_NEW = [
    "SCHOOL NAME",
    "SCHOOL TYPE",
    "MUNICIPALITY",
    "COUNTY",
    "5\r\nDTAP",
    "4\r\nPOLIO",
    "2\r\nMMR ",
    "3\r\nHEP B",
]
MA_OLD = [
    "School Name",
    "School Type",
    "City",
    "County",
    "5 DTaP",
    "4 Polio",
    "2 MMR",
    "3 Hep B",
]


def ma_sheet(header, rows):
    return pl.DataFrame(
        [header] + rows,
        orient="row",
        schema=[f"column_{i}" for i in range(len(header))],
        strict=False,
    )


def ma_row(name, county, mmr, city="BOSTON"):
    return [name, "PUBLIC", city, county, "0.9", "0.9", mmr, "0.9"]


def test_ma_fractions_become_percent_and_header_line_breaks_are_ignored():
    out, info = ssf.parse_massachusetts(
        ma_sheet(
            MA_NEW,
            [
                ma_row("A ELEM", "SUFFOLK", "0.98"),
                ma_row("B ELEM", "NORFOLK", "1"),
            ],
        ),
        2025,
    )
    assert out["school_name"].to_list() == ["A ELEM", "B ELEM"]
    assert out["rate"].to_list() == pytest.approx([98.0, 100.0])
    assert out["county_label"].to_list() == ["SUFFOLK", "NORFOLK"]
    assert out["city"].to_list() == ["BOSTON", "BOSTON"]
    assert out["rate_source"].unique().to_list() == ["mmr"] and out[
        "unit"
    ].unique().to_list() == ["school"]
    assert (
        out["year_start"].to_list() == [2025, 2025]
        and out["enrollment"].null_count() == 2
    )
    assert info["scale"] == "fraction x100"


def test_ma_a_percent_file_is_left_alone():
    out, info = ssf.parse_massachusetts(
        ma_sheet(
            MA_OLD,
            [ma_row("A", "SUFFOLK", "100"), ma_row("B", "SUFFOLK", "88.5")],
        ),
        2021,
    )
    assert (
        out["rate"].to_list() == [100.0, 88.5] and info["scale"] == "percent"
    )


def test_ma_symbols_are_missing_rates_and_the_school_stays():
    rows = [
        ma_row("A", "SUFFOLK", "0.95"),
        ma_row("B", "SUFFOLK", "†"),
        ma_row("C", "SUFFOLK", "*"),
        ma_row("D", "SUFFOLK", "¶"),
        ma_row("E", "SUFFOLK", "N/A"),
        ma_row("F", "SUFFOLK", None),
    ]
    log = QCLog()
    out, info = ssf.parse_massachusetts(ma_sheet(MA_NEW, rows), 2025, log)
    assert out.height == 6 and out["rate"].to_list() == [95.0] + [None] * 5
    assert (
        info["symbols"] == {"†": 1, "*": 1, "¶": 1, "N/A": 1}
        and info["rate_missing"] == 5
    )
    assert (
        log.rows_frame().filter(pl.col("reason").str.contains("symbol")).height
        == 5
    )


def test_ma_footnotes_and_empty_rows_are_dropped_and_a_total_row_is_logged():
    rows = [
        ma_row("A", "SUFFOLK", "0.95"),
        [None] * 8,
        [
            "•All data are self-reported",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        ma_row("Statewide", "ALL", "0.97"),
    ]
    log = QCLog()
    out, info = ssf.parse_massachusetts(ma_sheet(MA_OLD, rows), 2014, log)
    assert out["school_name"].to_list() == ["A"]
    assert (
        info["dropped_footnote_or_empty"] == 2 and info["dropped_totals"] == 1
    )
    assert log.rows_frame().filter(pl.col("action") == "dropped").height == 1


def test_ma_a_real_school_name_that_starts_like_a_total_stays():
    out, _ = ssf.parse_massachusetts(
        ma_sheet(
            MA_OLD,
            [
                ma_row("Total Care Academy", "SUFFOLK", "0.9"),
                ma_row("State Road School", "NORFOLK", "0.9"),
            ],
        ),
        2014,
    )
    assert out.height == 2


def test_ma_the_header_may_be_below_the_first_row():
    out, info = ssf.parse_massachusetts(
        ma_sheet(
            ["Notes"] + [None] * 7, [MA_OLD, ma_row("A", "SUFFOLK", "0.9")]
        ),
        2014,
    )
    assert info["header_at"] == 1 and out["school_name"].to_list() == ["A"]


def test_ma_no_header_is_an_error():
    with pytest.raises(ValueError, match="header"):
        ssf.parse_massachusetts(
            ma_sheet(["x", "y", "z", "w"], [["a", "b", "c", "d"]]), 2025
        )


# --- Minnesota ------------------------------------------------------------
MN_HEADER = [
    "School District",
    "School Name",
    "County",
    "Kindergarten Enrollment",
    "DTaP % Vaccinated",
    "MMR % Vaccinated",
    "MMR % Partially Vaccinated or No Doses",
]
MN_TITLE = "Percent of Kindergarteners Vaccinated by School 2025-26"


def mn_sheet(rows, title=MN_TITLE, header=MN_HEADER):
    allrows = [[title] + [None] * 6, header] + rows
    return pl.DataFrame(
        allrows,
        orient="row",
        schema=[f"column_{i}" for i in range(7)],
        strict=False,
    )


def mn_row(district, name, county, enrollment, mmr, dtap="0.9"):
    return [district, name, county, enrollment, dtap, mmr, "0.05"]


def test_mn_fractions_become_percent_and_the_mmr_column_is_used_not_dtap():
    out, info = ssf.parse_minnesota(
        mn_sheet(
            [
                mn_row("A", "A EL.", "ANOKA", "74", "0.8", dtap="0.5"),
                mn_row("B", "B EL.", "ST. LOUIS", "20", "1"),
            ]
        ),
        2025,
    )
    assert out["school_name"].to_list() == ["A EL.", "B EL."]
    assert out["county_label"].to_list() == ["ANOKA", "ST. LOUIS"]
    assert out["enrollment"].to_list() == [74.0, 20.0]
    assert out["rate"].to_list() == pytest.approx([80.0, 100.0])
    assert out["rate_source"].unique().to_list() == ["mmr"] and out[
        "unit"
    ].unique().to_list() == ["school"]
    assert (
        out["year_start"].unique().to_list() == [2025]
        and out["band"].null_count() == 2
    )
    assert info["scale"] == "fraction x100"


def test_mn_censored_cells_are_missing_and_the_school_stays():
    rows = [
        mn_row("A", "A", "ANOKA", "Did not report", "*"),
        mn_row("A", "B", "ANOKA", "Enrollment < 10", "**"),
        mn_row("A", "C", "ANOKA", "18", "***"),
        mn_row("A", "D", "ANOKA", "50", "0.9"),
    ]
    log = QCLog()
    out, info = ssf.parse_minnesota(mn_sheet(rows), 2025, log)
    assert out.height == 4 and out["rate"].to_list() == [
        None,
        None,
        None,
        90.0,
    ]
    assert out["enrollment"].to_list() == [None, None, 18.0, 50.0]
    assert info["rate_missing"] == 3 and info["enrollment_missing"] == 2
    assert info["censored_cells"] == {
        "Did not report": 1,
        "*": 1,
        "Enrollment < 10": 1,
        "**": 1,
        "***": 1,
    }
    assert (
        log.rows_frame()
        .filter(pl.col("reason").str.contains("rate censored"))
        .height
        == 3
    )


def test_mn_bands_become_the_middle_of_their_interval_and_keep_their_text():
    rows = [
        mn_row("A", "A", "ANOKA", "60", ">95%"),
        mn_row("A", "B", "ANOKA", "70", ">98%"),
        mn_row("A", "C", "ANOKA", "80", "0.9"),
    ]
    log = QCLog()
    out, info = ssf.parse_minnesota(mn_sheet(rows), 2025, log)
    assert out["band"].to_list() == [">95%", ">98%", None]
    assert out["rate"].to_list() == pytest.approx([96.5, 99.0, 90.0])
    assert out["enrollment"].to_list() == [60.0, 70.0, 80.0]
    assert info["banded"] == 2 and info["bands"] == [
        (">95%", 96.5),
        (">98%", 99.0),
    ]
    assert (
        log.rows_frame().filter(pl.col("reason").str.contains("banded")).height
        == 2
    )


def test_mn_the_statewide_row_is_dropped_and_logged():
    rows = [
        mn_row("Statewide", "Statewide", "Statewide", "65460", "0.87"),
        mn_row("A", "A", "ANOKA", "50", "0.9"),
    ]
    log = QCLog()
    out, info = ssf.parse_minnesota(mn_sheet(rows), 2025, log)
    assert (
        out["school_name"].to_list() == ["A"] and info["dropped_totals"] == 1
    )
    assert log.rows_frame().filter(pl.col("action") == "dropped").height == 1


def test_mn_the_title_year_must_agree_with_year_start():
    with pytest.raises(ValueError, match="2025-26"):
        ssf.parse_minnesota(
            mn_sheet([mn_row("A", "A", "ANOKA", "50", "0.9")]), 2024
        )


def test_mn_an_old_workbook_without_county_is_refused():
    old = [
        "School District",
        "School",
        "Kindergarten Enrollment",
        "% Vaccinated",
        None,
        None,
        None,
    ]
    with pytest.raises(ssf.MNOldLayout):
        ssf.parse_minnesota(
            mn_sheet(
                [["A", "B", "50", "0.9", None, None, None]],
                title="Percent of Kindergarteners Vaccinated by School",
                header=old,
            ),
            2015,
        )


# --- Pennsylvania ------------------------------------------------------------
PA_HEADER = [
    "County",
    "School",
    "Grade",
    "Total Students Enrolled",
    "DTaP Percent",
    "Polio Percent",
    "MMR Percent",
    "Hep B Percent",
]


def pa_sheet(rows):
    return pl.DataFrame(
        [PA_HEADER] + rows,
        orient="row",
        schema=[f"column_{i}" for i in range(8)],
        strict=False,
    )


def pa_row(county, school, grade, enrolled, mmr, dtap="50.0%"):
    return [county, school, grade, enrolled, dtap, "90.0%", mmr, "90.0%"]


def test_pa_only_kindergarten_rows_are_kept_with_percent_strings_and_thousands_separators():
    rows = [
        pa_row(
            "Adams", "A El Sch", "Kindergarten", "1,012", "92.6%", dtap="1.0%"
        ),
        pa_row("Adams", "A El Sch", "7th Grade", "99", "10.0%"),
        pa_row("Adams", "A El Sch", "12th Grade", "98", "11.0%"),
        pa_row("Allegheny", "B Sch", "Kindergarten", "30", "100.0%"),
    ]
    log = QCLog()
    out, info = ssf.parse_pennsylvania(pa_sheet(rows), 2025, log)
    assert out["school_name"].to_list() == ["A El Sch", "B Sch"]
    assert out["county_label"].to_list() == ["Adams", "Allegheny"]
    assert out["enrollment"].to_list() == [1012.0, 30.0]
    assert out["rate"].to_list() == pytest.approx([92.6, 100.0])
    assert out["rate_source"].unique().to_list() == ["mmr"] and out[
        "unit"
    ].unique().to_list() == ["school"]
    assert out["year_start"].unique().to_list() == [2025]
    assert info["dropped_other_grades"] == 2 and info["scale"] == "percent"
    assert log.rows_frame().filter(pl.col("action") == "dropped").height == 2


def test_pa_a_blank_rate_is_missing_and_the_school_and_its_enrollment_stay():
    log = QCLog()
    out, info = ssf.parse_pennsylvania(
        pa_sheet(
            [
                pa_row("Adams", "Tiny", "Kindergarten", "7", None),
                pa_row("Adams", "Big", "Kindergarten", "25", "88.0%"),
            ]
        ),
        2025,
        log,
    )
    assert out["rate"].to_list() == [None, 88.0] and out[
        "enrollment"
    ].to_list() == [7.0, 25.0]
    assert (
        info["rate_missing"] == 1
        and info["max_enrollment_of_blank_rate"] == 7.0
        and info["min_enrollment_of_rated"] == 25.0
    )
    assert (
        log.rows_frame().filter(pl.col("reason").str.contains("blank")).height
        == 1
    )


def test_pa_a_total_row_is_dropped_but_a_real_name_that_contains_the_word_stays():
    rows = [
        pa_row("Adams", "Total", "Kindergarten", "500", "90.0%"),
        pa_row(
            "Adams", "Tri-State Christian Academy", "Kindergarten", "7", None
        ),
        pa_row("Adams", "State Street El Sch", "Kindergarten", "24", "100.0%"),
    ]
    out, info = ssf.parse_pennsylvania(pa_sheet(rows), 2025)
    assert (
        out["school_name"].to_list()
        == ["Tri-State Christian Academy", "State Street El Sch"]
        and info["dropped_totals"] == 1
    )


def test_pa_a_file_in_fractions_is_scaled():
    out, info = ssf.parse_pennsylvania(
        pa_sheet(
            [
                pa_row("Adams", "A", "Kindergarten", "30", "0.95"),
                pa_row("Adams", "B", "Kindergarten", "30", "0.9"),
            ]
        ),
        2025,
    )
    assert (
        out["rate"].to_list() == pytest.approx([95.0, 90.0])
        and info["scale"] == "fraction x100"
    )


def test_pa_a_missing_header_or_column_is_an_error():
    with pytest.raises(ValueError, match="header"):
        ssf.parse_pennsylvania(
            pl.DataFrame([["x", "y"]], orient="row", schema=["a", "b"]),
            2025,
        )
    bad = pl.DataFrame(
        [["County", "School", "Grade", "Total Students Enrolled"]],
        orient="row",
        schema=list("abcd"),
    )
    with pytest.raises(ValueError, match="mmr percent"):
        ssf.parse_pennsylvania(bad, 2025)


# --- Texas ------------------------------------------------------------
TX_HEADER = [
    "Facility Number",
    "School Type",
    "Facility Name",
    "Facility Address",
    "County",
    "DTP/DTaP/DT/Td",
    "Hepatitis A",
    "Hepatitis B",
    "MMR",
    "Polio",
    "Varicella",
]
TX_TITLE = (
    "Vaccine Coverage Rates among Kindergarteners in Texas, ISD, 2025-2026"
)
TX_NOTE = "Note: This list does not include schools with equal or fewer than 5 students in kindergarten."


def tx_sheet(rows, title=TX_TITLE):
    allrows = [
        [title] + [None] * 10,
        [TX_NOTE] + [None] * 10,
        TX_HEADER,
    ] + rows
    return pl.DataFrame(
        allrows,
        orient="row",
        schema=[f"column_{i}" for i in range(11)],
        strict=False,
    )


def tx_row(
    fac,
    kind,
    name,
    county,
    mmr,
    addr="5701 RED BIRD CTR DR, DALLAS, TX, 75237",
):
    return [
        fac,
        kind,
        name,
        addr,
        county,
        "0.5",
        "0.5",
        "0.5",
        mmr,
        "0.5",
        "0.5",
    ]


def test_tx_public_rows_are_districts_private_rows_schools_and_fractions_become_percent():
    rows = [
        tx_row("9057829000", "Public School", "A+ ACADEMY", "Dallas", "1"),
        tx_row(
            "9109901000",
            "Public ISD",
            "ABBOTT ISD",
            "Hill",
            "0.923076923",
            addr="P O BOX 226, ABBOTT, TX, .",
        ),
        tx_row(
            "7221120101",
            "Private School",
            "Abilene Christian School",
            "Taylor",
            "0.8333",
        ),
    ]
    out, info = ssf.parse_texas(tx_sheet(rows), 2025)
    assert out["unit"].to_list() == ["district", "district", "school"]
    assert out["rate"].to_list() == pytest.approx([100.0, 92.3076923, 83.33])
    assert out["school_name"].to_list() == [
        "A+ ACADEMY",
        "ABBOTT ISD",
        "Abilene Christian School",
    ]
    assert out["county_label"].to_list() == ["Dallas", "Hill", "Taylor"]
    assert out["city"].to_list() == ["DALLAS", "ABBOTT", "DALLAS"]
    assert out["rate_source"].unique().to_list() == ["mmr"] and out[
        "year_start"
    ].unique().to_list() == [2025]
    assert out["enrollment"].null_count() == 3
    assert (
        info["district_rows"] == 2
        and info["school_rows"] == 1
        and info["district_facility_not_000"] == 0
    )


def test_tx_blank_and_nr_rates_are_missing_and_the_row_stays():
    rows = [
        tx_row("1000000000", "Public ISD", "A", "Dallas", None),
        tx_row("1100000000", "Public ISD", "B", "Dallas", "NR"),
        tx_row("1200000000", "Public ISD", "C", "Dallas", "0.9"),
    ]
    log = QCLog()
    out, info = ssf.parse_texas(tx_sheet(rows), 2025, log)
    assert out["rate"].to_list() == [None, None, 90.0]
    assert (
        info["rate_missing"] == 2
        and info["tokens"] == {"NR": 1}
        and info["blank_rates"] == 1
    )
    assert (
        log.rows_frame().filter(pl.col("reason").str.contains("NR")).height
        == 2
    )


def test_tx_an_unknown_school_type_is_an_error():
    with pytest.raises(ValueError, match="neither public nor private"):
        ssf.parse_texas(
            tx_sheet([tx_row("1", "Charter", "A", "Dallas", "0.9")]), 2025
        )


def test_tx_the_title_year_must_agree_and_the_header_must_exist():
    with pytest.raises(ValueError, match="2025-2026"):
        ssf.parse_texas(
            tx_sheet([tx_row("1", "Public ISD", "A", "Dallas", "0.9")]), 2024
        )
    with pytest.raises(ValueError, match="header"):
        ssf.parse_texas(
            pl.DataFrame([["a", "b"]], orient="row", schema=["x", "y"]),
            2025,
        )


def test_tx_a_total_row_is_dropped_but_a_real_name_that_starts_like_one_stays():
    rows = [
        tx_row("1", "Public ISD", "Total", "Dallas", "0.9"),
        tx_row("2", "Public ISD", "State Line ISD", "Dallas", "0.9"),
    ]
    out, info = ssf.parse_texas(tx_sheet(rows), 2025)
    assert (
        out["school_name"].to_list() == ["State Line ISD"]
        and info["dropped_totals"] == 1
    )


# --- the school year of a state file name --------------------------------------------


def test_school_year_of_a_massachusetts_file_name():
    f = ssf.file_year_massachusetts
    assert f("Massachusetts/kinder-rates-by-school-14-15.xlsx") == 2014
    assert (
        f(
            "Massachusetts/kindergarten-immunization-data-by-school-2018-2019.xlsx"
        )
        == 2018
    )
    assert (
        f(
            "Massachusetts/Kindergarten Immunization Data by School 2020- 2021.xlsx"
        )
        == 2020
    )
    assert f("Massachusetts/kinder-rates-by-school.xls") is None


def test_school_year_of_a_minnesota_file_name():
    assert ssf.file_year_minnesota("Minnesota/pervaxkind1920.xlsx") == 2019
    assert ssf.file_year_minnesota("Minnesota/kschool2526.xlsx") == 2025
    with pytest.raises(ValueError):
        ssf.file_year_minnesota("Minnesota/readme.xlsx")
    with pytest.raises(ValueError):
        ssf.file_year_minnesota("Minnesota/kschool2527.xlsx")


def test_school_year_of_a_pennsylvania_file_name():
    name = "Pennsylvania/Pennsylvania school immunization rates, 2025-2026 – x.csv"
    assert ssf.file_year_pennsylvania(name) == 2025
    with pytest.raises(ValueError):
        ssf.file_year_pennsylvania("Pennsylvania/rates.csv")


def test_school_year_of_a_texas_file_name():
    f = ssf.file_year_texas
    assert (
        f(
            "Texas/2019-2020-School-Vaccination-Coverage-Levels---Kindergarten.xlsx"
        )
        == 2019
    )
    assert (
        f(
            "Texas/22-23-School-Vaccination-Coverage-by-District-and-County-K.xlsx"
        )
        == 2022
    )
    assert (
        f(
            "Texas/2025-2026-school-vaccination-coverage-by-district-and-county-kg.xlsx"
        )
        == 2025
    )
    assert (
        f(
            "Texas/2023-2024_School_Vaccination_Coverage_Levels_Kindergarten.xlsx"
        )
        == 2023
    )
    with pytest.raises(ValueError):
        f("Texas/coverage.xlsx")


def test_new_york_collapses_a_school_listed_twice_with_the_same_id_year_and_rate():
    def row(school_id, name, measles, type_="Private School"):
        return {
            "School ID": school_id,
            "Report Period": "2023-2024",
            "Type": type_,
            "School Name": name,
            "County": "Kings",
            "Street": "1 Main St",
            "Percent Immunized Measles": measles,
            "Percent Medical Exemptions": "0.00%",
        }

    table = pl.DataFrame(
        [
            row("1", "A", "98.00%"),
            row("1", "A", "98.00%"),  # the same school twice: one
            row("2", "B", "96.00%"),
            row("2", "B", "100.00%"),  # the same ID, another rate: both kept
            row(
                "3", "A", "98.00%"
            ),  # another school, same name and rate: kept
            row(
                "4", "D", "100.00%", type_="School District"
            ),  # a district total
        ]
    )
    log = QCLog()
    out, _ = ssf.parse_new_york(table, log)
    assert out["school_name"].to_list() == ["A", "B", "B", "A"]
    assert (
        log.rows_frame()
        .filter(pl.col("reason").str.contains("listed twice"))
        .height
        == 1
    )
