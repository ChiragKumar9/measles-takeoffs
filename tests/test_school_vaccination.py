"""The school-level vaccination files (outbreak_size.school_vaccination)."""

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from pyproj import Transformer
from shapely.geometry import box

from outbreak_size import school_vaccination as sv


def schools(rows, source="chen_bento"):
    """A frame in the common layout from partial rows (a list of dicts)."""
    df = pl.DataFrame(rows)
    df = df.with_columns(
        [
            pl.col(c).cast(t)
            for c, t in sv.COLUMN_TYPES.items()
            if c in df.columns
        ]
    )
    if "rid" not in df.columns:
        df = df.with_row_index("rid")
    if "row_id" not in df.columns:
        df = df.with_columns(
            row_id=pl.concat_str(pl.lit("toy:"), pl.col("rid"))
        )
    if "source" not in df.columns:
        df = df.with_columns(source=pl.lit(source))
    return sv.standardize(df)


CFG = {
    "wapo_min_share_of_chen_bento": 0.9,
    "match_radius_m": 100,
    "rate_tolerance_points": 0.1,
    "placeholder_rate_below": 10,
    "secondary_included_states": ["IA"],
    "all_grade_enrollment_states": ["IA", "IL"],
    "chen_bento_district_states": ["TX"],
    "enrollment_basis": {
        "washington_post": {"denominator": ["IA"]},
        "chen_bento": {"not_denominator": ["TX"]},
    },
    "low_rate": {
        "below": 50,
        "below_min_enrollment": 50,
        "far_below_median_points": 50,
        "far_below_median_min_enrollment": 30,
    },
}


# --- the county-level numbers, by hand ----------------------------------------


def county_rows(rates, enrollments, types=None, county="01001"):
    n = len(rates)
    return [
        {
            "state": "AL",
            "county_fips": county,
            "county_name": "Autauga",
            "year": "2023-2024",
            "year_start": 2023,
            "year_end": 2024,
            "source": "washington_post",
            "rate": rates[i],
            "enrollment": enrollments[i],
            "school_type": types[i] if types else None,
            "enrollment_basis": "unknown",
            "grade_scope": None,
            "copied_year": False,
        }
        for i in range(n)
    ]


def county_table(rows):
    df = schools(rows)
    extra = ["enrollment_basis", "grade_scope"]
    return sv.county_file(
        df.with_columns([pl.Series(c, [r[c] for r in rows]) for c in extra])
    )


def test_weighted_and_unweighted_means_and_sds_by_hand():
    # rates 90 and 100, enrollments 10 and 30
    # unweighted: mean 95, sd sqrt((25 + 25) / 2) = 5 (population form)
    # weighted: mean (900 + 3000) / 40 = 97.5,
    #           sd sqrt((10 * 7.5**2 + 30 * 2.5**2) / 40) = sqrt(18.75) = 4.3301
    row = county_table(county_rows([90.0, 100.0], [10.0, 30.0])).row(
        0, named=True
    )
    assert row["mean_unweighted"] == pytest.approx(95.0)
    assert row["sd_unweighted"] == pytest.approx(5.0)
    assert row["mean_weighted"] == pytest.approx(97.5)
    assert row["sd_weighted"] == pytest.approx(4.3301270)
    assert row["mean_accumulated"] == pytest.approx(97.5)
    assert row["sd_accumulated"] == pytest.approx(4.3301270)
    assert row["accumulated_source"] == "weighted"
    assert (
        row["total_enrollment"] == 40 and row["n_schools_with_enrollment"] == 2
    )


def test_the_county_enrollment_basis_is_mixed_when_its_schools_differ():
    same = county_rows([90.0, 100.0], [10.0, 30.0])
    differ = county_rows([90.0, 100.0], [10.0, 30.0])
    differ[1]["enrollment_basis"] = "not_denominator"
    assert county_table(same)["enrollment_basis"].to_list() == ["unknown"]
    assert county_table(differ)["enrollment_basis"].to_list() == ["mixed"]


def test_weighted_values_need_every_school_to_have_enrollment():
    row = county_table(county_rows([90.0, 100.0], [10.0, None])).row(
        0, named=True
    )
    assert row["mean_weighted"] is None and row["sd_weighted"] is None
    assert row["mean_accumulated"] == pytest.approx(95.0)
    assert row["accumulated_source"] == "unweighted"
    assert row["n_schools"] == 2 and row["n_schools_with_enrollment"] == 1
    assert row["total_enrollment"] == 10  # the known enrollments only


def test_one_school_has_an_sd_of_zero():
    row = county_table(county_rows([90.0], [10.0])).row(0, named=True)
    assert row["mean_weighted"] == pytest.approx(90.0)
    assert row["sd_unweighted"] == 0.0 and row["sd_weighted"] == 0.0
    assert row["sd_accumulated"] == 0.0


def test_unit_type_and_overall_rate_share_of_a_county():
    def table(units, rate_sources):
        rows = county_rows([90.0] * len(units), [10.0] * len(units))
        for row, unit, rate_source in zip(rows, units, rate_sources):
            row["unit"], row["rate_source"] = unit, rate_source
        return county_table(rows).row(0, named=True)

    schools_only = table(["school", "school"], ["mmr", "overall"])
    assert (
        schools_only["unit_type"] == "school"
        and schools_only["n_districts"] == 0
    )
    assert schools_only["share_overall_rate"] == pytest.approx(0.5)
    districts = table(["district", "district"], ["mmr", "mmr"])
    assert (
        districts["unit_type"] == "district" and districts["n_districts"] == 2
    )
    assert districts["share_overall_rate"] == 0.0
    mixed = table(["district", "school", "school"], ["mmr"] * 3)
    assert mixed["unit_type"] == "mixed" and mixed["n_districts"] == 1
    # Chen & Bento says neither: empty
    unknown = table([None, None], ["chen_bento", "chen_bento"])
    assert (
        unknown["unit_type"] is None
        and unknown["n_districts"] is None
        and unknown["share_overall_rate"] is None
    )


def test_public_shares_of_schools_and_of_students():
    # public 20 and 30 students, private 50: 2 / 3 of the schools, 50 / 100 of the students
    rows = county_rows(
        [90.0, 95.0, 99.0], [20.0, 30.0, 50.0], ["PUBLIC", "PUBLIC", "PRIVATE"]
    )
    row = county_table(rows).row(0, named=True)
    assert row["fraction_public"] == pytest.approx(2 / 3)
    assert row["fraction_kids_public"] == pytest.approx(0.5)
    assert row["n_schools_with_type"] == 3


def test_public_shares_need_every_school_typed_and_the_second_every_enrollment():
    unknown_type = county_rows([90.0, 95.0], [20.0, 30.0], ["PUBLIC", None])
    row = county_table(unknown_type).row(0, named=True)
    assert (
        row["fraction_public"] is None and row["fraction_kids_public"] is None
    )
    other_type = county_rows([90.0, 95.0], [20.0, 30.0], ["PUBLIC", "OTHER"])
    assert (
        county_table(other_type).row(0, named=True)["fraction_public"] is None
    )
    no_enrollment = county_rows(
        [90.0, 95.0], [20.0, None], ["PUBLIC", "PRIVATE"]
    )
    row = county_table(no_enrollment).row(0, named=True)
    assert (
        row["fraction_public"] == pytest.approx(0.5)
        and row["fraction_kids_public"] is None
    )


def test_schools_without_a_rate_count_only_in_n_schools_all():
    rows = county_rows([90.0, None, 100.0], [10.0, 5.0, 30.0])
    row = county_table(rows).row(0, named=True)
    assert row["n_schools"] == 2 and row["n_schools_all"] == 3
    assert row["mean_unweighted"] == pytest.approx(95.0)


# --- reading -----------------------------------------------------------------------


def test_chen_bento_rates_become_percent_and_censored_counts_missing():
    web_mercator = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    x, y = web_mercator.transform(-84.5, 38.0)
    raw = pl.DataFrame(
        {
            "state": ["S1", "S1", "S1", "S2", "S2", "KY", "KY"],
            "year": ["2023-2024"] * 7,
            "enrollment": ["10", "<20", "30", "5", "6", "7", "8"],
            "mmr": ["0.9", "0.8", "1", "95", "99", "0.9", "0.9"],
            "x": ["-90", "-90", "-90", "-95", "-95", str(x), str(x)],
            "y": ["35", "35", "35", "40", "40", str(y), str(y)],
            "crs": ["NAD83"] * 5 + ["EPSG:3857"] * 2,
        }
    )
    df = sv.read_chen_bento(raw, sv.QCLog()).sort("rid")
    # S1's median is 0.9 (a proportion): times 100; S2's is 97 (percent): as it is
    assert df["rate"].to_list()[:5] == pytest.approx([90, 80, 100, 95, 99])
    assert df["enrollment"].to_list()[:2] == [10.0, None]
    assert df["lon"].to_list()[5:] == pytest.approx([-84.5, -84.5])
    assert df["lat"].to_list()[5:] == pytest.approx([38.0, 38.0])
    assert df["year_start"][0] == 2023 and df["year_end"][0] == 2024
    with pytest.raises(ValueError, match="coordinate systems"):
        sv.read_chen_bento(
            raw.with_columns(crs=pl.lit("EPSG:2278")), sv.QCLog()
        )


def test_washington_post_rates_are_percent_and_fall_back_to_the_overall_rate():
    raw = pl.DataFrame(
        {
            "year": ["2024-2025", "2024-2025", "2024-2025"],
            "school_name": ["a", "b", "c"],
            "school_type": ["PUBLIC", None, "PRIVATE"],
            "address": ["x", "y", "z"],
            "students_enrolled": [10, None, 5],
            "mmr_rate": [0.9, None, None],
            "overall_rate": [0.8, 0.7, None],
            "medical_exemption_rate": [0.01, None, None],
            "religious_exemption_rate": [None] * 3,
            "personal_exemption_rate": [None] * 3,
            "nonmedical_exemption_rate": [None] * 3,
            "overall_exemption_rate": [None] * 3,
            "LAT": [35.0] * 3,
            "LON": [-90.0] * 3,
            "county": ["A", "B", "C"],
            "state": ["AR"] * 3,
            "grade": ["K"] * 3,
        }
    )
    df = sv.read_washington_post(raw, sv.QCLog()).sort("rid")
    assert df["rate"][0] == pytest.approx(90.0) and df["rate"][
        1
    ] == pytest.approx(70.0)
    assert df["rate"][2] is None
    assert df["rate_source"].to_list() == ["mmr", "overall", None]
    assert df["exemption_medical"][0] == pytest.approx(1.0)
    assert df["school_type_source"].to_list() == ["reported", None, "reported"]


# --- value checks ---------------------------------------------------------------------


def test_placeholder_rates_and_zero_enrollment():
    df = schools(
        [
            {
                "rate": 0.0,
                "enrollment": None,
            },  # missing enrollment, rate exactly 0: a placeholder, missing
            {"rate": 0.0, "enrollment": 20.0},  # a school at 0%: kept
            {
                "rate": 5.0,
                "enrollment": None,
            },  # missing enrollment, a low rate that is not 0: kept
            {
                "rate": 0.0,
                "enrollment": 0.0,
            },  # enrollment 0: enrollment and rate missing
            {
                "rate": 50.0,
                "enrollment": 0.0,
            },  # enrollment 0: the rate means nothing either, missing
            {
                "rate": 100.0,
                "enrollment": -1.0,
            },  # negative enrollment is the same
            {
                "rate": 60.0,
                "enrollment": None,
            },  # missing enrollment (not zero): the rate stays
            {"rate": 95.0, "enrollment": 30.0},
        ]
    )
    log = sv.QCLog()
    out = sv.value_checks(df, log).sort("rid")
    assert out["rate"].to_list() == [
        None,
        0.0,
        5.0,
        None,
        None,
        None,
        60.0,
        95.0,
    ]
    assert out["enrollment"].to_list() == [
        None,
        20.0,
        None,
        None,
        None,
        None,
        None,
        30.0,
    ]
    assert out["enrollment_source"].to_list() == [
        None,
        "chen_bento",
        None,
        None,
        None,
        None,
        None,
        "chen_bento",
    ]
    # three rows had an enrollment of 0 or less, and one rate was a placeholder (a zero-enrollment row is counted once)
    changed = log.rows_frame().filter(pl.col("action") == "changed")
    assert changed.height == 4
    assert (
        changed.filter(
            pl.col("reason").str.contains("enrollment of 0 or less")
        ).height
        == 3
    )


def test_rates_outside_0_to_100_are_missing_and_logged():
    # an error, not a rate near the edge: not clipped (method changed from clipping)
    log = sv.QCLog()
    out = sv.missing_rates_outside_range(
        schools(
            [
                {"rate": 103.0},
                {"rate": -0.01},
                {"rate": 50.0},
                {"rate": 100.0},
                {"rate": 0.0},
            ]
        ),
        log,
    ).sort("rid")
    assert out["rate"].to_list() == [None, None, 50.0, 100.0, 0.0]
    assert log.rows_frame().height == 2


def test_the_low_rate_placeholder_rule_is_for_chen_bento_only():
    # the Washington Post has no enrollment at all, so a low rate there is a rate
    rows = [
        {"rate": 0.0, "enrollment": None},
        {"rate": 0.0, "enrollment": 20.0},
    ]
    cb = sv.value_checks(schools(rows), sv.QCLog()).sort("rid")
    wapo = sv.value_checks(
        schools(rows, source="washington_post"), sv.QCLog()
    ).sort("rid")
    state_file = sv.value_checks(
        schools(rows, source="state_file"), sv.QCLog()
    ).sort("rid")
    assert cb["rate"].to_list() == [None, 0.0]
    assert wapo["rate"].to_list() == [0.0, 0.0]
    assert state_file["rate"].to_list() == [0.0, 0.0]


def test_a_district_row_at_exactly_0_percent_is_missing_and_a_school_row_is_not():
    df = schools(
        [
            {
                "state": "TX",
                "rate": 0.0,
                "school_name": None,
            },  # Chen & Bento Texas: a district
            {
                "state": "AL",
                "rate": 0.0,
                "school_name": None,
            },  # Chen & Bento Alabama: a school
            {
                "state": "TX",
                "rate": 3.0,
                "school_name": None,
            },  # a district, not at 0
        ]
    )
    wapo = schools(
        [
            {
                "state": "TX",
                "rate": 0.0,
                "school_name": "TEXAS TECH UNIVERSITY K-12 ISD",
            },
            {"state": "TX", "rate": 0.0, "school_name": "Small Elementary"},
        ],
        source="washington_post",
    )
    log = sv.QCLog()
    cb_out = sv.missing_district_zero_rates(df, ["TX"], log).sort("rid")
    wapo_out = sv.missing_district_zero_rates(wapo, ["TX"], log).sort("rid")
    assert cb_out["rate"].to_list() == [None, 0.0, 3.0]
    assert wapo_out["rate"].to_list() == [None, 0.0]
    assert log.rows_frame().height == 2


# --- duplicates -------------------------------------------------------------------------


def chen_bento_rows(rows):
    base = {"state": "TX", "year": "2023-2024", "lon": -97.0, "lat": 30.0}
    return schools([{**base, **r} for r in rows])


def test_chen_bento_duplicates_are_the_same_point_and_rate_unless_both_enrollments_differ():
    df = chen_bento_rows(
        [
            {"enrollment": 50.0, "rate": 90.0},
            {
                "enrollment": 50.0,
                "rate": 90.0,
            },  # same enrollment: a copy, dropped
            {
                "enrollment": 60.0,
                "rate": 90.0,
            },  # another known enrollment: a different school, kept
            {
                "enrollment": None,
                "rate": 90.0,
            },  # missing enrollment, the group has known ones: dropped
            {"enrollment": None, "rate": 95.0},
            {"enrollment": None, "rate": 95.0},  # all missing: one kept
            {"enrollment": 5.0, "rate": 100.0},
            {
                "enrollment": 5.0,
                "rate": 100.0,
            },  # tiny school at 100% is no exception: dropped
            {"enrollment": 40.0, "rate": None},
            {"enrollment": 40.0, "rate": None},  # no rate: never checked, kept
        ]
    )
    log = sv.QCLog()
    out = sv.drop_duplicates_chen_bento(df, log)
    assert out.height == 6
    assert sorted(out["rid"].to_list()) == [
        0,
        2,
        4,
        6,
        8,
        9,
    ]  # the first of each kept group
    reasons = log.rows_frame()["reason"].to_list()
    assert len(reasons) == 4 and all(
        r.startswith("identical Chen & Bento record") for r in reasons
    )
    assert (
        sum("enrollment missing" in r for r in reasons) == 2
    )  # rid 3 and rid 5


def test_records_at_a_point_that_holds_two_schools_in_another_year_are_not_copies():
    y1, y2, y3 = "2021-2022", "2022-2023", "2023-2024"
    df = chen_bento_rows(
        [
            # year 1: two records at the point with one rate and a known equal enrollment; year 2 shows two schools here: kept
            {"year": y1, "enrollment": 50.0, "rate": 90.0},
            {"year": y1, "enrollment": 50.0, "rate": 90.0},
            {"year": y2, "enrollment": 50.0, "rate": 90.0},
            {"year": y2, "enrollment": 60.0, "rate": 100.0},
            # a point where the other years show a single school: the pair of year 1 is a copy
            {"year": y1, "lon": -98.0, "enrollment": None, "rate": 95.0},
            {"year": y1, "lon": -98.0, "enrollment": None, "rate": 95.0},
            {"year": y2, "lon": -98.0, "enrollment": None, "rate": 95.0},
            {"year": y3, "lon": -98.0, "enrollment": None, "rate": 95.0},
            # another state's point does not count as evidence
            {"year": y1, "lon": -99.0, "enrollment": None, "rate": 80.0},
            {"year": y1, "lon": -99.0, "enrollment": None, "rate": 80.0},
            {
                "year": y2,
                "lon": -99.0,
                "state": "AR",
                "enrollment": None,
                "rate": 70.0,
            },
            {
                "year": y2,
                "lon": -99.0,
                "state": "AR",
                "enrollment": None,
                "rate": 60.0,
            },
        ]
    )
    log = sv.QCLog()
    out = sv.drop_duplicates_chen_bento(df, log)
    dropped = sorted(set(df["rid"].to_list()) - set(out["rid"].to_list()))
    assert (
        dropped == [5, 9]
    )  # the second record of each pair at the single-school point and at the other-state point
    rows = log.rows_frame()
    assert rows.filter(pl.col("action") == "flagged")["row_id"].to_list() == [
        "toy:1"
    ]  # kept back by the other-years test
    assert rows.filter(pl.col("action") == "dropped").height == 2


def test_a_missing_enrollment_is_dropped_only_as_a_copy_of_the_same_point_and_rate():
    df = chen_bento_rows(
        [
            {"enrollment": 50.0, "rate": 90.0},
            {"enrollment": None, "rate": 91.0},  # another rate: kept
            {
                "enrollment": None,
                "rate": 90.0,
                "lon": -98.0,
            },  # another point: kept
            {
                "enrollment": None,
                "rate": 90.0,
                "state": "AR",
            },  # another state: kept
            {
                "enrollment": None,
                "rate": 90.0,
                "year": "2022-2023",
            },  # another year: kept
        ]
    )
    assert sv.drop_duplicates_chen_bento(df, sv.QCLog()).height == 5


def test_fallback_points_are_dropped_when_many_records_are_at_least_half_of_the_county_year():
    def rows(n_at_point, n_elsewhere, county="01001", year="2023-2024"):
        point = [
            {
                "county_fips": county,
                "year": year,
                "lon": -97.0,
                "lat": 30.0,
                "rate": None,
            }
            for _ in range(n_at_point)
        ]
        spread = [
            {
                "county_fips": county,
                "year": year,
                "lon": -97.0 - 0.01 * (i + 1),
                "lat": 30.0,
                "rate": 90.0,
            }
            for i in range(n_elsewhere)
        ]
        return point + spread

    # 16 at one point of 20 records: more than 15 and at least half: all 16 dropped (a rate is not needed)
    log = sv.QCLog()
    out = sv.drop_fallback_points(chen_bento_rows(rows(16, 4)), 15, 0.5, log)
    assert out.height == 4 and log.rows_frame().height == 16
    # exactly 15 at the point: kept
    assert (
        sv.drop_fallback_points(
            chen_bento_rows(rows(15, 4)), 15, 0.5, sv.QCLog()
        ).height
        == 19
    )
    # 16 at the point but under half of the county-year (16 of 40): kept (a big county, as Cook)
    assert (
        sv.drop_fallback_points(
            chen_bento_rows(rows(16, 24)), 15, 0.5, sv.QCLog()
        ).height
        == 40
    )
    # exactly half (16 of 32): dropped
    assert (
        sv.drop_fallback_points(
            chen_bento_rows(rows(16, 16)), 15, 0.5, sv.QCLog()
        ).height
        == 16
    )
    # the same coordinate in another county-year is counted separately
    two = rows(16, 4) + rows(10, 0, year="2022-2023")
    assert (
        sv.drop_fallback_points(
            chen_bento_rows(two), 15, 0.5, sv.QCLog()
        ).height
        == 14
    )


def washington_post_rows(rows):
    base = {
        "state": "AR", "year": "2024-2025", "lon": -90.0, "lat": 35.0, "address": "1 Main St",
        "school_name": "Elm", "school_type": "PUBLIC", "enrollment": 10.0, "rate": 90.0,
        "wapo_mmr": 0.9, "wapo_overall": None,
    }  # fmt: skip
    return schools([{**base, **r} for r in rows], source="washington_post")


def test_washington_post_duplicates_need_the_same_name_point_and_rate():
    df = washington_post_rows(
        [
            {},
            {},  # same name, point, rate and enrollment: dropped
            {
                "school_type": "PRIVATE"
            },  # type and address are not part of the rule: dropped
            {
                "enrollment": None
            },  # missing enrollment, the group has a known one: dropped
            {"enrollment": 12.0},  # both known and different: kept
            {"school_name": "Oak"},  # another name: kept
            {"rate": 80.0},  # another rate: kept
        ]
    )
    log = sv.QCLog()
    out = sv.drop_duplicates_washington_post(df, log)
    assert sorted(out["rid"].to_list()) == [0, 4, 5, 6]
    assert all(
        r.startswith("identical Washington Post record")
        for r in log.rows_frame()["reason"].to_list()
    )


def test_schools_at_a_shared_point_are_counted():
    df = chen_bento_rows(
        [{"rate": 1.0}, {"rate": 2.0}, {"rate": 3.0, "lon": -98.0}]
    )
    out = sv.count_schools_at_point(df).sort("rid")
    assert out["n_at_point"].to_list() == [2, 2, 1]


# --- matching ------------------------------------------------------------------------------


def points(coordinates):
    return pl.DataFrame(
        {
            "lon": [c[0] for c in coordinates],
            "lat": [c[1] for c in coordinates],
        }
    )


def test_pairs_need_one_candidate_in_each_direction():
    a = points([(-90.0, 35.0), (-91.0, 35.0), (-92.0, 35.0)])
    b = points(
        [(-90.0001, 35.0), (-91.0, 35.0), (-91.0001, 35.0), (-92.0, 36.0)]
    )
    i, j = sv.unambiguous_pairs(a, b, 100)
    # a[1] has two schools of b within 100 m; a[2] has none; only a[0] pairs
    assert (i.tolist(), j.tolist()) == ([0], [0])
    # a school of b with two schools of a close by is not paired either
    i, j = sv.unambiguous_pairs(
        points([(-90.0, 35.0), (-90.0001, 35.0)]),
        points([(-90.00005, 35.0)]),
        100,
    )
    assert len(i) == 0


def test_pairs_do_not_depend_on_the_order_of_the_rows():
    a = points([(-90.0 - 0.01 * k, 35.0) for k in range(20)])
    b = points([(-90.0 - 0.01 * k + 0.0001, 35.0) for k in range(20)])
    i, j = sv.unambiguous_pairs(a, b, 100)
    order_a, order_b = (
        np.random.default_rng(1).permutation(20),
        np.random.default_rng(2).permutation(20),
    )
    i2, j2 = sv.unambiguous_pairs(
        a[order_a.tolist()], b[order_b.tolist()], 100
    )
    assert sorted(zip(order_a[i2].tolist(), order_b[j2].tolist())) == sorted(
        zip(i.tolist(), j.tolist())
    )


# --- borrowed enrollment and inherited types -------------------------------------------------


def test_enrollment_is_borrowed_only_from_the_same_school_with_the_same_rate():
    wapo = washington_post_rows(
        [
            {
                "lon": -90.0,
                "rate": 95.0,
                "enrollment": None,
                "school_name": "same",
            },
            {
                "lon": -91.0,
                "rate": 90.0,
                "enrollment": None,
                "school_name": "other rate",
            },
            {
                "lon": -92.0,
                "rate": 80.0,
                "enrollment": 12.0,
                "school_name": "has its own",
            },
            {
                "lon": -93.0,
                "rate": 70.0,
                "enrollment": None,
                "school_name": "no donor",
            },
        ]
    ).with_columns(year_start=pl.lit(2024))
    donors = chen_bento_rows(
        [
            {
                "lon": -90.0001,
                "rate": 95.05,
                "enrollment": 40.0,
            },  # within 0.1 point: the same school
            {
                "lon": -91.0001,
                "rate": 95.0,
                "enrollment": 41.0,
            },  # 5 points apart: another school
            {"lon": -92.0001, "rate": 80.0, "enrollment": 99.0},
        ]
    ).with_columns(
        state=pl.lit("AR"), year_start=pl.lit(2024), lat=pl.lit(35.0)
    )
    out, report = sv.borrow_enrollment(wapo, donors, CFG)
    by_name = {r["school_name"]: r for r in out.iter_rows(named=True)}
    assert (
        by_name["same"]["enrollment"] == 40.0
        and by_name["same"]["enrollment_source"] == "chen_bento"
    )
    assert by_name["other rate"]["enrollment"] is None
    assert by_name["has its own"]["enrollment"] == 12.0
    assert by_name["no donor"]["enrollment"] is None
    assert report["matched_same_rate"][0] == 2  # "same" and "has its own"


def test_nothing_is_borrowed_where_chen_bento_has_no_school_of_that_state_and_year():
    wapo = washington_post_rows([{"enrollment": None}]).with_columns(
        state=pl.lit("NY"), year_start=pl.lit(2024)
    )
    donors = chen_bento_rows(
        [{"lon": -90.0001, "rate": 90.0, "enrollment": 40.0}]
    ).with_columns(
        state=pl.lit("NY"), year_start=pl.lit(2023), lat=pl.lit(35.0)
    )  # another school year
    out, _ = sv.borrow_enrollment(wapo, donors, CFG)
    assert out["enrollment"][0] is None


def test_chen_bento_schools_inherit_the_reported_type_at_their_point():
    wapo = washington_post_rows(
        [
            {"lon": -90.0, "school_type": "PRIVATE"},
            {"lon": -91.0, "school_type": None},
            {"lon": -92.0, "school_type": "PUBLIC"},
            {
                "lon": -92.0001,
                "school_type": "PRIVATE",
            },  # two candidates at -92: refused
            {"lon": -93.0, "school_type": "PUBLIC"},
        ]
    )
    cb = chen_bento_rows(
        [
            {"lon": -90.0001, "year": "2022-2023", "n_at_point": 1},
            {
                "lon": -90.0001,
                "year": "2023-2024",
                "n_at_point": 1,
            },  # the same school in another year
            {
                "lon": -91.0001,
                "n_at_point": 1,
            },  # the Washington Post school has no type
            {"lon": -92.00005, "n_at_point": 1},
            {
                "lon": -93.0,
                "year": "2022-2023",
                "n_at_point": 2,
            },  # two schools at one point in a year: refused
            {"lon": -93.0, "year": "2022-2023", "n_at_point": 2},
            {
                "lon": -93.0,
                "year": "2023-2024",
                "n_at_point": 1,
            },  # alone in this year, but the point is shared in another
        ]
    ).with_columns(state=pl.lit("AR"), lat=pl.lit(35.0))
    out = sv.inherit_school_types(cb, wapo, CFG).sort("rid")
    assert out["school_type"].to_list() == [
        "PRIVATE",
        "PRIVATE",
        None,
        None,
        None,
        None,
        None,
    ]
    assert (
        out["school_type_source"].to_list()[:2]
        == ["inherited_washington_post"] * 2
    )


# --- source selection -----------------------------------------------------------------------
# (the method changed: one source for each state and school year, not the Washington Post
# replacing Chen & Bento county by county, and no state lists)


def selection_rows(source, state, year, n, rate=90.0, county="A"):
    """n rated schools of one source, state and school year."""
    return schools(
        [
            {
                "state": state,
                "county_fips": county,
                "year_start": year,
                "rate": rate + i * 0.01,
            }
            for i in range(n)
        ],
        source=source,
    )


def test_the_washington_post_year_is_used_and_the_other_years_come_from_chen_bento():
    cb = pl.concat(
        [
            selection_rows("chen_bento", "AL", 2022, 10),
            selection_rows("chen_bento", "AL", 2023, 10, rate=91.0),
        ]
    )
    wapo = selection_rows("washington_post", "AL", 2023, 10, rate=95.0)
    out, choice = sv.select_sources(
        cb, wapo, pl.DataFrame(schema=sv.COLUMN_TYPES), CFG
    )
    by_year = {
        r["year_start"]: r["source"] for r in choice.iter_rows(named=True)
    }
    assert by_year == {2022: "chen_bento", 2023: "washington_post"}
    assert out.filter(pl.col("year_start") == 2023)[
        "source"
    ].unique().to_list() == ["washington_post"]
    assert out.filter(pl.col("year_start") == 2022)[
        "source"
    ].unique().to_list() == ["chen_bento"]


def test_a_partial_washington_post_file_does_not_replace_chen_bento():
    # 89 of 100 schools (89%) is partial; 90 of 100 is not (params wapo_min_share_of_chen_bento 0.9)
    cb = pl.concat(
        [
            selection_rows("chen_bento", "IL", 2023, 100),
            selection_rows("chen_bento", "OH", 2023, 100),
        ]
    )
    wapo = pl.concat(
        [
            selection_rows("washington_post", "IL", 2023, 89),
            selection_rows("washington_post", "OH", 2023, 90),
        ]
    )
    out, choice = sv.select_sources(
        cb, wapo, pl.DataFrame(schema=sv.COLUMN_TYPES), CFG
    )
    chosen = {r["state"]: r["source"] for r in choice.iter_rows(named=True)}
    assert chosen == {"IL": "chen_bento", "OH": "washington_post"}
    assert (
        "89 schools, under 90%"
        in choice.filter(pl.col("state") == "IL")["why"][0]
    )
    assert (
        out.filter(pl.col("state") == "IL").height == 100
        and out.filter(pl.col("state") == "OH").height == 90
    )


def test_the_choice_is_for_the_whole_state_year_not_county_by_county():
    # the Post lacks county B: B's Chen & Bento schools are NOT kept in that school year
    cb = pl.concat(
        [
            selection_rows("chen_bento", "AL", 2023, 10, county="A"),
            selection_rows("chen_bento", "AL", 2023, 1, county="B"),
        ]
    )
    wapo = selection_rows(
        "washington_post", "AL", 2023, 10, rate=95.0, county="A"
    )
    out, _ = sv.select_sources(
        cb, wapo, pl.DataFrame(schema=sv.COLUMN_TYPES), CFG
    )
    assert out["source"].unique().to_list() == ["washington_post"]
    assert out["county_fips"].unique().to_list() == ["A"]


def test_the_state_files_are_used_only_where_neither_other_source_has_the_state_year():
    cb = selection_rows("chen_bento", "AL", 2023, 10)
    wapo = selection_rows("washington_post", "NY", 2023, 10)
    sf = pl.concat(
        [
            selection_rows("state_file", "AL", 2023, 10),
            selection_rows("state_file", "NY", 2023, 10),
            selection_rows("state_file", "NY", 2022, 10),
            selection_rows("state_file", "MD", 2022, 10),
        ]
    )
    out, choice = sv.select_sources(cb, wapo, sf, CFG)
    chosen = {
        (r["state"], r["year_start"]): r["source"]
        for r in choice.iter_rows(named=True)
    }
    assert chosen == {
        ("AL", 2023): "chen_bento",
        ("NY", 2023): "washington_post",
        ("NY", 2022): "state_file",
        ("MD", 2022): "state_file",
    }
    assert out.height == 40


def test_a_washington_post_year_without_rated_schools_does_not_count():
    cb = selection_rows("chen_bento", "AL", 2023, 10)
    wapo = selection_rows("washington_post", "AL", 2023, 10).with_columns(
        rate=pl.lit(None, dtype=pl.Float64)
    )
    out, choice = sv.select_sources(
        cb, wapo, pl.DataFrame(schema=sv.COLUMN_TYPES), CFG
    )
    assert choice["source"].to_list() == ["chen_bento"]
    assert out["source"].unique().to_list() == ["chen_bento"]


def test_a_partial_state_file_year_is_not_used_but_a_state_without_other_sources_is():
    # AL has Chen & Bento (100 schools in 2023); its state file has 50 in 2022 (partial) and 95 in 2024 (not)
    cb = selection_rows("chen_bento", "AL", 2023, 100)
    sf = pl.concat(
        [
            selection_rows("state_file", "AL", 2022, 50),
            selection_rows("state_file", "AL", 2024, 95),
            selection_rows(
                "state_file", "NY", 2022, 10
            ),  # no other source in NY: nothing to compare with
        ]
    )
    out, choice = sv.select_sources(
        cb, pl.DataFrame(schema=sv.COLUMN_TYPES), sf, CFG
    )
    by = {
        (r["state"], r["year_start"]): r["source"]
        for r in choice.iter_rows(named=True)
    }
    assert by == {
        ("AL", 2022): None,
        ("AL", 2023): "chen_bento",
        ("AL", 2024): "state_file",
        ("NY", 2022): "state_file",
    }
    assert (
        "50 schools, under 90%"
        in choice.filter(pl.col("year_start") == 2022)["why"][0]
    )
    assert (
        out.filter(
            (pl.col("state") == "AL") & (pl.col("year_start") == 2022)
        ).height
        == 0
    )


# --- places ----------------------------------------------------------------------------------


def counties_frame(rows):
    return gpd.GeoDataFrame(
        {
            "county_fips": [r[0] for r in rows],
            "county_name": [r[1] for r in rows],
            "county_state": [r[2] for r in rows],
        },
        geometry=[box(*r[3]) for r in rows],
        crs="EPSG:4269",
    )


def test_a_school_always_gets_a_county_of_its_own_state():
    counties = counties_frame(
        [
            ("AA001", "One", "AA", (-100.1, 40.0, -100.0, 40.1)),
            ("BB001", "Two", "BB", (-99.9, 40.0, -99.8, 40.1)),
        ]
    )
    df = schools(
        [
            {
                "state": "AA",
                "lon": -100.05,
                "lat": 40.05,
            },  # inside its own state's county
            {
                "state": "AA",
                "lon": -99.85,
                "lat": 40.05,
            },  # inside BB's county, 13 km from AA's
            {
                "state": "AA",
                "lon": -100.15,
                "lat": 40.05,
            },  # in no county, 4 km from AA's
            {
                "state": "AA",
                "lon": -98.0,
                "lat": 40.05,
            },  # 170 km from AA: dropped
        ]
    )
    log = sv.QCLog()
    out = sv.assign_counties(df, counties, 20, log).sort("rid")
    assert out["rid"].to_list() == [0, 1, 2]
    assert out["county_fips"].to_list() == ["AA001"] * 3
    assert out["county_join"].to_list() == [
        "contained",
        "nearest_in_state",
        "nearest_in_state",
    ]
    assert log.rows_frame().filter(pl.col("action") == "dropped").height == 1


def test_county_names_are_normalised():
    assert sv.normalize_county_name("Doña Ana") == "DONAANA"
    assert sv.normalize_county_name("Richmond city") == "RICHMOND"
    assert sv.normalize_county_name("St. Louis") == sv.normalize_county_name(
        "Saint Louis County"
    )
    assert (
        sv.normalize_county_name("City") == "CITY"
    )  # a name that is only the suffix stays


def test_a_label_that_names_two_counties_agrees_with_either():
    counties = counties_frame(
        [
            ("51760", "Richmond city", "VA", (-77.5, 37.5, -77.4, 37.6)),
            ("51159", "Richmond County", "VA", (-77.3, 37.5, -77.2, 37.6)),
            ("51001", "Accomack", "VA", (-77.1, 37.5, -77.0, 37.6)),
            ("51099", "King George", "VA", (-75.05, 37.5, -74.95, 37.6)),
        ]
    )
    df = schools(
        [
            {
                "state": "VA",
                "county_label": "RICHMOND",
                "county_fips": "51159",
                "lon": -77.25,
                "lat": 37.55,
            },
            {
                "state": "VA",
                "county_label": "RICHMOND",
                "county_fips": "51760",
                "lon": -77.45,
                "lat": 37.55,
            },
            # the point is in Accomack, 9 km from Richmond County: kept, flagged
            {
                "state": "VA",
                "county_label": "RICHMOND",
                "county_fips": "51001",
                "lon": -77.05,
                "lat": 37.55,
            },
            # the point is in King George, 200 km from either: dropped
            {
                "state": "VA",
                "county_label": "RICHMOND",
                "county_fips": "51099",
                "lon": -75.0,
                "lat": 37.55,
            },
            {
                "state": "VA",
                "county_label": "Nowhere",
                "county_fips": "51001",
                "lon": -77.05,
                "lat": 37.55,
            },
            {
                "state": "CT",
                "county_label": "CAPITOL",
                "county_fips": "09003",
                "lon": -72.7,
                "lat": 41.7,
            },
        ]
    )
    counties = pd_concat_state(counties, "CT")
    log = sv.QCLog()
    out = sv.check_county_labels(df, counties, 20, log).sort("rid")
    assert out["rid"].to_list() == [0, 1, 2, 4, 5]
    assert out["county_label_differs"].to_list() == [
        False,
        False,
        True,
        None,
        None,
    ]


def pd_concat_state(counties, state):
    extra = counties_frame(
        [("09003", "Hartford", state, (-72.8, 41.6, -72.6, 41.8))]
    )
    return gpd.GeoDataFrame(
        pl.concat(
            [
                pl.from_pandas(counties.drop(columns="geometry")),
                pl.from_pandas(extra.drop(columns="geometry")),
            ]
        ).to_pandas(),
        geometry=list(counties.geometry) + list(extra.geometry),
        crs=counties.crs,
    )


# --- copied years ------------------------------------------------------------------------------


def series_rows(state, years_rates, n=40):
    rows = []
    for year, shift in years_rates:
        for k in range(n):
            rows.append(
                {
                    "state": state, "year": f"{year}-{year + 1}", "year_start": year,
                    "lon": -90.0 - 0.01 * k, "lat": 35.0, "rate": 60.0 + k + shift,
                }
            )  # fmt: skip
    return rows


def test_a_year_that_repeats_the_year_before_is_flagged_and_kept():
    rows = series_rows("ZZ", [(2020, 0.0), (2021, 0.0), (2022, 0.5)])
    rows += [  # schools at 100% in every year never count as unchanged
        {
            "state": "ZZ",
            "year": f"{y}-{y + 1}",
            "year_start": y,
            "lon": -80.0,
            "lat": 35.0,
            "rate": 100.0,
        }
        for y in (2020, 2021, 2022)
    ]
    df = schools(rows)
    out, checks = sv.flag_copied_years(df, 0.99, 30)
    assert out.height == df.height
    flagged = (
        out.filter(pl.col("copied_year"))["year_start"].unique().to_list()
    )
    assert flagged == [2021]
    assert checks.filter(pl.col("copied_year"))[
        "year_start_second"
    ].to_list() == [2021]


def test_a_repeat_among_few_schools_is_not_flagged():
    df = schools(series_rows("ZZ", [(2020, 0.0), (2021, 0.0)], n=10))
    out, _ = sv.flag_copied_years(df, 0.99, 30)
    assert not out["copied_year"].any()


# --- flags and the enrollment test ------------------------------------------------------------


def test_flags_unit_grade_scope_enrollment_basis_and_low_rates():
    rows = [
        {
            "state": "IA",
            "year_start": 2024,
            "school_name": f"Elm {k}",
            "rate": 95.0,
            "enrollment": 100.0,
            "source": "washington_post",
            "enrollment_source": "washington_post",
        }
        for k in range(10)
    ]
    rows.append(
        {
            "state": "IA",
            "year_start": 2024,
            "school_name": "Brown ISD",
            "rate": 30.0,
            "enrollment": 100.0,
            "source": "washington_post",
            "enrollment_source": "washington_post",
        }
    )
    rows.append(
        {
            "state": "IA",
            "year_start": 2024,
            "school_name": "Tiny",
            "rate": 5.0,
            "enrollment": 8.0,
            "source": "washington_post",
            "enrollment_source": "washington_post",
        }
    )
    rows.append(
        {
            "state": "TX",
            "year_start": 2024,
            "school_name": None,
            "rate": 99.0,
            "enrollment": 50.0,
            "source": "chen_bento",
            "enrollment_source": "chen_bento",
        }
    )
    # a Washington Post school (Texas) whose enrollment was borrowed takes Chen & Bento's basis, not the Washington Post's
    rows.append(
        {
            "state": "TX",
            "year_start": 2024,
            "school_name": "Oak",
            "rate": 90.0,
            "enrollment": 60.0,
            "source": "washington_post",
            "enrollment_source": "chen_bento",
        }
    )
    rows.append(
        {
            "state": "AR",
            "year_start": 2024,
            "school_name": None,
            "rate": 97.0,
            "enrollment": 40.0,
            "source": "chen_bento",
            "enrollment_source": "chen_bento",
        }
    )
    out = sv.add_flags(schools(rows), CFG).sort("rid")
    brown, tiny, texas, borrowed, arkansas = (
        out[10],
        out[11],
        out[12],
        out[13],
        out[14],
    )
    assert borrowed["enrollment_basis"][0] == "not_denominator"
    assert brown["unit"][0] == "district" and out["unit"][0] == "school"
    # Chen & Bento has no names: its unit comes from the state (Texas: districts)
    assert texas["unit"][0] == "district" and arkansas["unit"][0] == "school"
    assert brown["grade_scope"][0] == "secondary_included;all_grade_enrollment"
    assert (
        out["enrollment_basis"].to_list()[:1] == ["denominator"]
        and texas["enrollment_basis"][0] == "not_denominator"
    )
    # 30%: below 50 in a school of 100 where enrollment is the denominator, and 65 points under the median
    assert (
        brown["low_rate_flag"][0]
        == "low_in_large_school;far_below_state_median"
    )
    # a rate below 10 with known enrollment (8) is flagged; it is too small for the other two flags
    assert tiny["low_rate_flag"][0] == "below_placeholder_rate"
    assert out["low_rate_flag"][0] is None


def test_the_enrollment_denominator_test():
    # 3 of 7 students is 42.857142857 percent: a whole number of students
    whole = [
        {"state": "AA", "enrollment": 7.0, "rate": 100 * 3 / 7}
        for _ in range(40)
    ]
    off = [
        {"state": "BB", "enrollment": 7.0, "rate": 42.5} for _ in range(40)
    ]  # 2.975 students
    percent = [
        {"state": "CC", "enrollment": 7.0, "rate": 43.0} for _ in range(40)
    ]  # a whole percent: untestable
    out = sv.enrollment_denominator_test(schools(whole + off + percent))
    assert dict(zip(out["state"], out["enrollment_basis"])) == {
        "AA": "denominator",
        "BB": "not_denominator",
    }


def test_a_single_point_projects_like_the_same_point_among_others():
    one = sv.project(points([(-90.0, 35.0)]))
    several = sv.project(points([(-90.0, 35.0), (-91.0, 36.0)]))
    assert one.shape == (1, 2) and one[0] == pytest.approx(several[0])


def test_a_nan_rate_is_missing_and_never_becomes_100_percent():
    raw = pl.DataFrame(
        {
            "state": ["S1"] * 4,
            "year": ["2023-2024"] * 4,
            "enrollment": ["10", "NaN", "20", "30"],
            "mmr": ["0.9", "0.8", "NaN", "0.95"],
            "x": ["-90"] * 4,
            "y": ["35"] * 4,
            "crs": ["NAD83"] * 4,
        }
    )
    log = sv.QCLog()
    df = sv.read_chen_bento(raw, log).sort("rid")
    assert (
        df["rate"].to_list()[0] == pytest.approx(90.0)
        and df["rate"][2] is None
    )
    assert df["enrollment"][1] is None
    assert (
        not df["rate"].is_nan().any() and not df["enrollment"].is_nan().any()
    )
    clipped = sv.missing_rates_outside_range(df, log).sort("rid")
    assert (
        clipped["rate"][2] is None
    )  # a missing rate stays missing (and never becomes 100)
    assert (
        log.rows_frame()
        .filter(pl.col("reason") == "NaN treated as missing")
        .height
        == 2
    )


def test_the_borrow_report_counts_the_counties_that_can_be_weighted():
    wapo = washington_post_rows(
        [
            {
                "lon": -90.0,
                "rate": 95.0,
                "enrollment": None,
                "county_fips": "A",
            },
            {
                "lon": -91.0,
                "rate": 90.0,
                "enrollment": None,
                "county_fips": "B",
            },
        ]
    ).with_columns(year_start=pl.lit(2024))
    donors = chen_bento_rows(
        [
            {"lon": -90.0001, "rate": 95.0, "enrollment": 40.0},
            {"lon": -91.0001, "rate": 80.0, "enrollment": 41.0},
        ]
    ).with_columns(
        state=pl.lit("AR"), year_start=pl.lit(2024), lat=pl.lit(35.0)
    )
    _, report = sv.borrow_enrollment(wapo, donors, CFG)
    row = report.row(0, named=True)
    assert (
        row["counties"] == 2
        and row["counties_weightable_before"] == 0
        and row["counties_weightable_after"] == 1
    )
    assert row["without_enrollment"] == 2 and row["matched_same_rate"] == 1


def test_county_mmr_qc_flag():
    # school means of 90 in the school year 2024-2025 (year_start 2024) and county MMRs of the same school year:
    # 99.5 (gap 9.5), 100 (exactly 10), 78.4 (gap -11.6), 95 but imputed, and none
    county = pl.DataFrame(
        {
            "county_fips": ["01001", "01003", "01005", "01007", "01009"],
            "year_start": [2024] * 5,
            "year_end": [2025] * 5,
            "mean_unweighted": [90.0] * 5,
        }
    )
    county_mmr = pl.DataFrame(
        {
            "fips": ["01003", "01001", "01005", "01001", "01007"],
            "year": [
                2024,
                2024,
                2024,
                2025,
                2024,
            ],  # the 2025 value of 01001 is the next school year
            "estimated_mmr_cov": [100.0, 99.5, 78.4, 50.0, 95.0],
            "is_imputed": [False, False, False, False, True],
        }
    )
    out = sv.add_county_mmr_qc(county, county_mmr, 10)
    assert out["county_fips"].to_list() == [
        "01001",
        "01003",
        "01005",
        "01007",
        "01009",
    ]  # the order is kept
    assert out["county_mmr"].to_list() == [
        99.5,
        100.0,
        78.4,
        None,
        None,
    ]  # an imputed value is not used
    assert out["county_mmr_imputed"].to_list() == [
        False,
        False,
        False,
        True,
        None,
    ]
    assert out["mean_gap_pp"][0] == pytest.approx(9.5)
    assert out["mean_gap_pp"][1] == pytest.approx(10.0)
    assert out["mean_gap_pp"][2] == pytest.approx(-11.6)
    assert out["mean_gap_pp"][3] is None and out["mean_gap_pp"][4] is None
    # 10 points exactly passes; imputed or no county MMR: inconclusive (empty), neither ok nor not ok
    assert out["qc_gap_ok"].to_list() == [True, True, False, None, None]
