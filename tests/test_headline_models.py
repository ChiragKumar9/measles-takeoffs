"""The choice of the headline models and how the scripts read it
(outbreak_size.headline_models)."""

import json

import polars as pl
import pytest

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.headline_models import (
    CHOOSE_SCRIPT,
    HEADLINE_FILE_NAME,
    ask_choice,
    candidate_table,
    changed_tables,
    check_names,
    mechanistic_model_names,
    outcome_counts,
    read_headline_models,
    write_headline_models,
)
from outbreak_size.model_designs import DESIGNS_BY_NAME
from outbreak_size.model_selection import (
    SUMMARY_LABEL,
    WEIGHTED_GAP_COLUMN,
    YEARS,
)

STATISTICAL = "Statistical: All covariates + spatial + state-specific vaccine coefficients, trained since 2019"
OTHER = "Statistical: MMR coverage, trained on last two years"
MECHANISTIC = "Mechanistic: MMR coverage, trained since 2019"
MULTI_CHAIN = (
    "Mechanistic with multi-chain outbreaks: MMR coverage, trained since 2019"
)


@pytest.fixture
def tables(tmp_path):
    s1 = tmp_path / "s1.csv"
    s2 = tmp_path / "s2.csv"
    s1.write_text("a,b\n1,2\n")
    s2.write_text("c,d\n3,4\n")
    return {"S1": str(s1), "S2": str(s2)}


def write(tmp_path, tables, statistical=STATISTICAL, mechanistic=MECHANISTIC):
    path = str(tmp_path / HEADLINE_FILE_NAME)
    write_headline_models(
        path, statistical, mechanistic, "the reason", tables, "2026-11-10"
    )
    return path


# --- writing and reading the choice ---------------------------------------------


def test_the_choice_is_read_back_as_it_was_written(tmp_path, tables):
    headline = read_headline_models(write(tmp_path, tables))
    assert headline.statistical == DESIGNS_BY_NAME[STATISTICAL]
    assert headline.mechanistic == MECHANISTIC
    assert headline.mechanistic_start_year == 2019
    assert headline.rationale == "the reason"
    assert headline.chosen_on == "2026-11-10"
    assert set(headline.tables) == {"S1", "S2"}


def test_a_missing_choice_says_which_script_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="06b_choose_headline_models"):
        read_headline_models(str(tmp_path / HEADLINE_FILE_NAME))
    assert "06b_choose_headline_models" in CHOOSE_SCRIPT


def test_an_unknown_statistical_model_is_refused(tmp_path, tables):
    with pytest.raises(ValueError, match="not one of the"):
        write(tmp_path, tables, statistical="Statistical: nothing")


def test_only_a_single_chain_mechanistic_model_can_be_the_headline(
    tmp_path, tables
):
    assert len(mechanistic_model_names()) == 3
    assert MECHANISTIC in mechanistic_model_names()
    with pytest.raises(ValueError, match="several chains"):
        write(tmp_path, tables, mechanistic=MULTI_CHAIN)
    with pytest.raises(ValueError, match="cannot be the headline"):
        check_names(STATISTICAL, "Mechanistic: something else")


def test_a_file_with_a_name_that_is_no_longer_a_model_is_refused(
    tmp_path, tables
):
    path = write(tmp_path, tables)
    with open(path) as f:
        record = json.load(f)
    record["statistical"]["name"] = "Statistical: renamed"
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.raises(ValueError, match="not one of the"):
        read_headline_models(path)


def test_a_recorded_design_that_changed_is_refused(tmp_path, tables):
    path = write(tmp_path, tables)
    with open(path) as f:
        record = json.load(f)
    record["statistical"]["design"]["pooled"] = not record["statistical"][
        "design"
    ]["pooled"]
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.raises(AssertionError, match="not the design"):
        read_headline_models(path)


# --- tables that changed after the choice ---------------------------------------


def test_the_tables_of_the_choice_are_recognized_by_their_contents(
    tmp_path, tables
):
    headline = read_headline_models(write(tmp_path, tables))
    assert changed_tables(headline, tables) == []
    with open(tables["S2"], "a") as f:
        f.write("5,6\n")
    assert changed_tables(headline, tables) == ["S2"]
    # a table the choice does not mention counts as changed
    assert changed_tables(headline, {**tables, "S9": tables["S1"]}) == [
        "S2",
        "S9",
    ]


# --- asking for a choice --------------------------------------------------------


def answers(*lines):
    it = iter(lines)

    def ask(_question):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return ask


def test_a_choice_is_a_number_or_a_name():
    options = ["a", "b", "c"]
    shown = []
    assert ask_choice("?", options, answers("2"), shown.append) == "b"
    assert ask_choice("?", options, answers(" c "), shown.append) == "c"
    assert ask_choice("?", options, answers("a"), shown.append) == "a"
    # the options are listed with their numbers
    assert shown[:3] == ["   1. a", "   2. b", "   3. c"]


def test_a_name_not_listed_is_accepted_but_has_no_number():
    shown = []
    options = ["a", "b", "c", "d"]
    listed = options[:2]
    assert (
        ask_choice("?", options, answers("d"), shown.append, listed=listed)
        == "d"
    )
    # 3 is not a number of the list shown, 2 is
    assert (
        ask_choice(
            "?", options, answers("3", "2"), shown.append, listed=listed
        )
        == "b"
    )
    assert sum("is not a number" in line for line in shown) == 1


def test_a_wrong_answer_is_asked_again():
    shown = []
    chosen = ask_choice(
        "?", ["a", "b"], answers("0", "3", "x", "", "2"), shown.append
    )
    assert chosen == "b"
    assert sum("is not a number" in line for line in shown) == 4


def test_no_more_input_stops_the_script_with_a_message():
    with pytest.raises(SystemExit, match="no answer"):
        ask_choice("which?", ["a"], answers(), lambda _line: None)


# --- the tables shown with the choice -------------------------------------------


def toy_s2():
    rows = []
    for model, pool, gap in (
        (STATISTICAL, "Statistical", 1.0),
        (OTHER, "Statistical", 0.5),
        (MECHANISTIC, "Mechanistic", 0.2),
    ):
        row = {"Pool": pool, "Model": model, WEIGHTED_GAP_COLUMN: gap}
        row.update({f"AUROC, 5+ (%), {y}": 80.0 + y - 2021 for y in YEARS})
        rows.append(row)
    return pl.from_dicts(rows)


def toy_s1():
    rows = []
    for model in (STATISTICAL, OTHER, MECHANISTIC):
        for condition, shift in zip(CONDITIONS, (0.0, 9.0)):
            rows.append(
                {
                    "Model": model,
                    "Evaluation year": SUMMARY_LABEL,
                    "Condition": condition,
                    "AUROC, 1+ (%)": 70.0 + shift,
                    "AUROC, 5+ (%)": 71.0 + shift,
                    "AUROC, 10+ (%)": 72.0 + shift,
                    "AUPRC, 5+ (%)": 5.0 + shift,
                }
            )
        rows.append(
            {
                "Model": model,
                "Evaluation year": "2021-2025",  # another window: not shown
                "Condition": CONDITIONS[0],
                "AUROC, 1+ (%)": 1.0,
                "AUROC, 5+ (%)": 1.0,
                "AUROC, 10+ (%)": 1.0,
                "AUPRC, 5+ (%)": 1.0,
            }
        )
    return pl.from_dicts(rows)


def test_the_candidates_are_listed_with_the_smallest_gap_first():
    table = candidate_table(toy_s1(), toy_s2(), "Statistical")
    assert table["Model"].to_list() == [OTHER, STATISTICAL]
    assert table[WEIGHTED_GAP_COLUMN].to_list() == [0.5, 1.0]
    # the S1 means are those of the window of the choice, the cases with imports
    assert table["AUROC, 5+ (%)"].to_list() == [71.0, 71.0]
    assert candidate_table(toy_s1(), toy_s2(), "Statistical", limit=1)[
        "Model"
    ].to_list() == [OTHER]
    mechanistic = candidate_table(toy_s1(), toy_s2(), "Mechanistic")
    assert mechanistic["Model"].to_list() == [MECHANISTIC]
    assert f"AUROC, 5+ (%), {YEARS[0]}" in table.columns


def test_the_outcome_counts_are_those_of_the_cases_including_imports():
    counts = pl.from_dicts(
        [
            {
                "Evaluation year": 2023,
                "Condition": condition,
                "Places scored": scored,
                "Places with 1+ cases": 29,
                "Places with 3+ cases": 5,
                "Places with 5+ cases": 4,
                "Places with 10+ cases": 1,
                "Places with 20+ cases": 0,
            }
            for condition, scored in zip(CONDITIONS, (3136, 29))
        ]
    )
    out = outcome_counts(counts)
    assert out.height == 1
    assert out["Places scored"].item() == 3136
    assert "Places with 3+ cases" not in out.columns


def test_a_table_written_again_after_the_choice_is_warned_about(
    tmp_path, capsys
):
    from outbreak_size.headline_models import TABLE_FILES

    figures = tmp_path / "figures"
    figures.mkdir()
    paths = {}
    for label, name in TABLE_FILES.items():
        (figures / name).write_text(f"{label} as chosen\n")
        paths[label] = str(figures / name)
    path = write(tmp_path, paths)
    # the tables as they were: no warning
    read_headline_models(path, fig_dir=str(figures))
    assert "WARNING" not in capsys.readouterr().out
    # a table that is written again: a warning that names it, and the choice
    # is read all the same
    (figures / TABLE_FILES["S2"]).write_text("S2 written again\n")
    headline = read_headline_models(path, fig_dir=str(figures))
    out = capsys.readouterr().out
    assert headline.statistical == DESIGNS_BY_NAME[STATISTICAL]
    assert "WARNING: S2 is not the table" in out
    assert CHOOSE_SCRIPT in out
    # a table that is not there cannot be compared: no warning for it
    (figures / TABLE_FILES["S2"]).unlink()
    read_headline_models(path, fig_dir=str(figures))
    assert "WARNING" not in capsys.readouterr().out
    # without fig_dir nothing is compared
    read_headline_models(path)
    assert capsys.readouterr().out == ""


# --- the tables as text ---------------------------------------------------------


def test_the_names_are_shortened_without_changing_what_they_say():
    from outbreak_size.headline_models import short_name

    assert short_name(STATISTICAL) == (
        "All covariates + spatial + state-specific vaccine coefficients, "
        "since 2019"
    )
    assert short_name(OTHER) == "MMR coverage, last two years"
    assert short_name(MECHANISTIC) == "MMR coverage, since 2019"
    assert short_name(MULTI_CHAIN) == "multi-chain: MMR coverage, since 2019"


def test_the_candidates_are_numbered_in_the_order_of_the_table():
    from outbreak_size.headline_models import format_candidates

    table = candidate_table(toy_s1(), toy_s2(), "Statistical")
    lines = format_candidates(table)
    # the caption, the header and a row for each model, the smallest gap first
    assert len(lines) == 2 + table.height
    assert lines[2].lstrip().startswith("1  MMR coverage, last two years")
    assert lines[3].lstrip().startswith("2  All covariates + spatial")
    assert "0.50" in lines[2] and "1.00" in lines[3]
    # every row has the same width, so the columns line up
    assert len({len(line) for line in lines[1:]}) == 1


def test_a_model_that_cannot_be_chosen_has_no_number():
    from outbreak_size.headline_models import format_candidates

    s2 = pl.concat(
        [
            toy_s2(),
            toy_s2()
            .filter(pl.col("Pool") == "Mechanistic")
            .with_columns(
                pl.lit(MULTI_CHAIN).alias("Model"),
                pl.lit(0.1).alias(WEIGHTED_GAP_COLUMN),
            ),
        ]
    )
    s1 = pl.concat(
        [
            toy_s1(),
            toy_s1()
            .filter(pl.col("Model") == MECHANISTIC)
            .with_columns(pl.lit(MULTI_CHAIN).alias("Model")),
        ]
    )
    table = candidate_table(s1, s2, "Mechanistic")
    lines = format_candidates(table, numbered=[MECHANISTIC])
    multi = next(line for line in lines if "multi-chain" in line)
    single = next(
        line for line in lines if "since 2019" in line and "multi" not in line
    )
    assert multi.startswith("    ") or multi[:2].strip() == ""
    assert single.lstrip().startswith("1  ")


def test_a_missing_score_is_a_dash():
    from outbreak_size.headline_models import format_candidates

    s1 = toy_s1().filter(pl.col("Model") != OTHER)
    lines = format_candidates(candidate_table(s1, toy_s2(), "Statistical"))
    assert " -" in next(line for line in lines if "last two years" in line)


def test_the_places_with_cases_are_one_line_a_year():
    from outbreak_size.headline_models import format_outcome_counts

    counts = pl.from_dicts(
        [
            {
                "Evaluation year": year,
                "Places scored": 3000 + year,
                "Places with 1+ cases": 20,
                "Places with 5+ cases": 4,
                "Places with 10+ cases": 1,
            }
            for year in (2022, 2023)
        ]
    )
    lines = format_outcome_counts(counts)
    assert len(lines) == 3
    assert "5022" in lines[1] and "5023" in lines[2]


def test_the_menu_can_leave_the_list_to_the_table_the_user_sees():
    shown = []
    chosen = ask_choice(
        "?", ["a", "b"], answers("2"), shown.append, show_list=False
    )
    assert chosen == "b"
    assert shown == []
