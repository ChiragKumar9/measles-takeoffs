"""Script 06b: it prints the tables, asks for the models and writes the choice,
run as the user runs it (a process with the tables in figures/)."""

import json
import os
import subprocess
import sys

import polars as pl
import pytest

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.model_selection import (
    LARGEST_GAP_COLUMN,
    MEAN_AUROC_COLUMN,
    MEAN_GAP_COLUMN,
    SUMMARY_LABEL,
    YEARS,
)

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "analysis",
    "06b_choose_headline_models.py",
)
A = "Statistical: All covariates + spatial + state-specific vaccine coefficients, trained since 2019"
B = "Statistical: MMR coverage, trained on last two years"
C = "Statistical: MMR coverage + exemptions, trained since 2011"
M = "Mechanistic: MMR coverage, trained since 2019"
M_LAST_TWO = "Mechanistic: MMR coverage, trained on last two years"
M_MULTI = (
    "Mechanistic with multi-chain outbreaks: MMR coverage, trained since 2019"
)
GAPS = {A: 0.4, B: 0.1, C: 0.9, M: 0.3, M_LAST_TWO: 0.0, M_MULTI: 0.2}


def write_tables(directory):
    figures = os.path.join(directory, "figures")
    os.makedirs(figures)
    s2 = []
    for model, gap in GAPS.items():
        row = {
            "Pool": "Statistical"
            if model.startswith("Stat")
            else "Mechanistic",
            "Model": model,
            MEAN_AUROC_COLUMN: 80.0 - gap,
            MEAN_GAP_COLUMN: gap,
            LARGEST_GAP_COLUMN: 2 * gap,
        }
        row.update({f"AUROC, 5+ (%), {y}": 80.0 for y in YEARS})
        s2.append(row)
    pl.from_dicts(s2).write_csv(
        os.path.join(
            figures, "supplemental_data_s2_model_consistency_across_years.csv"
        )
    )
    s1 = [
        {
            "Model": model,
            "Evaluation year": SUMMARY_LABEL,
            "Condition": CONDITIONS[0],
            "AUROC, 1+ (%)": 70.0,
            "AUROC, 5+ (%)": 71.0,
            "AUROC, 10+ (%)": 72.0,
            "AUPRC, 5+ (%)": 5.0,
            "AUPRC, 10+ (%)": 4.0,
        }
        for model in GAPS
    ]
    pl.from_dicts(s1).write_csv(
        os.path.join(
            figures, "supplemental_data_s1_county_level_prediction_scoring.csv"
        )
    )
    counts = [
        {
            "Evaluation year": year,
            "Condition": condition,
            "Places scored": 3000,
            "Places with 1+ cases": 20,
            "Places with 3+ cases": 10,
            "Places with 5+ cases": 4,
            "Places with 10+ cases": 1,
            "Places with 20+ cases": 0,
        }
        for year in YEARS
        for condition in CONDITIONS
    ]
    pl.from_dicts(counts).write_csv(
        os.path.join(figures, "table_supplemental_outcome_counts.csv")
    )


def run(directory, stdin="", *args):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=directory,
        input=stdin,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def directory(tmp_path):
    write_tables(str(tmp_path))
    return str(tmp_path)


def written(directory):
    with open(os.path.join(directory, "outputs", "headline_models.json")) as f:
        return json.load(f)


def test_the_user_chooses_by_number_and_the_choice_is_written(directory):
    # the statistical models are listed by gap: B (0.1), A (0.4), C (0.9);
    # the mechanistic menu is the single-chain models by gap: M_LAST_TWO (0.0),
    # M (0.3)
    done = run(directory, "2\n2\nit is the model of the paper\ny\n")
    assert done.returncode == 0, done.stderr
    record = written(directory)
    assert record["statistical"]["name"] == A
    assert record["mechanistic"]["name"] == M
    assert record["rationale"] == "it is the model of the paper"
    assert set(record["tables"]) == {"S1", "S2"}
    # the tables shown: the places with cases, and the models with their gaps
    assert "with 1+ cases" in done.stdout
    assert "multi-chain: MMR coverage, since 2019" in done.stdout
    # the models are numbered once, in the tables: the menu does not list them
    # again, and the shortened names have no "Statistical: " in front
    assert done.stdout.count("MMR coverage, last two years") <= 4
    assert (
        "Statistical: MMR coverage, trained"
        not in done.stdout.split(
            "The statistical model the table ranks first"
        )[0]
    )
    assert f"The statistical model the table ranks first: {B}" in done.stdout


def test_a_name_can_be_typed_and_the_choice_can_be_refused(directory):
    done = run(directory, f"{C}\n2\nbecause\nn\n")
    assert done.returncode != 0
    assert "nothing was written" in done.stderr
    assert not os.path.exists(os.path.join(directory, "outputs"))


def test_the_models_can_be_given_for_a_run_without_questions(directory):
    done = run(
        directory,
        "",
        "--statistical",
        B,
        "--mechanistic",
        M_LAST_TWO,
        "--rationale",
        "chosen on 2022-2025",
    )
    assert done.returncode == 0, done.stderr
    assert written(directory)["mechanistic"]["name"] == M_LAST_TWO


def test_a_multi_chain_model_is_not_accepted_as_the_headline(directory):
    done = run(
        directory,
        "",
        "--statistical",
        A,
        "--mechanistic",
        M_MULTI,
        "--rationale",
        "x",
    )
    assert done.returncode != 0
    assert "several chains" in done.stderr
    assert not os.path.exists(os.path.join(directory, "outputs"))


def test_no_reason_means_no_choice(directory):
    done = run(directory, "1\n1\n\n")
    assert done.returncode != 0
    assert "a reason is needed" in done.stderr
    assert not os.path.exists(os.path.join(directory, "outputs"))


def test_without_the_tables_it_says_which_script_makes_them(tmp_path):
    done = run(str(tmp_path))
    assert done.returncode != 0
    assert "table_model_performance_validation.py" in done.stderr


def test_a_choice_made_before_is_shown_and_replaced_only_on_yes(directory):
    first = run(directory, "2\n2\nthe first reason\ny\n")
    assert first.returncode == 0, first.stderr
    assert "now chosen" not in first.stdout
    assert "Write outputs" in first.stdout
    # a second run shows the choice and the reason, and asks to replace it
    refused = run(directory, "1\n1\nthe second reason\nn\n")
    assert "now chosen" in refused.stdout
    assert "the first reason" in refused.stdout
    assert "Replace outputs/headline_models.json?" in refused.stdout
    assert refused.returncode != 0
    assert written(directory)["rationale"] == "the first reason"
    replaced = run(directory, "1\n1\nthe second reason\ny\n")
    assert replaced.returncode == 0, replaced.stderr
    assert written(directory)["rationale"] == "the second reason"
    assert written(directory)["statistical"]["name"] == B


def test_a_file_that_cannot_be_read_does_not_stop_a_new_choice(directory):
    os.makedirs(os.path.join(directory, "outputs"))
    with open(
        os.path.join(directory, "outputs", "headline_models.json"), "w"
    ) as f:
        f.write(
            '{"statistical": {"name": "Statistical: gone", "design": {}}, '
            '"mechanistic": {"name": "x"}, "rationale": "", '
            '"chosen_on": "", "tables": {}}'
        )
    done = run(directory, "1\n1\nbecause\ny\n")
    assert done.returncode == 0, done.stderr
    assert "cannot be read" in done.stdout
    assert written(directory)["statistical"]["name"] == B


def test_only_one_model_given_asks_for_the_other(directory):
    done = run(directory, "1\nthe reason\ny\n", "--statistical", A)
    assert done.returncode == 0, done.stderr
    record = written(directory)
    assert record["statistical"]["name"] == A
    assert record["mechanistic"]["name"] == M_LAST_TWO


def test_a_number_outside_the_table_or_a_multi_chain_name_is_asked_again(
    directory,
):
    # 9 is not a number of the table (3 statistical models); the multi-chain
    # model is no choice; then valid answers
    done = run(directory, f"9\n1\n{M_MULTI}\n2\nbecause\ny\n")
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("is not a number from 1 to") == 2
    assert written(directory)["statistical"]["name"] == B
    assert written(directory)["mechanistic"]["name"] == M


def test_the_help_text_runs():
    done = subprocess.run(
        [sys.executable, SCRIPT, "--help"], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert "--statistical" in done.stdout and "--rationale" in done.stdout


def test_a_run_with_the_models_given_says_what_it_replaces(directory):
    first = run(directory, "1\n1\nthe first reason\ny\n")
    assert first.returncode == 0, first.stderr
    second = run(
        directory,
        "",
        "--statistical",
        A,
        "--mechanistic",
        M,
        "--rationale",
        "the second reason",
    )
    assert second.returncode == 0, second.stderr
    assert "Replacing the earlier choice of" in second.stdout
    assert "the first reason" in second.stdout
    assert written(directory)["rationale"] == "the second reason"


def test_a_file_with_a_field_of_the_wrong_kind_does_not_stop_a_new_choice(
    directory,
):
    os.makedirs(os.path.join(directory, "outputs"))
    path = os.path.join(directory, "outputs", "headline_models.json")
    with open(path, "w") as f:
        f.write('{"statistical": "a string, not an object"}')
    done = run(directory, "1\n1\nbecause\ny\n")
    assert done.returncode == 0, done.stderr
    assert "cannot be read" in done.stdout
    assert written(directory)["statistical"]["name"] == B
