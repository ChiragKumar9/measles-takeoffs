"""
Chooses the headline models: the statistical model that scripts 06
(--headline-2026) and 07 fit and the figures show, and the mechanistic model
that is scored beside it. The choice is made by the user, from the tables of
table_model_performance_validation.py, and written to
outputs/headline_models.json with its reason, its date and a fingerprint of
S1 and S2, which every script that needs the models reads (no script chooses a
model by itself).

It prints the number of places with cases in each year (the scores of a year
with few places are not stable), the statistical models with the smallest mean
gap from the best of the pool over 2022-2025 (S2) and their scores (S1), and
the mechanistic models, then asks for the two models and one line on why. Only
a single-chain mechanistic model can be the headline (the models with the
outbreaks of several chains are sensitivity analyses).

    uv run analysis/06b_choose_headline_models.py

The models can also be given, for a run without questions (the name of a model
as in S1 and S2):

    uv run analysis/06b_choose_headline_models.py --statistical "..." \\
        --mechanistic "..." --rationale "..."
"""

import argparse
import datetime
import os

import polars as pl

from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    ask_choice,
    candidate_table,
    check_names,
    format_candidates,
    format_outcome_counts,
    mechanistic_model_names,
    outcome_counts,
    read_headline_models,
    write_headline_models,
)
from outbreak_size.model_designs import select_best_statistical_model
from outbreak_size.model_selection import SUMMARY_LABEL

fig_dir = "figures"
output_dir = "outputs"
S1_FILE = os.path.join(
    fig_dir, "supplemental_data_s1_county_level_prediction_scoring.csv"
)
S2_FILE = os.path.join(
    fig_dir, "supplemental_data_s2_model_consistency_across_years.csv"
)
COUNTS_FILE = os.path.join(fig_dir, "table_supplemental_outcome_counts.csv")
LISTED = 10  # the statistical models listed with a number
HEADLINE_FILE = os.path.join(output_dir, HEADLINE_FILE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose the headline models from the tables of the scoring"
    )
    parser.add_argument(
        "--statistical", help="the name of the statistical model"
    )
    parser.add_argument(
        "--mechanistic", help="the name of the mechanistic model"
    )
    parser.add_argument("--rationale", help="one line on why they were chosen")
    args = parser.parse_args()

    for table_file in (S1_FILE, S2_FILE, COUNTS_FILE):
        if not os.path.exists(table_file):
            raise FileNotFoundError(
                f"{table_file} is missing: run "
                "table_model_performance_validation.py first"
            )
    s1 = pl.read_csv(S1_FILE)
    s2 = pl.read_csv(S2_FILE)
    counts = pl.read_csv(COUNTS_FILE)

    # a choice made before is replaced only if the user says so: show it first
    existing = os.path.exists(HEADLINE_FILE)
    before = None
    if existing:
        try:
            before = read_headline_models(HEADLINE_FILE)
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            # a file that cannot be read must not stop a new choice
            print(f"{HEADLINE_FILE} cannot be read ({error!r}).\n")
        else:
            print(
                f"The headline models now chosen (on {before.chosen_on}):\n"
                f"  statistical: {before.statistical.name}\n"
                f"  mechanistic: {before.mechanistic}\n"
                f"  why: {before.rationale}\n"
            )

    print("Places with cases, by year (the cases including imports):")
    print("\n".join(format_outcome_counts(outcome_counts(counts))))
    statistical = candidate_table(s1, s2, "Statistical")
    print(
        f"\nStatistical models, the smallest mean gap from the best of the "
        f"pool over {SUMMARY_LABEL} first (the first {LISTED} of "
        f"{statistical.height}; the gap is in AUROC points for 5+ cases):"
    )
    listed_statistical = statistical["Model"].to_list()[:LISTED]
    print(
        "\n".join(
            format_candidates(statistical.head(LISTED), listed_statistical)
        )
    )
    mechanistic = candidate_table(s1, s2, "Mechanistic")
    # the single-chain models, in the order of the table: the ones to choose
    # from (the others are sensitivity analyses)
    selectable = [
        model
        for model in mechanistic["Model"].to_list()
        if model in mechanistic_model_names()
    ]
    print(
        "\nMechanistic models (the numbered ones can be the headline; the "
        "multi-chain models are sensitivity analyses):"
    )
    print("\n".join(format_candidates(mechanistic, selectable)))
    recommended = select_best_statistical_model(s2, s1)[0].name
    print(
        f"\nThe statistical model the table ranks first: {recommended}\n"
        "(a recommendation to look at, not a choice: the scores of the "
        "years with few places with 5+ cases move a model up or down)\n"
    )

    asked = args.statistical is None or args.mechanistic is None
    if args.statistical is not None:
        chosen_statistical = args.statistical
    else:
        print("Which statistical model is the headline?")
        chosen_statistical = ask_choice(
            "Number (of the table) or name:",
            statistical["Model"].to_list(),
            listed=listed_statistical,
            show_list=False,
        )
    if args.mechanistic is not None:
        chosen_mechanistic = args.mechanistic
    else:
        print("\nWhich mechanistic model is the headline?")
        chosen_mechanistic = ask_choice(
            "Number (of the table) or name:", selectable, show_list=False
        )
    check_names(chosen_statistical, chosen_mechanistic)
    if args.rationale is not None:
        rationale = args.rationale
    else:
        try:
            rationale = input(
                "\nOne line on why (saved with the choice): "
            ).strip()
        except EOFError:
            raise SystemExit("no reason given: run the script in a terminal")
    if not rationale:
        raise SystemExit("a reason is needed (--rationale, or type one)")

    print(
        f"\nHeadline statistical model: {chosen_statistical}\n"
        f"Headline mechanistic model: {chosen_mechanistic}\n"
        f"Why: {rationale}"
    )
    if asked:
        try:
            verb = "Replace" if existing else "Write"
            answer = input(f"{verb} {HEADLINE_FILE}? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            raise SystemExit("nothing was written")
    if existing and not asked:
        # given on the command line: no question is asked, so say what goes
        print(
            "Replacing the earlier choice"
            + (
                f" of {before.chosen_on} ({before.rationale})"
                if before is not None
                else " (it could not be read)"
            )
        )
    os.makedirs(output_dir, exist_ok=True)
    write_headline_models(
        HEADLINE_FILE,
        chosen_statistical,
        chosen_mechanistic,
        rationale,
        {"S1": S1_FILE, "S2": S2_FILE},
        datetime.date.today().isoformat(),
    )
    print(f"Wrote {HEADLINE_FILE}")
