"""
The headline models: the statistical model and the mechanistic model that the
user chooses (analysis/06b_choose_headline_models.py) after looking at the
tables of the model scoring, and that every script reads from
outputs/headline_models.json (06 --headline-2026, 07, the headline table of
2026, the discordance table and the figures). Nothing here chooses a model:
the choice, its reason and the date are written down, with a fingerprint of the
tables it was made from, so that a later run of the table script (which can
change them) does not silently change what the scripts fit and score.

The mechanistic headline is one of the single-chain models: the models with
outbreaks of several chains per import are sensitivity analyses of the table
script, and only the single-chain model is scored on 2026.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import polars as pl

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.model_designs import (
    DESIGNS_BY_NAME,
    START_YEARS,
    ModelDesign,
    start_year_of_model_name,
    trained_since,
)
from outbreak_size.model_selection import (
    SUMMARY_LABEL,
    WEIGHTED_GAP_COLUMN,
    YEARS,
)

HEADLINE_FILE_NAME = "headline_models.json"
# the tables the choice is made from (written by the table script into figures/)
TABLE_FILES = {
    "S1": "supplemental_data_s1_county_level_prediction_scoring.csv",
    "S2": "supplemental_data_s2_model_consistency_across_years.csv",
}
CHOOSE_SCRIPT = "analysis/06b_choose_headline_models.py"
# the scores of S1 (the means over the years of the choice) shown next to
# the gaps of S2 when the models are listed
SHOWN_S1_COLUMNS = (
    "AUROC, 1+ (%)",
    "AUROC, 5+ (%)",
    "AUROC, 10+ (%)",
    "AUPRC, 5+ (%)",
)


def mechanistic_model_names():
    """The names of the mechanistic models that can be the headline: the
    single-chain model of each training window."""
    return [
        "Mechanistic: MMR coverage" + trained_since(start_year)
        for start_year in START_YEARS
    ]


@dataclass(frozen=True)
class HeadlineModels:
    statistical: ModelDesign
    mechanistic: str  # its name in the tables of the scoring
    rationale: str
    chosen_on: str
    tables: dict  # {"S1": {"file": name, "sha256": hash}, "S2": {...}}

    @property
    def mechanistic_start_year(self):
        return start_year_of_model_name(self.mechanistic)


def file_fingerprint(path):
    """The hash of the contents of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_names(statistical, mechanistic):
    """Raises a ValueError if a name is not a model that can be the headline."""
    if statistical not in DESIGNS_BY_NAME:
        raise ValueError(
            f"{statistical!r} is not one of the {len(DESIGNS_BY_NAME)} "
            "statistical models of the table script"
        )
    if mechanistic not in mechanistic_model_names():
        raise ValueError(
            f"{mechanistic!r} cannot be the headline mechanistic model: the "
            f"choices are {mechanistic_model_names()} (the models with "
            "outbreaks of several chains are sensitivity analyses)"
        )


def write_headline_models(
    path, statistical, mechanistic, rationale, table_paths, chosen_on
):
    """Writes the choice. table_paths: {"S1": path, "S2": path} of the tables
    it was made from; chosen_on: the date (text)."""
    check_names(statistical, mechanistic)
    record = dict(
        statistical=dict(
            name=statistical, design=asdict(DESIGNS_BY_NAME[statistical])
        ),
        mechanistic=dict(name=mechanistic),
        rationale=rationale,
        chosen_on=chosen_on,
        tables={
            label: dict(
                file=os.path.basename(table_path),
                sha256=file_fingerprint(table_path),
            )
            for label, table_path in table_paths.items()
        },
    )
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def read_headline_models(path, fig_dir=None):
    """The choice written by write_headline_models. Stops, saying which script
    to run, if there is none. With `fig_dir` (where the table script writes S1
    and S2), prints a warning if either table is not the one the choice was
    made from (the choice stays as it is; the tables may have been written
    again by a new run of the table script)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing: run {CHOOSE_SCRIPT} to choose the headline "
            "models from the tables of the model scoring"
        )
    with open(path) as f:
        record = json.load(f)
    statistical = record["statistical"]["name"]
    mechanistic = record["mechanistic"]["name"]
    check_names(statistical, mechanistic)
    design = DESIGNS_BY_NAME[statistical]
    assert asdict(design) == record["statistical"]["design"], (
        f"{statistical}: the recorded design is not the design of that "
        "name any more"
    )
    headline = HeadlineModels(
        statistical=design,
        mechanistic=mechanistic,
        rationale=record["rationale"],
        chosen_on=record["chosen_on"],
        tables=record["tables"],
    )
    if fig_dir is not None:
        present = {
            label: os.path.join(fig_dir, name)
            for label, name in TABLE_FILES.items()
            if os.path.exists(os.path.join(fig_dir, name))
        }
        changed = changed_tables(headline, present)
        if changed:
            plural = len(changed) > 1
            print(
                f"WARNING: {' and '.join(changed)} "
                f"{'are not the tables' if plural else 'is not the table'} the "
                f"headline models were chosen from (on {headline.chosen_on}): "
                f"look at the new {'tables' if plural else 'table'} with "
                f"{CHOOSE_SCRIPT} and choose again if the choice should "
                "change (this run uses the choice as it is)"
            )
    return headline


def changed_tables(headline, table_paths):
    """The labels of the tables (of table_paths: {label: path}) that are not
    the ones the choice was made from: a table that has been written again, or
    one the choice does not mention."""
    changed = []
    for label, table_path in table_paths.items():
        recorded = headline.tables.get(label)
        if recorded is None or recorded["sha256"] != file_fingerprint(
            table_path
        ):
            changed.append(label)
    return changed


def ask_choice(
    question, options, ask=input, show=print, listed=None, show_list=True
):
    """
    Asks for one of `options` (names): by its number in the list (1 to
    len(listed)) or by typing the name. `listed` are the options numbered, the
    first of them by default all, and `options` all that are accepted (by
    name). The list is printed with `show` unless `show_list` is False (when a
    table that the user sees has numbered the same models). `ask` reads a line
    (input); the question is asked again until the answer is one of the
    options. Raises a SystemExit, saying so, if there is no more input.
    """
    listed = list(options) if listed is None else list(listed)
    if show_list:
        for number, option in enumerate(listed, start=1):
            show(f"  {number:2d}. {option}")
    while True:
        try:
            answer = ask(f"{question} ").strip()
        except EOFError:
            raise SystemExit(
                f"no answer to: {question} (run the script in a terminal, "
                "or give the models with --statistical and --mechanistic)"
            )
        if answer in options:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(listed):
            return listed[int(answer) - 1]
        show(
            f"  {answer!r} is not a number from 1 to {len(listed)} or the "
            "name of a model that can be chosen"
        )


def outcome_counts(counts):
    """Places scored and places with 1+, 5+ and 10+ cases in each year, from
    the table of outcome counts (the cases including imports)."""
    return counts.filter(pl.col("Condition") == CONDITIONS[0]).select(
        "Evaluation year",
        "Places scored",
        "Places with 1+ cases",
        "Places with 5+ cases",
        "Places with 10+ cases",
    )


def candidate_table(s1, s2, pool, limit=None):
    """
    The models of `pool` ("Statistical" or "Mechanistic") with their
    event-weighted mean gap from the best of the pool over the years of the
    choice and their AUROC for
    5+ cases in every year (S2), and the means of S1 over the same years, the
    smallest gap first (the first `limit` models if given).
    """
    means = s1.filter(
        (pl.col("Evaluation year") == SUMMARY_LABEL)
        & (pl.col("Condition") == CONDITIONS[0])
    ).select("Model", *SHOWN_S1_COLUMNS)
    yearly = [f"AUROC, 5+ (%), {year}" for year in YEARS]
    table = (
        s2.filter(pl.col("Pool") == pool)
        .select("Model", WEIGHTED_GAP_COLUMN, *yearly)
        .join(means, on="Model", how="left", maintain_order="left")
        .sort(WEIGHTED_GAP_COLUMN, maintain_order=True)
    )
    return table if limit is None else table.head(limit)


def short_name(model):
    """The name of a model of the tables without what the table around it
    already says (the pool) and with its training window abbreviated."""
    for old, new in (
        ("Statistical: ", ""),
        ("Mechanistic with multi-chain outbreaks: ", "multi-chain: "),
        ("Mechanistic: ", ""),
        (", trained on last two years", ", last two years"),
        (", trained since ", ", since "),
    ):
        model = model.replace(old, new)
    return model


def _number(value, decimals):
    return "-" if value is None else f"{value:.{decimals}f}"


def format_outcome_counts(counts):
    """The lines of a table of the places with cases in each year (outcome_
    counts)."""
    lines = [
        "year   places   with 1+ cases   5+   10+",
    ]
    for row in counts.iter_rows(named=True):
        lines.append(
            f"{row['Evaluation year']!s:<5}{row['Places scored']:>8}"
            f"{row['Places with 1+ cases']:>14}"
            f"{row['Places with 5+ cases']:>7}"
            f"{row['Places with 10+ cases']:>6}"
        )
    return lines


def format_candidates(table, numbered=None):
    """
    The lines of a text table of candidate_table: the mean gap from the best of
    the pool, the AUROC for 5+ cases of each year (S2) and the means of S1 over
    the years of the choice. The models of `numbered` (all by default) have
    their number in the first column, in the order of the table, and are the
    ones the user can pick by number; the others have none.
    """
    models = table["Model"].to_list()
    numbered = models if numbered is None else list(numbered)
    names = [short_name(m) for m in models]
    width = max(len(name) for name in names)
    yearly = [f"AUROC, 5+ (%), {y}" for y in YEARS]
    s1_labels = ("AUROC 1+", "AUROC 5+", "AUROC 10+", "AUPRC 5+")
    year_width = 5 * len(YEARS) + len(YEARS) - 1
    s1_width = 10 * len(s1_labels) - 1
    header = (
        f"{'#':>2}  {'model':<{width}}  {'wgap':>5}  "
        + " ".join(f"{y:>5}" for y in YEARS)
        + "  "
        + " ".join(f"{label:>9}" for label in s1_labels)
    )
    caption = (
        f"{'':>2}  {'':<{width}}  {'S2':>5}  "
        f"{'S2: AUROC 5+ of each year':^{year_width}}  "
        f"{'S1 means ' + SUMMARY_LABEL:^{s1_width}}"
    )
    lines = [caption, header]
    for row, name in zip(table.iter_rows(named=True), names):
        number = (
            str(numbered.index(row["Model"]) + 1)
            if row["Model"] in numbered
            else ""
        )
        lines.append(
            f"{number:>2}  {name:<{width}}  "
            f"{_number(row[WEIGHTED_GAP_COLUMN], 2):>5}  "
            + " ".join(f"{_number(row[c], 1):>5}" for c in yearly)
            + "  "
            + " ".join(
                f"{_number(row[c], 1):>9}"
                for c in (
                    "AUROC, 1+ (%)",
                    "AUROC, 5+ (%)",
                    "AUROC, 10+ (%)",
                    "AUPRC, 5+ (%)",
                )
            )
        )
    return lines
