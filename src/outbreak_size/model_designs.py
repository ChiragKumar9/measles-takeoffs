"""
The designs of the statistical outbreak models scored by the table script
(analysis/table_model_performance_validation.py), their names, and the
projection model of scripts 07 and 11.

A design is a set of covariates, spatial term or not, state-specific
vaccination coefficients (pooling) or not, and a training window. The table
script fits and scores all 36 designs; the user chooses one of them as the
headline model (analysis/06b_choose_headline_models.py, outbreak_size.
headline_models; select_best_statistical_model gives the one that S2 ranks
first, as a recommendation to look at), script 07 fits it on the data up to
2026 and projects with it, and script 11 uses the design 07 recorded
(write_record, read_record), never the choice file, so that it always matches
the posterior samples it predicts from.
"""

import json
from dataclasses import asdict, dataclass

import polars as pl

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.model_inputs import (
    IV_COLUMNS,
    REGRESSION_COLUMNS,
    no_regression_covariates,
    pooled_model,
)
from outbreak_size.model_selection import (
    LARGEST_GAP_COLUMN,
    MEAN_AUROC_COLUMN,
    SUMMARY_LABEL,
    WEIGHTED_GAP_COLUMN,
)

# the training windows: the years since this one, or (negative) that many
# years before the year to predict
START_YEARS = [2011, 2019, -2]
# the sets of covariates of the statistical models
COVARIATE_SETS = [
    "MMR coverage",
    "MMR coverage + exemptions",
    "All covariates",
]
# the pooled ("iv") columns and the ordinary regression columns of each set
SET_COLUMNS = {
    "MMR coverage": (IV_COLUMNS[:1], []),
    "MMR coverage + exemptions": (IV_COLUMNS, []),
    "All covariates": (IV_COLUMNS, REGRESSION_COLUMNS),
}
# S2 ranks the models by their event-weighted mean gap from the best of the
# pool; models tied on it are ordered by these scores of the cases including imports, averaged
# over 2022-2025 (S1), the first one first, higher first
TIE_BREAKS = ("AUPRC, 5+ (%)", "AUROC, 10+ (%)", "AUPRC, 10+ (%)")
GAP_COLUMN = WEIGHTED_GAP_COLUMN
CANDIDATES_SHOWN = 3


def training_years(start_year, validation_year):
    """The years a model trained on `start_year` is fitted to before it
    predicts `validation_year`."""
    if start_year < 0:
        return list(range(validation_year + start_year, validation_year))
    return list(range(start_year, validation_year))


def trained_since(start_year):
    """The part of a model name that says how long it was trained."""
    if start_year == -2:
        return ", trained on last two years"
    assert start_year in (2011, 2019), "Invalid training start year"
    return f", trained since {start_year}"


def start_year_of_model_name(name):
    """The start year (one of START_YEARS) that the name of a model says it was
    trained from (the inverse of trained_since)."""
    found = [s for s in START_YEARS if name.endswith(trained_since(s))]
    if len(found) != 1:
        raise ValueError(
            f"the name of the model has no training window: {name}"
        )
    return found[0]


def statistical_model_name(model_variant, car, pool, start_year):
    """The name of a statistical model in the scoring tables."""
    name = f"Statistical: {model_variant}"
    if car:
        name += " + spatial"
    if pool:
        name += " + state-specific vaccine coefficients"
    return name + trained_since(start_year)


@dataclass(frozen=True)
class ModelDesign:
    covariates: str  # one of COVARIATE_SETS
    spatial: bool
    pooled: bool  # state-specific vaccination coefficients
    start_year: int  # one of START_YEARS

    @property
    def name(self):
        return statistical_model_name(
            self.covariates, self.spatial, self.pooled, self.start_year
        )

    @property
    def has_exemptions(self):
        return self.covariates != "MMR coverage"


DESIGNS_BY_NAME = {
    design.name: design
    for design in (
        ModelDesign(covariates, spatial, pooled, start_year)
        for spatial in (False, True)
        for pooled in (False, True)
        for covariates in COVARIATE_SETS
        for start_year in START_YEARS
    )
}


def design_model(design, rows, scaled, adjacency_matrix, **model_arguments):
    """
    The outbreak model of `design` on `rows`: `scaled` has the standardized
    covariates of every row (at least the columns of IV_COLUMNS and
    REGRESSION_COLUMNS, each standardized on its own, so the columns the design
    does not use are ignored). `model_arguments` are those of
    model_inputs.pooled_model (priors, likelihood, non-centred CAR, ...).
    All covariates has the IV_COLUMNS as pooled covariates and the
    REGRESSION_COLUMNS as ordinary ones; a set with fewer covariates has no
    ordinary regression covariates (a placeholder column), as in the rolling
    validation.
    """
    iv_columns, regression_columns = SET_COLUMNS[design.covariates]
    if regression_columns:
        regression = scaled.select(regression_columns).to_jax()
    else:
        regression = no_regression_covariates(rows.height)
    return pooled_model(
        rows,
        scaled.select(iv_columns),
        regression,
        pool=design.pooled,
        spatial=design.spatial,
        adjacency_matrix=adjacency_matrix,
        **model_arguments,
    )


def select_best_statistical_model(s3, s1):
    """
    The design that the S2 table (model_consistency_across_years) ranks first
    among the statistical models: the smallest event-weighted mean gap from
    the best of the pool in the AUROC for 5+ cases, ties (the same rounded gap) broken by
    TIE_BREAKS on the S1 means. Returns the design and the first
    CANDIDATES_SHOWN models of the ranking with their gaps (the models near the top are
    often within a tenth of a point).
    """
    cases = CONDITIONS[0]
    means = s1.filter(
        (pl.col("Evaluation year") == SUMMARY_LABEL)
        & (pl.col("Condition") == cases)
    ).select("Model", *TIE_BREAKS)
    ranking = (
        s3.filter(pl.col("Pool") == "Statistical")
        .select(
            "Model",
            MEAN_AUROC_COLUMN,
            GAP_COLUMN,
            LARGEST_GAP_COLUMN,
        )
        .join(means, on="Model", how="left", maintain_order="left")
        .sort(
            GAP_COLUMN,
            *TIE_BREAKS,
            descending=[False] + [True] * len(TIE_BREAKS),
            nulls_last=True,
            maintain_order=True,
        )
    )
    best = ranking["Model"][0]
    return DESIGNS_BY_NAME[best], ranking.head(CANDIDATES_SHOWN)


def write_record(path, design, years, choice):
    """Record the design a fit used (and `choice`, the reason and the date of
    the choice of the headline model) so that the scripts that predict from
    its draws build the same model."""
    record = dict(
        name=design.name,
        design=asdict(design),
        training_years=list(years),
        choice=choice,
    )
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def read_record(path):
    """The design and the training years recorded by write_record."""
    with open(path) as f:
        record = json.load(f)
    design = ModelDesign(**record["design"])
    assert design.name == record["name"], (
        "the recorded design has changed name"
    )
    return design, record["training_years"]


def record_file_name(globals_prior_sd):
    """The name of the record of the design that script 07 fitted with this
    globals prior sd (in the outputs folder, next to its posterior samples)."""
    return f"risk_projection_model_globals_prior_sd={globals_prior_sd}.json"
