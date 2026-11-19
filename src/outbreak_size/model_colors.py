"""
The colours of the figures that stand for the same thing in every figure: the
two models (statistical blue, mechanistic orange, the pair of
figure_r01_validation_comparison.py; grey for the ranking by vaccine coverage
alone) and the two periods of the data (2011-2020 green, 2021-2025 purple). Under
simulated colour-vision deficiency the pair of models stays far apart (CIELAB
distance 116 for deuteranopia and 97 for protanopia, against 24 and 10 for the
green and red the figures used before).
"""

STATISTICAL_COLOR = "#2a78d6"
MECHANISTIC_COLOR = "#eb6834"
COVERAGE_COLOR = "gray"
EARLIER_PERIOD_COLOR = "green"  # 2011-2020
LATER_PERIOD_COLOR = (
    "#6a3d9a"  # 2021-2025: not the blue of the statistical model
)
# the covariate sets of the statistical models compared in the figures of the
# effect of coverage: the headline set is the statistical colour, the others
# are slate and gold, which differ from the green and purple of the two
# periods that these figures show next to them (in normal vision and in
# simulated deuteranopia and protanopia the closest pair of the five colours
# is the purple and the statistical blue, 34, 20 and 19 CIELAB units apart)
COVARIATE_SET_COLORS = {
    "MMR coverage": "#4a5b6d",
    "MMR coverage + exemptions": "#c9a000",
    "All covariates": STATISTICAL_COLOR,
}
# the sensitivity analysis of the headline set with a Poisson likelihood
POISSON_COLOR = "#bcbd22"
# the other rankings of the case-concentration curves (the vaccine coverage
# alone is COVERAGE_COLOR): the population size and the imports alone
POPULATION_COLOR = "#7b4ea3"
IMPORTS_COLOR = "#d6529b"
