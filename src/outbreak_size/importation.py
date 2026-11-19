"""
The importation model: the yearly import-rate posteriors of the states and
the probabilities of imports in counties (see analysis/08_importation_model.py).
"""

import numpy as np
import polars as pl
from scipy import stats

from outbreak_size.regions import higher_level_regions
from outbreak_size.validation import projection_twin_mask, region_member_mask

# How far from 1 the population shares of a state's rows may add up (one row per
# place, see distribute_state_level_risk_to_county). A train row's population
# can differ a little from that of its projection twin; a place counted twice
# or a region with the wrong population is off by much more
SHARE_SUM_TOLERANCE = 0.05

# States with no 2025 import data (their 2025 outbreak is handled separately,
# so the 2025 import file has no rows for them). 2025 is a missing year for
# them, not a year with 0 imports, so it is not counted in their years of data
STATES_WITHOUT_2025_IMPORTS = ["South Carolina"]

# The imports are known through 2025 only. The importation posterior of the
# projection rows of validation year 2026 (all years of data, no pre-year
# posterior) serves 2026 and every later year, so the projections of 2027 and
# after join the importation rows of this validation year.
IMPORT_CONDITIONING_VALIDATION_YEAR = 2026


def scale_importation_probability(probability, scaler):
    """
    Probability of at least one importation after multiplying the importation
    rate by scaler. The probability is 1 - exp(-rate), so the scaled one is
    1 - exp(-scaler * rate) = 1 - (1 - probability) ** scaler.
    """
    return 1 - (1 - probability) ** scaler


def future_importation_draws(import_df, future_base):
    """
    The draws of the importation probability and of the expected imports of
    the counties of the projections (2027 and later), as numpy arrays of shape
    (n_draws, n_counties) in the order of `future_base`'s counties.

    `import_df` has the importation rows with their draws
    (`importation_probability_samples` and `expected_importations_samples`,
    one list of draws for each row) and `future_base` the counties of the
    projections (a `fips` column). The draws are those of the projection rows
    of IMPORT_CONDITIONING_VALIDATION_YEAR, one row per county. A county
    with no row, or a row twice, is an error.
    """
    latest = import_df.filter(
        (pl.col("validation_year") == IMPORT_CONDITIONING_VALIDATION_YEAR)
        & (pl.col("type") == "projection")
    )
    if latest["fips"].is_duplicated().any():
        raise ValueError(
            "two importation rows of the same county in validation year "
            f"{IMPORT_CONDITIONING_VALIDATION_YEAR}"
        )
    joined = future_base.select("fips").join(
        latest.select(
            [
                "fips",
                "importation_probability_samples",
                "expected_importations_samples",
            ]
        ),
        on="fips",
        how="left",
        maintain_order="left",
    )
    if future_base["fips"].is_duplicated().any():
        raise ValueError("a county is twice in the projections")
    for column in (
        "importation_probability_samples",
        "expected_importations_samples",
    ):
        if joined[column].null_count() > 0:
            raise ValueError(
                f"some counties of the projections have no {column}"
            )
    return tuple(
        np.vstack(joined[column].to_numpy()).T  # type: ignore
        for column in (
            "importation_probability_samples",
            "expected_importations_samples",
        )
    )


def probability_at_least_one_import(county_alpha, beta, share):
    """
    A county's probability of at least one import in a year.

    As in analysis/08_importation_model.py, the county's rate is share times
    its state's rate, and the state's rate has a Gamma(alpha, beta) posterior.
    The probability of no import is then E[exp(-share * rate)], which is
    (beta / (beta + share)) ** alpha. county_alpha is the state's alpha times
    the share, as stored in the import validation file, so the state's alpha
    is county_alpha / share.
    """
    return 1 - (beta / (beta + share)) ** (county_alpha / share)


def draw_imports(expected_imports, rng):
    """
    Number of imported cases of each county in each draw: Poisson with the
    county's expected imports. The probability of at least one import is then
    1 - exp(-expected imports), the importation probability the importation
    model reports.

    expected_imports: [n_draws, n_locations]; rng: a numpy Generator.
    """
    return rng.poisson(expected_imports).astype(np.int32)


def total_case_draws(outbreak_size_draws, import_draws):
    """
    Draws of all the cases of each county in a year: the imported cases plus,
    in a draw with at least one imported case, the outbreak that follows them.
    A draw with no import has no cases.

    outbreak_size_draws: the outbreak size if there is one; import_draws: the
    imported cases (see draw_imports); both [n_draws, n_locations].
    """
    return np.where(import_draws >= 1, import_draws + outbreak_size_draws, 0)


def expected_case_draws(
    outbreak_size_draws, import_probability, expected_imports
):
    """
    The expected cases of each county in each draw: the outbreak size times
    the probability of at least one import, plus the expected imports. It is
    the average of total_case_draws over the imports drawn from the expected
    imports of the draw (n imports are n cases, plus the outbreak when n >= 1:
    n has the mean `expected_imports` and is at least 1 with probability
    `import_probability`). The mean over the draws is the risk score and the
    forecast of a county, without the noise that drawing the imports adds.

    All three are [n_draws, n_locations] (the probability and the expected
    imports of the same draw).
    """
    return np.asarray(outbreak_size_draws) * np.asarray(
        import_probability
    ) + np.asarray(expected_imports)


def calculate_yearly_import_probability(
    df, years_of_data, prior_alpha=1, prior_beta=1
):
    """
    The Gamma posterior of each state's yearly import rate.

    df has one row per state with its imports over all the years (`imported`)
    and over the years before i (`imported_pre_i`, i = 2021 to 2025).
    years_of_data is the number of consecutive years of data, ending in 2025.
    Adds `posterior_alpha` and `posterior_beta` (all the years) and
    `posterior_alpha_pre_i` and `posterior_beta_pre_i` (the years before i),
    for a Gamma(prior_alpha, prior_beta) prior; `observed` is False for a state
    with no imports.
    """
    df = df.with_columns(observed=pl.lit(True))
    # we now observe all states because we have cdc importations data
    # but we still want to flag them as unobserved because we did not observe
    # an import in those states
    df = df.with_columns(
        observed=pl.when(pl.col("imported") == 0)
        .then(pl.lit(False))
        .otherwise(pl.col("observed"))
    )

    # the number of imports is a Poisson process, and we want to fit the rate
    # parameter
    # using an exponential(1) / gamma(1, 1) prior on the rate parameter, which
    # yields a uniform(0, 1) probability of there being at least one import in
    # a year, we update the gamma parameters

    # use a distribution with uniform prior to calculate import probability
    df = df.with_columns(
        *[
            (pl.col(f"imported_pre_{i}") + prior_alpha).alias(
                f"posterior_alpha_pre_{i}"
            )
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        # the posterior before year i only has the years before i: all the
        # years of data minus the (2026 - i) years from i through 2025
        *[
            (pl.lit(years_of_data - (2026 - i)) + prior_beta).alias(
                f"posterior_beta_pre_{i}"
            )
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        posterior_alpha=pl.col("imported") + prior_alpha,
        posterior_beta=pl.lit(years_of_data)
        - pl.col("state").is_in(STATES_WITHOUT_2025_IMPORTS).cast(pl.Int64)
        + prior_beta,
    )

    return df


def pre_year_for_validation_year(validation_year: int) -> int | None:
    """
    Which "imports before year i" posterior the rows of a validation year may
    use (the i), or None for the posterior that uses all the years of data.

    The rows of a validation year from 2021 to 2025 (the observed county-years
    of that year and the projection rows that predict it) must use what was
    known before that year: using all the years would put the imports of the
    year being predicted (and of later years) into its own prediction. The
    age of a county's vaccination data does not matter here, every county of
    the year uses the same posterior. The rows that predict 2026 may use every
    year of data.
    """
    if 2021 <= validation_year <= 2025:
        return validation_year
    return None


def region_unit_population(county_data, regions):
    """
    county_data with the population of each region unit set to the population
    the unit stands for in the importation shares: that of its member counties
    that have no row of their own in the same validation year.

    A region unit (a place of several counties, such as a health district: its
    `fips` is the name of a region in `regions`) is one row with the
    population of the whole region. A member with an observed row of its own
    stays its own row next to the unit (see validation.region_member_mask), so
    the unit's own population would count that member twice in its state's
    shares. The members' populations are those of the projection rows of the
    same validation year, which is what the shares are over: the rows that
    remain (the unit, the members with their own row and the other counties)
    then add up to the state. A unit with no member among the projection rows
    keeps its population.
    """
    pairs = pl.DataFrame(
        [
            (unit, member)
            for unit, members in regions.items()
            for member in members
        ],
        schema={"unit": pl.String, "member": pl.String},
        orient="row",
    )
    has_own_row = (
        county_data.filter(pl.col("type") == "train")
        .select(member="fips", validation_year="validation_year")
        .unique()
        .with_columns(has_own_row=pl.lit(True))
    )
    unit_pop = (
        county_data.filter(pl.col("type") == "projection")
        .select(
            member="fips", validation_year="validation_year", member_pop="pop"
        )
        .join(pairs, on="member")
        .join(has_own_row, on=["member", "validation_year"], how="left")
        .filter(pl.col("has_own_row").is_null())
        .group_by("unit", "validation_year")
        .agg(unit_pop=pl.col("member_pop").sum())
    )
    return (
        county_data.join(
            unit_pop,
            left_on=["fips", "validation_year"],
            right_on=["unit", "validation_year"],
            how="left",
            maintain_order="left",
        )
        .with_columns(pop=pl.coalesce("unit_pop", "pop"))
        .drop("unit_pop")
    )


def distribute_state_level_risk_to_county(
    df_state, county_data, counties_with_known_imports, seed, regions=None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # each state has a population-weighted value of the county's importation probability
    regions = higher_level_regions() if regions is None else regions
    # a region unit's share is that of the members it stands for, so that no
    # person is counted twice in the state's shares
    county_data = region_unit_population(county_data, regions)
    # join state level data to county data

    df = county_data.join(
        df_state, on="state", how="left", maintain_order="left"
    )

    # one generator for all the draws, so that a run is reproducible
    rng = np.random.default_rng(seed)

    # a state's rate is one unknown: its counties (and the years that use the
    # same posterior) share each draw of it, so that the uncertainty of the
    # state's rate does not average out when the counties are summed
    state_rate_draws = {}

    def draw_rate_samples(x):
        i = pre_year_for_validation_year(x["validation_year"])
        if i is None:
            alpha, beta = x["posterior_alpha"], x["posterior_beta"]
        else:
            alpha = x[f"posterior_alpha_pre_{i}"]
            beta = x[f"posterior_beta_pre_{i}"]
        key = (x["state"], alpha, beta)
        if key not in state_rate_draws:
            # 8000 draws, one for each posterior draw of the outbreak models:
            # the importation draws are combined with them draw by draw
            state_rate_draws[key] = stats.gamma.rvs(
                a=alpha, scale=1 / beta, size=8000, random_state=rng
            ).tolist()  # type: ignore
        return state_rate_draws[key]

    # get samples from the posterior
    df = df.with_columns(
        # calculate a bunch of samples for later use
        importation_probability_samples=pl.struct(
            [
                *[
                    "posterior_alpha",
                    "posterior_beta",
                    "validation_year",
                    "state",
                ],
                *[
                    f"posterior_alpha_pre_{i}"
                    for i in [2021, 2022, 2023, 2024, 2025]
                ],
                *[
                    f"posterior_beta_pre_{i}"
                    for i in [2021, 2022, 2023, 2024, 2025]
                ],
            ]
        ).map_elements(draw_rate_samples, pl.List(pl.Float64)),
    )
    # calculate the population weight: a county's share of its state's rate is
    # its population over the population of all the state's counties in the
    # same validation year (the projection rows of that year are one per
    # county), for train and projection rows alike. A share must not depend on
    # which counties happen to be in the same year and type: the train rows
    # only cover the counties that reported cases, so a share among them would
    # be larger just because the county was observed. A place without a county
    # FIPS (a region or district of the data) uses its own population against
    # the same state total; a state without projection rows in that year uses
    # the rows of its own year and type
    state_universe = (
        county_data.filter(pl.col("type") == "projection")
        .group_by("state", "validation_year")
        .agg(
            universe_pop=pl.col("pop").sum(),
            universe_pop_log=pl.col("pop").log10().sum(),
        )
    )
    df = df.join(
        state_universe,
        on=["state", "validation_year"],
        how="left",
        maintain_order="left",
    )
    df = df.with_columns(
        pop_weight=pl.col("pop")
        / pl.coalesce(
            "universe_pop",
            pl.col("pop").sum().over(["state", "validation_year", "type"]),
        ),
        pop_weight_log=pl.col("pop").log10()
        / pl.coalesce(
            "universe_pop_log",
            pl.col("pop")
            .log10()
            .sum()
            .over(["state", "validation_year", "type"]),
        ),
    )
    # a share is positive; it can exceed 1 slightly where a county's
    # population in its year is larger than in the 2026 rows (e.g. a state
    # with one county)
    no_share = df.select(
        "fips", "state", "year", "type", "pop_weight", "pop_weight_log"
    ).filter(
        ~(pl.col("pop_weight") > 0).fill_null(False)
        | ~(pl.col("pop_weight_log") > 0).fill_null(False)
    )
    if no_share.height > 0:
        raise ValueError(
            f"{no_share.height} rows have no population share of their "
            "state (missing or non-positive population): "
            f"{no_share.drop('pop_weight', 'pop_weight_log').head(5).rows()}"
        )
    # one row per place (the train rows, and the projection rows that are
    # neither the twin of a train row nor a member of a region unit, as the
    # validation keeps them) must carry the state's whole rate: their shares
    # add up to 1. A place counted twice or a region with the wrong population
    # would add up to more or less
    shares = df.select(
        "fips",
        "state",
        "validation_year",
        "type",
        "pop_weight",
        "universe_pop",
    )
    one_row_per_place = ~projection_twin_mask(shares)
    one_row_per_place[one_row_per_place] = ~region_member_mask(
        shares.filter(pl.Series(one_row_per_place)), regions
    )
    share_sums = (
        shares.filter(pl.Series(one_row_per_place))
        .filter(pl.col("universe_pop").is_not_null())
        .group_by("state", "validation_year")
        .agg(share_sum=pl.col("pop_weight").sum())
    )
    off = share_sums.filter(
        (pl.col("share_sum") - 1).abs() > SHARE_SUM_TOLERANCE
    ).sort("state", "validation_year")
    if off.height > 0:
        raise ValueError(
            f"the population shares of the rows of {off.height} state-years "
            "do not add up to 1 (a place counted twice, or a region whose "
            f"population is not its members'): {off.head(5).rows()}"
        )
    # distribute the state level importation probability to counties based on population weight
    df = df.with_columns(
        *[
            (pl.col(f"posterior_alpha_pre_{i}") * pl.col("pop_weight")).alias(
                f"posterior_alpha_pre_{i}"
            )
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        *[
            (
                pl.col(f"posterior_alpha_pre_{i}") * pl.col("pop_weight_log")
            ).alias(f"posterior_alpha_pre_{i}_log")
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        importation_probability_samples=pl.struct(
            ["pop_weight", "importation_probability_samples"]
        ).map_elements(
            lambda x: (
                x["pop_weight"]
                * np.array(x["importation_probability_samples"])
            ).tolist(),
            pl.List(pl.Float64),
        ),
        posterior_alpha=pl.col("posterior_alpha") * pl.col("pop_weight"),
        posterior_alpha_log=pl.col("posterior_alpha")
        * pl.col("pop_weight_log"),
    )

    # calculate summary statistics on the importation probabilities
    # first we need to convert the rate parameter to poisson probabilities of
    # at least one import

    # Save the rate samples (expected number of importations) before transforming
    df = df.with_columns(
        expected_importations_samples=pl.col("importation_probability_samples")
    )

    # Transform rate samples to probability of at least one import
    df = df.with_columns(
        importation_probability_samples=pl.col(
            "importation_probability_samples"
        ).map_elements(
            lambda x: (1 - np.exp(-np.array(x))).tolist(), pl.List(pl.Float64)
        )
    )

    # now we can calculate mean, median, 2.5th and 97.5th percentiles
    df = df.with_columns(
        importation_probability_mean=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.mean(np.array(x)), pl.Float64),
        importation_probability_median=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.median(np.array(x)), pl.Float64),
        importation_probability_lower=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.percentile(np.array(x), 2.5), pl.Float64),
        importation_probability_upper=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.percentile(np.array(x), 97.5), pl.Float64),
    )

    # take only the quantities we need
    df = df.select(
        [
            *[
                f"posterior_alpha_pre_{i}"
                for i in [2021, 2022, 2023, 2024, 2025]
            ],
            *[
                f"posterior_alpha_pre_{i}_log"
                for i in [2021, 2022, 2023, 2024, 2025]
            ],
            *[
                f"posterior_beta_pre_{i}"
                for i in [2021, 2022, 2023, 2024, 2025]
            ],
            *[
                "fips",
                "state",
                "county",
                "year",
                "validation_year",
                "type",
                "observed",
                "posterior_alpha",
                "posterior_beta",
                "posterior_alpha_log",
                "importation_probability_samples",
                "importation_probability_mean",
                "importation_probability_median",
                "importation_probability_lower",
                "importation_probability_upper",
                "expected_importations_samples",
            ],
        ]
    )

    # counties that we never observe must have no imports
    counties_with_known_imports_never_observed = df.filter(~pl.col("observed"))
    counties_with_known_imports_never_observed = (
        counties_with_known_imports_never_observed.with_columns(
            imported=pl.lit(0)
        )
    )
    # we know that other counties in state-years with known imports besides
    # the particular county-year with a known import also had 0 imports
    # we want to find those state-years
    counties_with_known_imports = counties_with_known_imports.with_columns(
        flag=pl.lit(True)
    )
    state_years_with_known_county_imports = df.join(
        counties_with_known_imports.select(["year", "state", "flag"]).unique(),
        on=["year", "state"],
        how="left",
        maintain_order="left",
    )
    # take states with the flag as true
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.filter(pl.col("flag") == True)
    )
    # drop the counties with known imports
    # make a dummy column of a county-state-year combination identifier
    counties_with_known_imports = counties_with_known_imports.with_columns(
        county_state_year=(
            pl.col("county") + pl.col("state") + pl.col("year").cast(pl.String)
        )
    )
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.with_columns(
            county_state_year=(
                pl.col("county")
                + pl.col("state")
                + pl.col("year").cast(pl.String)
            )
        )
    )
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.filter(
            ~pl.col("county_state_year").is_in(
                counties_with_known_imports.select("county_state_year")
                .to_series()
                .implode()
            )
        )
    )
    # assign 0 imports to these counties
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.with_columns(imported=pl.lit(0))
    )

    counties_with_known_imports = pl.concat(
        [
            counties_with_known_imports,
            counties_with_known_imports_never_observed,
            state_years_with_known_county_imports,
        ],
        how="diagonal_relaxed",
    ).select(["year", "state", "county", "imported"])

    return df, counties_with_known_imports
