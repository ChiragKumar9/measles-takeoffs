import numpy as np
import polars as pl

from outbreak_size.model_inputs import COVARIATE_SCALES
from outbreak_size.poisson import LIKELIHOODS


def variant_file_suffix(likelihood: str, covariate_scale: str) -> str:
    """
    The part of a rolling-validation file name that says which variant made
    it. The variant of every fit made before there were variants (a negative
    binomial likelihood with the natural covariate scale) has no suffix, so
    those files keep their names; any other variant adds what differs, for
    example "_likelihood=poisson" or "_covariate_scale=raw".
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError(
            f"likelihood must be one of {LIKELIHOODS}, not {likelihood!r}"
        )
    if covariate_scale not in COVARIATE_SCALES:
        raise ValueError(
            f"covariate_scale must be one of {COVARIATE_SCALES}, "
            f"not {covariate_scale!r}"
        )
    suffix = ""
    if likelihood != "negative_binomial":
        suffix += f"_likelihood={likelihood}"
    if covariate_scale != "natural":
        suffix += f"_covariate_scale={covariate_scale}"
    return suffix


# the file system allows 255 bytes in a name and save_samples writes the name plus
# ".part" first: a name of more bytes than this cannot be saved
MAX_FILE_NAME_BYTES = 250


def validation_file_name(
    years, pool, car, model_variant: str, globals_prior_sd
) -> str:
    """
    The name of the file of the samples of one rolling-validation fit: the
    training years, the pooling, the CAR switch, the model with its variant
    suffix (model name + variant_file_suffix) and the prior. A name too long
    for the file system writes the years as a range, [2011-2024] instead of
    [2011, 2012, ..., 2024]; every other name is the full one, so the files
    already saved keep theirs.
    """

    def name(years_text: str) -> str:
        return (
            f"samples_partial_pooled_poisson_validation_testing_{years_text}"
            f"_pool={pool}_CAR={car}_{model_variant}_globals_prior_sd={globals_prior_sd}.npz"
        )

    full = name(str(list(years)))
    if len(full.encode()) <= MAX_FILE_NAME_BYTES:
        return full
    years = list(years)
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(
            f"cannot shorten the name for years that are not consecutive: {years}"
        )
    return name(f"[{years[0]}-{years[-1]}]")


def require_no_nulls(df: pl.DataFrame, columns: list[str], name: str) -> None:
    """
    Raise if any of `columns` of `df` has a null (or NaN) value.

    A null model input becomes NaN, the Poisson mean of the county becomes
    NaN, and the draw of a Poisson with a NaN mean is -1 cases, with no
    error: the county silently ends up with the lowest risk of all.
    """
    counts = {}
    missing = pl.lit(False)
    for column in columns:
        is_missing = pl.col(column).is_null()
        if df.schema[column] in (pl.Float32, pl.Float64):
            is_missing = is_missing | pl.col(column).is_nan()
        n = df.filter(is_missing).height
        if n:
            counts[column] = n
            missing = missing | is_missing
    if counts:
        places = [
            c for c in ("state", "county", "fips", "year") if c in df.columns
        ]
        examples = df.filter(missing).select(places).head(5).rows()
        raise ValueError(
            f"{name}: model inputs with missing values (rows per column: "
            f"{counts}); first rows ({', '.join(places)}): {examples}"
        )


def keep_rows(draws, keep, name: str):
    """
    The columns of `draws` (n_draws, n_rows) where the boolean array `keep`
    is True.

    Posterior draws and the rows they belong to are matched by position only,
    so a file whose number of columns is not the number of rows (a stale file
    from another run) would be shifted onto the wrong counties without any
    error. This raises instead.
    """
    return require_width(draws, len(keep), name)[:, keep]


def align_samples_to_counties(draws, row_fips, county_fips):
    """
    The draws (n_draws, n_rows) of the rows with the fips `row_fips`, put in
    the order of the counties `county_fips`: for each county the column of the
    row that has its fips, and a column of zeros for a county without a row.
    Also returns the boolean array that says which counties have a row.

    The rows must be one per fips (twins of observed rows and the member
    counties of regions already dropped); a fips that is twice raises.
    """
    column = {}
    for i, fips in enumerate(row_fips):
        if fips in column:
            raise ValueError(f"{fips} is the fips of two rows")
        column[fips] = i
    draws = np.asarray(draws)
    aligned = np.zeros((draws.shape[0], len(county_fips)), dtype=draws.dtype)
    has_row = np.zeros(len(county_fips), dtype=bool)
    for j, fips in enumerate(county_fips):
        if fips in column:
            aligned[:, j] = draws[:, column[fips]]
            has_row[j] = True
    return aligned, has_row


def require_unique_places(df: pl.DataFrame, name: str) -> None:
    """
    Raise if a county (the 5-character fips) appears twice in the same
    validation year of `df` (columns fips and validation_year). Places
    without a county fips (regions, districts) are keyed by name and not
    checked here.
    """
    counties = df.filter(pl.col("fips").str.len_chars() == 5)
    repeated = counties.filter(
        counties.select("fips", "validation_year").is_duplicated()
    )
    if repeated.height:
        raise ValueError(
            f"{name}: {repeated.height} rows repeat a county in a validation "
            f"year, e.g. "
            f"{repeated.select('fips', 'validation_year').head(5).rows()}"
        )


def require_width(draws, n_rows: int, name: str):
    """
    Return `draws` (n_draws, n_rows) after checking it has one column per row.

    Draws and counties are matched by position, so draws from a different
    run (another number of counties) must stop the script, not be shifted
    onto the wrong counties.
    """
    if draws.shape[1] != n_rows:
        raise ValueError(
            f"{name}: {draws.shape[1]} columns of draws for {n_rows} rows; "
            "the draws and the rows come from different runs"
        )
    return draws


def require_same_fips(fips, expected_fips, label: str) -> None:
    """
    Raise unless `fips` is the list of counties `expected_fips`, in the same
    order.

    Predictions are matched to counties by position, so each file's county
    list must be the one the column indices were built from.
    """
    if list(fips) != list(expected_fips):
        raise ValueError(
            f"{label}: the counties of this file are not in the order of the "
            "first year's file"
        )


def projection_twin_mask(rows: pl.DataFrame) -> np.ndarray:
    """
    True for the projection rows that have a train row for the same county in
    the same validation year.

    The projection datasets hold each observed county-year twice: as a train
    row (matched to the observed outbreak) and as a projection row (like every
    other county). Dropping the marked rows keeps one row per county-year, the
    train row. rows needs the columns fips, validation_year and type
    ("train" or "projection"); places without a county fips are keyed by the
    name in `fips`.
    """
    train = (
        rows.filter(pl.col("type") == "train")
        .select("fips", "validation_year")
        .unique()
        .with_columns(has_train=pl.lit(True))
    )
    return (
        rows.join(
            train,
            on=["fips", "validation_year"],
            how="left",
            maintain_order="left",
        )
        .select(
            (
                (pl.col("type") == "projection")
                & pl.col("has_train").fill_null(False)
            ).alias("is_twin")
        )["is_twin"]
        .to_numpy()
    )


def region_member_groups(
    rows: pl.DataFrame, regions: dict[str, list[str]]
) -> list[tuple[int, list[int]]]:
    """
    The region units among `rows` and the rows of their member counties.

    A region unit (a multi-county place such as a health district) is a row
    whose `fips` is the name of a region in `regions` (region name -> member
    county FIPS), or a statewide place (a row whose `fips` is the 2-character
    FIPS of a state, as Delaware's) with every county of that state among
    the rows as its members. Its members are the rows of the same validation
    year that have a member's FIPS and no observed cases of their own (type
    "projection"): a member with its own observed row stays its own row.
    rows needs the columns fips, validation_year and type.

    Returns, for each region unit that has members among the rows, its row
    index and the row indices of its members. Raises if a row is a member of
    two regions, or a region unit is itself a member.
    """
    fips = rows["fips"].to_list()
    years = rows["validation_year"].to_list()
    types = rows["type"].to_list()
    index = {}
    for i, key in enumerate(zip(fips, years)):
        index.setdefault(key, i)

    # a statewide place (Delaware reports its cases and vaccination at the
    # state only, so the case data's row has the 2-character state FIPS) is a
    # region unit of every county of its state among the rows
    counties_of_state = {}
    for county in dict.fromkeys(fips):
        if len(county) == 5:
            counties_of_state.setdefault(county[:2], []).append(county)

    groups = []
    member_of = {}
    for unit, name in enumerate(fips):
        if name in regions:
            member_fips = regions[name]
        elif len(name) == 2:
            member_fips = counties_of_state.get(name, [])
        else:
            continue
        members = [
            index[(member, years[unit])]
            for member in member_fips
            if (member, years[unit]) in index
            and types[index[(member, years[unit])]] == "projection"
        ]
        if not members:
            print(f"region {name} ({years[unit]}) has no member rows")
            continue
        for row in members:
            if row in member_of:
                raise ValueError(
                    f"{fips[row]} ({years[row]}) is a member of two regions: "
                    f"{member_of[row]} and {name}"
                )
            member_of[row] = name
        groups.append((unit, members))
    for unit, _ in groups:
        if unit in member_of:
            raise ValueError(
                f"{fips[unit]} ({years[unit]}) is a region unit and a member "
                f"of {member_of[unit]}"
            )
    return groups


def region_member_mask(
    rows: pl.DataFrame, regions: dict[str, list[str]]
) -> np.ndarray:
    """
    True for the member rows of the region units among `rows` (see
    region_member_groups): the rows to drop, so that a region is predicted
    and scored as one place and its members are not counted again. rows needs
    the columns fips, validation_year and type, with the twin rows already
    dropped (a member with its own observed row stays its own row).
    """
    mask = np.zeros(rows.height, dtype=bool)
    for _, members in region_member_groups(rows, regions):
        mask[members] = True
    return mask
