"""
The county-years the importation model is validated on: every one for which
we know something about the imports, whether or not the county is in the
analysis or projection data.

A county-year is known when
  - the county-level import data names it (its imports, possibly 0), or
  - it is another county of a state-year whose named counties account for all
    of the state's imports: it had 0, or
  - it is a county of a state-year with no imports at all: it had 0.
A county of a state-year with imports that no county-level row accounts for is
unknown and left out.

Places that are not one county are units: New York City (five counties) and
the regions in `regions` (name -> county fips). Each unit's share of its state
is the population of its counties.
"""

import math
import os
import re

import polars as pl
from addfips import AddFIPS

NYC_COUNTIES = ["36005", "36047", "36061", "36081", "36085"]
# misspelled county names in the import data: (state, name) -> county
SPELLING_FIXES = {("Michigan", "Washentaw"): "Washtenaw"}


def county_universe(data_dir: str) -> pl.DataFrame:
    """All the counties of the 50 states and DC: fips, state, county."""
    import geopandas as gpd

    counties = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    states = gpd.read_file(os.path.join(data_dir, "cb_2020_us_state_20m.shp"))
    state_names = dict(zip(states["STATEFP"], states["NAME"]))
    counties = counties[counties["STATEFP"] <= "56"]
    return pl.DataFrame(
        {
            "fips": counties["GEOID"].tolist(),
            "state": [state_names[s] for s in counties["STATEFP"]],
            "county": counties["NAME"].tolist(),
        }
    )


def _normalized(name: str) -> str:
    name = re.sub(r"\b(county|parish|city|borough)\b", "", name.lower())
    return re.sub(r"[^a-z]", "", name)


def resolve_known_rows(
    known: pl.DataFrame,
    universe: pl.DataFrame,
    regions: dict[str, list[str]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Map the county-level import rows (year, state, county, imported) to places
    made of counties.

    Returns (resolved, dropped). resolved has year, state, place, members (the
    county fips of the place) and imported; dropped holds the rows that match
    no county and no region.
    """
    add_fips = AddFIPS()
    by_state = {}
    for row in universe.iter_rows(named=True):
        names = by_state.setdefault(row["state"], {})
        names[_normalized(row["county"])] = row["fips"]
    universe_fips = set(universe["fips"])

    resolved, dropped = [], []
    for row in known.iter_rows(named=True):
        state, county = row["state"], row["county"]
        base = {
            "year": row["year"],
            "state": state,
            "imported": row["imported"],
        }
        if county.strip().lower() == "new york city":
            resolved.append({**base, "place": county, "members": NYC_COUNTIES})
            continue
        if county in regions:
            members = [f for f in regions[county] if f in universe_fips]
            if members:
                resolved.append({**base, "place": county, "members": members})
            else:
                dropped.append({**base, "county": county})
            continue
        county = SPELLING_FIXES.get((state, county)) or county
        fips = add_fips.get_county_fips(county, state=state)
        # a code that is not a county of the universe (a county that no longer
        # exists, say) is not used
        if fips not in universe_fips:
            fips = None
        if fips is None:
            fips = by_state.get(state, {}).get(_normalized(county))
        if fips is None:
            dropped.append({**base, "county": county})
        else:
            resolved.append({**base, "place": county, "members": [fips]})
    # the same place twice in a state-year is one place
    merged = {}
    for row in resolved:
        key = (row["year"], row["state"], tuple(row["members"]))
        if key in merged:
            merged[key]["imported"] += row["imported"]
        else:
            merged[key] = dict(row)
    resolved = list(merged.values())
    return (
        pl.DataFrame(
            resolved,
            schema={
                "year": pl.Int64,
                "state": pl.String,
                "imported": pl.Int64,
                "place": pl.String,
                "members": pl.List(pl.String),
            },
        ),
        pl.DataFrame(
            dropped,
            schema={
                "year": pl.Int64,
                "state": pl.String,
                "imported": pl.Int64,
                "county": pl.String,
            },
        ),
    )


def validation_county_years(
    resolved: pl.DataFrame,
    dropped: pl.DataFrame,
    state_year_totals: pl.DataFrame,
    universe: pl.DataFrame,
    populations: pl.DataFrame,
    years: list[int],
) -> pl.DataFrame:
    """
    The known county-years with their state's population shares.

    state_year_totals: state, year, total (the state's imports that year; a
    state-year that is absent had none). populations: fips, year, pop for
    every county of the universe and every year.

    Returns one row per known place and year: year, state, place, fips (the
    first county of the place), n_counties, imported (0 for the unnamed
    counties), source, pop, share and share_log (the place's share of its
    state's population, and of the state's log10-population).
    """
    missing = (
        universe.select("fips")
        .join(pl.DataFrame({"year": years}), how="cross")
        .join(populations, on=["fips", "year"], how="anti")
        # (a cross and an anti join have no fixed order: the examples in the
        # error below are the first county-years)
        .sort("fips", "year")
    )
    if missing.height > 0:
        raise ValueError(
            f"no population for {missing.height} county-years, e.g. "
            f"{missing.head(5).rows()}"
        )
    totals = {
        (r["state"], r["year"]): r["total"]
        for r in state_year_totals.iter_rows(named=True)
    }
    pop = {
        (r["fips"], r["year"]): r["pop"]
        for r in populations.iter_rows(named=True)
    }
    state_counties = {}
    for r in universe.iter_rows(named=True):
        state_counties.setdefault(r["state"], []).append(r["fips"])
    # a state spelled another way would be missed by the loops below: its
    # counties would be taken for a state-year without imports
    in_years = pl.col("year").is_in(years)
    other_states = (
        set(resolved.filter(in_years)["state"])
        | set(dropped.filter(in_years)["state"])
        | set(state_year_totals.filter(in_years)["state"])
    ) - set(state_counties)
    if other_states:
        raise ValueError(
            f"states that are not in the county universe: {other_states}"
        )
    named_rows, lost_rows = {}, {}
    for r in resolved.iter_rows(named=True):
        named_rows.setdefault((r["state"], r["year"]), []).append(r)
    for r in dropped.iter_rows(named=True):
        lost_rows.setdefault((r["state"], r["year"]), []).append(r)
    rows = []
    over_attributed = []
    for year in years:
        for state, counties in state_counties.items():
            named = named_rows.get((state, year), [])
            lost = lost_rows.get((state, year), [])
            total = totals.get((state, year), 0)
            attributed = sum(r["imported"] for r in named) + sum(
                r["imported"] for r in lost
            )
            lost_positive = any(r["imported"] >= 1 for r in lost)
            if attributed > total:
                over_attributed.append((state, year, attributed, total))
            if named:
                others_known = attributed >= total and not lost_positive
                others_source = "other county of a state-year with county rows"
            elif total == 0 and not lost_positive:
                others_known = True
                others_source = "state-year without imports"
            else:
                others_known = False
                others_source = ""
            # the state's partition into places: the named places and the
            # remaining counties
            places = []
            used = set()
            for r in named:
                members = r["members"]
                if used & set(members):
                    raise ValueError(
                        f"{state} {year}: {r['place']} overlaps another place"
                    )
                used.update(members)
                places.append((r["place"], members, r["imported"], "named"))
            for fips in counties:
                if fips not in used:
                    places.append((fips, [fips], 0, others_source))
            sizes = [
                sum(pop[(f, year)] for f in members)
                for _, members, _, _ in places
            ]
            state_pop = sum(sizes)
            state_log = sum(math.log10(size) for size in sizes)
            for (place, members, imported, source), size in zip(places, sizes):
                if source == "named" or others_known:
                    rows.append(
                        {
                            "year": year,
                            "state": state,
                            "place": place,
                            "fips": members[0],
                            "n_counties": len(members),
                            "imported": imported,
                            "source": source,
                            "pop": size,
                            "share": size / state_pop,
                            "share_log": math.log10(size) / state_log,
                        }
                    )
    if over_attributed:
        # the county rows and the state's imports disagree; the county rows
        # are taken as complete (the unnamed counties had 0)
        print(
            "import validation: state-years whose county rows sum to more "
            "than the state's imports (state, year, county rows, state): "
            f"{over_attributed}"
        )
    return pl.DataFrame(rows)


def fill_missing_populations(
    universe: pl.DataFrame,
    years: list[int],
    populations: pl.DataFrame,
    query_pop,
) -> pl.DataFrame:
    """
    The populations (fips, year, pop) with every county-year of the universe:
    the ones missing from `populations` are asked from query_pop(years, fips)
    (the Census), which returns their populations in the same order.
    """
    missing = (
        universe.select("fips")
        .join(pl.DataFrame({"year": years}), how="cross")
        .join(populations, on=["fips", "year"], how="anti")
    )
    if missing.height == 0:
        return populations
    print(
        f"import validation: asking the Census for {missing.height} "
        "county-year populations"
    )
    asked = missing.with_columns(
        pop=pl.Series(
            query_pop(missing["year"].to_list(), missing["fips"].to_list())
        ).cast(pl.Float64)
    )
    return pl.concat(
        [
            populations.select("fips", "year", "pop").cast(
                {"pop": pl.Float64}
            ),
            asked,
        ],
        how="vertical",
    )
