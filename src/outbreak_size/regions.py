"""
Places made of several counties: the regions and health districts that the
outbreak and vaccination data report as one place (see
analysis/00_explore_data.py).
"""

import polars as pl


def expand_to_member_counties(
    place_rows: pl.DataFrame, fips_list: list[str]
) -> pl.DataFrame:
    """
    One row per county of the place for each row of the place (a row per year
    of the place; a place can have rows in several years), with the county's
    fips in `fips`. For a place with one row this is the row repeated for
    every county.
    """
    # (a cross join has no fixed order: the rows of the place in their order,
    # each with its counties in the order of `fips_list`)
    return (
        place_rows.drop("fips")
        .with_row_index("_place_row")
        .join(
            pl.DataFrame({"fips": fips_list}).with_row_index("_member"),
            how="cross",
        )
        .sort("_place_row", "_member")
        .drop("_place_row", "_member")
    )


def higher_level_regions() -> dict[str, list[str]]:
    assignment_special_places = {
        # Georgia 2024 (6 cases) and 2019 (7 cases) report these places; the
        # names are the case data's. Fulton is in both: they are places of
        # different years (see tests/test_regions.py)
        "Fulton, Gwinett": ["13121", "13135"],
        "Forsyth, Fulton, Paulding": ["13117", "13121", "13223"],
        # "Blue Ridge": ['51023', '51067', '51161'],
        # 'District 1': ['22051', '22071', '21075', '21087'],
        "Gulf Coast": ["02150", "02122", "02063", "02066"],
        # The Tennessee Department of Health's East Tennessee Region (15
        # counties; Knox and the metro regions are separate), the unit of the
        # 2026 case data and of the 2019 "East Tennessee" row:
        # https://www.tn.gov/health/local.html (East Region)
        "East tennessee?": [
            "47001",
            "47009",
            "47013",
            "47025",
            "47029",
            "47057",
            "47063",
            "47089",
            "47105",
            "47123",
            "47129",
            "47145",
            "47151",
            "47155",
            "47173",
        ],
        "Southwest Health District, Utah": [
            "49001",
            "49017",
            "49021",
            "49025",
            "49053",
        ],
        # TDH Mid Cumberland Region, 12 counties:
        # https://www.tn.gov/health/health-program-areas/localdepartments/mid-cumberland-region.html
        "Mid-Cumberland Region, Tennessee": [
            "47021",
            "47043",
            "47083",
            "47085",
            "47125",
            "47147",
            "47149",
            "47161",
            "47165",
            "47169",
            "47187",
            "47189",
        ],
        "Bear River, Utah": ["49003", "49005", "49033"],
        "Southeast Health District, Utah": ["49007", "49015", "49019"],
        # TDH Upper Cumberland Region (Cumberland, not Grundy):
        # https://www.tn.gov/health/health-program-areas/localdepartments/upper-cumberland-region.html
        "Upper Cumberland Region, Tennessee": [
            "47015",
            "47027",
            "47035",
            "47041",
            "47049",
            "47087",
            "47111",
            "47133",
            "47137",
            "47141",
            "47159",
            "47175",
            "47177",
            "47185",
        ],
        "Nashville-Davidson County Region, Tennessee": ["47037"],
        # MDH: "The Twin Cities metro includes the 7-county metropolitan area
        # (Anoka, Carver, Dakota, Hennepin, Ramsey, Scott, and Washington
        # counties)": https://www.health.state.mn.us/diseases/respiratory/stats/hosp.html
        "Twin Cities Metro Area, Minnesota": [
            "27003",
            "27053",
            "27139",
            "27019",
            "27037",
            "27123",
            "27163",
        ],
        # LDH Region 9: Livingston, St. Helena, St. Tammany, Tangipahoa and
        # Washington parishes
        # (https://www.kedm.org/louisiana-news/2025-11-11/confirmed-measles-case-in-region-9,
        # reprinting the LDH release)
        "Region 9, Louisiana": ["22063", "22091", "22103", "22105", "22117"],
        # The Alabama Department of Public Health named only "North Alabama";
        # these are the counties of its Northern District
        # (https://www.alabamapublichealth.gov/about/roster.html), an inference
        "North, Alabama": [
            "01033",
            "01043",
            "01059",
            "01071",
            "01077",
            "01079",
            "01083",
            "01089",
            "01093",
            "01095",
            "01103",
            "01133",
        ],
        "Region 1, Louisiana": ["22051", "22071", "22075", "22087"],
        # https://www.vdh.virginia.gov/content/uploads/2020/05/Locality-to-HD-to-HPR.pdf
        "Piedmont, Virginia": [
            "51007",
            "51029",
            "51037",
            "51049",
            "51111",
            "51135",
            "51147",
        ],
        # Utah local health districts (Utah IBIS):
        # https://ibis.utah.gov/ibisph-view/about/LocalHealth.html
        "Central Utah, Utah": [
            "49023",  # Juab
            "49027",  # Millard
            "49031",  # Piute
            "49039",  # Sanpete
            "49041",  # Sevier
            "49055",  # Wayne
        ],
        "TriCounty, Utah": [
            "49009",  # Daggett
            "49013",  # Duchesne
            "49047",  # Uintah
        ],
        # Weber-Morgan Health Department serves Weber and Morgan counties:
        # https://www.webermorganhealth.gov/about/
        "Weber-Morgan, Utah": ["49057", "49029"],
        # Iowa HHS public health districts (the district map of the Iowa HHS
        # measles page, 4.30.26): https://hhs.iowa.gov/media/19018/download?inline
        "HHS District 3, Iowa": [
            "19131",  # Mitchell
            "19089",  # Howard
            "19191",  # Winneshiek
            "19005",  # Allamakee
            "19033",  # Cerro Gordo
            "19067",  # Floyd
            "19037",  # Chickasaw
            "19065",  # Fayette
            "19043",  # Clayton
            "19069",  # Franklin
            "19023",  # Butler
            "19017",  # Bremer
            "19083",  # Hardin
            "19075",  # Grundy
            "19127",  # Marshall
            "19171",  # Tama
        ],
        "HHS District 4, Iowa": [
            "19165",  # Shelby
            "19009",  # Audubon
            "19077",  # Guthrie
            "19155",  # Pottawattamie
            "19029",  # Cass
            "19001",  # Adair
            "19129",  # Mills
            "19137",  # Montgomery
            "19003",  # Adams
            "19175",  # Union
            "19071",  # Fremont
            "19145",  # Page
            "19173",  # Taylor
            "19159",  # Ringgold
        ],
        "HHS District 5, Iowa": [
            "19079",  # Hamilton
            "19073",  # Greene
            "19015",  # Boone
            "19169",  # Story
            "19049",  # Dallas
            "19153",  # Polk
            "19099",  # Jasper
            "19121",  # Madison
            "19181",  # Warren
            "19125",  # Marion
            "19039",  # Clarke
            "19117",  # Lucas
            "19053",  # Decatur
            "19185",  # Wayne
        ],
        "HHS District 6, Iowa": [
            "19011",  # Benton
            "19157",  # Poweshiek
            "19095",  # Iowa
            "19123",  # Mahaska
            "19107",  # Keokuk
            "19183",  # Washington
            "19135",  # Monroe
            "19179",  # Wapello
            "19101",  # Jefferson
            "19087",  # Henry
            "19007",  # Appanoose
            "19051",  # Davis
            "19177",  # Van Buren
            "19111",  # Lee
        ],
        # Georgia's Coastal Health District: "Bryan, Camden, Chatham,
        # Effingham, Glynn, Liberty, Long & McIntosh": https://coastalhealthdistrict.org/
        "Coastal Health District, Georgia": [
            "13029",  # Bryan
            "13039",  # Camden
            "13051",  # Chatham
            "13103",  # Effingham
            "13127",  # Glynn
            "13179",  # Liberty
            "13183",  # Long
            "13191",  # McIntosh
        ],
        # The Northeast Kansas Healthcare Coalition's 23 counties (the
        # coalition's own site; KDHE's page was not reachable, so medium
        # confidence): https://www.kshcc.com/about-nekshcc.html
        "Northeast HCC Region, Kansas": [
            "20003",  # Anderson
            "20005",  # Atchison
            "20013",  # Brown
            "20017",  # Chase
            "20031",  # Coffey
            "20043",  # Doniphan
            "20045",  # Douglas
            "20059",  # Franklin
            "20061",  # Geary
            "20085",  # Jackson
            "20087",  # Jefferson
            "20107",  # Linn
            "20111",  # Lyon
            "20117",  # Marshall
            "20121",  # Miami
            "20127",  # Morris
            "20131",  # Nemaha
            "20139",  # Osage
            "20149",  # Pottawatomie
            "20161",  # Riley
            "20177",  # Shawnee
            "20197",  # Wabaunsee
            "20201",  # Washington
        ],
        # 'Central Region, Virginia': [
        #     '51007', '51025', '51029', '51036', '51037', '51041', '51570', '51049', '51053', '51595', '51075', '51081', '51083', '51650', '51085', '51087', '51670', '51111', '51117', '51127', '51135', '51730', '51145', '51147', '51149', '51760', '51181', '51183'
        # ],
        # "Northwest Region, Virginia": [
        #     "51015",
        #     "51017",
        #     "51091",
        #     "51163",
        #     "51165",
        #     "51530",
        #     "51660",
        #     "51790",
        #     "51820",
        #     "51043",
        #     "51069",
        #     "51139",
        #     "51171",
        #     "51187",
        #     "51840",
        #     "51033",
        #     "51099",
        #     "51177",
        #     "51179",
        #     "51630",
        #     "51047",
        #     "51061",
        #     "51113",
        #     "51137",
        #     "51157",
        #     "51003",
        #     "51065",
        #     "51079",
        #     "51109",
        #     "51125",
        #     "51540",
        # ],
        # "Eastern Region, Virginia": [
        #     "51550",
        #     "51001",
        #     "51131",
        #     "51650",
        #     "51710",
        #     "51095",
        #     "51199",
        #     "51700",
        #     "51735",
        #     "51830",
        #     "51740",
        #     "51057",
        #     "51073",
        #     "51097",
        #     "51101",
        #     "51103",
        #     "51115",
        #     "51119",
        #     "51133",
        #     "51159",
        #     "51193",
        #     "51810",
        #     "51093",
        #     "51175",
        #     "51620",
        #     "51800",
        # ],
    }
    return assignment_special_places


def members_with_own_row(
    case_rows: pl.DataFrame,
    regions: dict[str, list[str]],
    county_fips: pl.DataFrame,
) -> pl.DataFrame:
    """
    The member counties of each region that have a case row of their own in
    the same year: (place, year, fips), one row per region, year and member.

    A region's cases are those of its counties that the health department does
    not report county by county, so a member with a case row of its own is not
    among the people of the region that year (see members_without_own_row).

    case_rows has the columns year, state, county and fips (null where the case
    sheet gives none), one row per place and year, as in
    outbreaks.process_outbreaks_df. A row is placed as the merge with the
    vaccination data places it (analysis_dataset.merge_outbreaks_vaccinations):
    by its state and county name in county_fips (the columns state, county and
    fips of the vaccination data, whose county names are those of the case
    data), and by the case sheet's own fips only where the name finds none (a
    name with two fips there, a county and an independent city of the same
    name, finds none). The rows of the regions themselves are skipped.
    """
    pairs = pl.DataFrame(
        [
            (place, member)
            for place, members in regions.items()
            for member in members
        ],
        schema={"place": pl.String, "fips": pl.String},
        orient="row",
    )
    by_name = (
        county_fips.filter(pl.col("fips").str.len_chars() == 5)
        .select("state", "county", name_fips="fips")
        .unique()
        .unique(subset=["state", "county"], keep="none")
    )
    return (
        case_rows.filter(~pl.col("county").is_in(list(regions)))
        .join(by_name, on=["state", "county"], how="left")
        .select("year", fips=pl.coalesce("name_fips", "fips"))
        .join(pairs, on="fips")
        .select("place", "year", "fips")
        .unique()
        .sort("place", "year", "fips")
    )


def members_without_own_row(
    regions: dict[str, list[str]],
    own_rows: pl.DataFrame,
    place: str,
    year: int,
) -> list[str]:
    """
    The member fips of `place` that its row of `year` stands for: the members
    that have no case row of their own that year (all members if none has),
    in the order of `regions`. own_rows is members_with_own_row.
    """
    own = set(
        own_rows.filter((pl.col("place") == place) & (pl.col("year") == year))[
            "fips"
        ]
    )
    return [fips for fips in regions[place] if fips not in own]


def expand_to_residual_members(
    place_rows: pl.DataFrame,
    place: str,
    regions: dict[str, list[str]],
    own_rows: pl.DataFrame,
) -> pl.DataFrame:
    """
    expand_to_member_counties for the rows of one place, in each year only to
    the members that have no case row of their own that year
    (members_without_own_row). Raises if a year has no such member: the
    place's cases would belong to counties that all have rows of their own.
    """
    expanded = []
    for year in place_rows["year"].unique(maintain_order=True).to_list():
        members = members_without_own_row(regions, own_rows, place, year)
        if not members:
            raise ValueError(
                f"{place} {year}: every member county has a case row of its "
                "own, so no county is left for the cases of the place"
            )
        expanded.append(
            expand_to_member_counties(
                place_rows.filter(pl.col("year") == year), members
            )
        )
    return pl.concat(expanded)


def require_no_own_row_members(
    used_members: pl.DataFrame, case_rows: pl.DataFrame
) -> None:
    """
    Raise if a county that a region was built from (used_members: place, year,
    fips) has a case row of its own that year (case_rows: year and fips, null
    where unknown). It would be counted in the region and in its own row.
    """
    own = (
        case_rows.filter(pl.col("fips").is_not_null())
        .select("year", "fips")
        .unique()
    )
    overlap = used_members.join(own, on=["year", "fips"], how="inner")
    if overlap.height:
        raise ValueError(
            f"{overlap.height} counties are in the people of a region and "
            "have a case row of their own that year (place, year, fips): "
            f"{overlap.sort('place', 'year', 'fips').head(5).rows()}"
        )


def require_filled_rows_use_same_members(
    rows: pl.DataFrame,
    regions: dict[str, list[str]],
    own_rows: pl.DataFrame,
) -> None:
    """
    Raise if a row of a region that took the vaccination values of the earlier
    school year (vaccine_from_earlier_year, see
    analysis_dataset.fill_from_earlier_school_year) took them from a row
    built for other counties.

    A case row of year Y usually takes the vaccination row of school year Y-1,
    which was averaged over the members the region stands for in Y. The filled
    row takes the school year Y-2, averaged over the members of Y-1: the same
    counties only if no member has a case row of its own in exactly one of the
    two years. rows has the columns county, year and vaccine_from_earlier_year.
    """
    filled = rows.filter(
        pl.col("county").is_in(list(regions))
        & pl.col("vaccine_from_earlier_year").fill_null(False)
    )
    different = [
        (place, year)
        for place, year in filled.select("county", "year").iter_rows()
        if members_without_own_row(regions, own_rows, place, year)
        # the members of the PREVIOUS OUTBREAK year (not a data year)
        != members_without_own_row(regions, own_rows, place, year - 1)
    ]
    if different:
        raise ValueError(
            "regions that took the vaccination values of the earlier school "
            "year from a row built for other counties (place, year): "
            f"{different}"
        )


def require_own_row_members_in_main_rows(
    own_rows: pl.DataFrame,
    region_rows: pl.DataFrame,
    main_rows: pl.DataFrame,
) -> None:
    """
    Raise if a member that a region leaves out because it has a case row of
    its own has no main row (the rows with vaccine data, which are the train
    rows of the projection datasets) in a year in which the region has a row.

    Without a train row the member's projection row would be dropped as a
    member of the region (validation.region_member_mask), so its people would
    be in no row at all. own_rows is members_with_own_row; region_rows has
    the columns place and year (the rows of the regions); main_rows has year
    and fips.
    """
    needed = own_rows.join(
        region_rows.select("place", "year").unique(),
        on=["place", "year"],
        how="inner",
    )
    missing = needed.join(
        main_rows.select("year", "fips").unique(),
        on=["year", "fips"],
        how="anti",
    )
    if missing.height:
        raise ValueError(
            f"{missing.height} members left out of a region because they "
            "have a case row of their own have no row with vaccine data, so "
            "they would be in no row (place, year, fips): "
            f"{missing.sort('place', 'year', 'fips').head(5).rows()}"
        )
