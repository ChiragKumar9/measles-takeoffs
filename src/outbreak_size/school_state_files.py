"""
School-level vaccination files published by states (the "state files" of the
school-data zip), put in the layout of school_vaccination.py (`standardize`).

Fifteen readers, one parse function each (parse_<state>), in the order they
were written: Maryland (kindergarten MMR, every school), New York (all grades,
school-wide measles rate, no enrollment), California (kindergarten MMR, mostly
as bands such as "≥95%"), South Carolina (45-day report PDFs: all grades, the
share of students with the required immunizations, not MMR), Ohio, Iowa (PDFs,
all grades, all-vaccine rate), Maine (Excel and PDFs), Wisconsin, Colorado,
Indiana, North Carolina, Massachusetts, Minnesota, Pennsylvania and Texas
(kindergarten MMR unless a reader says otherwise; each section below and
analysis/read_school_state_files.py say what a file holds).

The parsing functions take a table that is already read (a polars frame of the
sheet with no header, or the text of a PDF), so that they can be tested without
files. The readers of a file only read it and call them.

Rules that are the same in all the readers
    * A censored count ("<10", "*", "DS") is a missing enrollment (as in
      school_vaccination.read_chen_bento); the rate of the school is kept.
    * A suppressed rate ("**", "--*", ".") is a missing rate.
    * A band ("≥95%", ">96%", "≤5%") is the middle of the interval it stands
      for, given the other bands of the same file (`band_midpoints`).
    * Rates are percentages (0-100). A file in fractions (median at most 1.5) is
      multiplied by 100, as in read_chen_bento.
    * Every dropped or changed row goes to the QC log with its reason.
    * The county of a school is the county the file names, looked up in the
      county polygons' names of the state (`map_counties`); a name that matches
      no county or several counties is an error.
"""

import html
import io
import re

import numpy as np
import polars as pl

from outbreak_size.school_vaccination import standardize

STATE_FILE = "state_file"
BAND = re.compile(r"\s*([≥>≤<])\s*(\d+(?:\.\d+)?)\s*%?\s*$")


# --- values -----------------------------------------------------------------


def band_midpoints(texts):
    """
    The value of each band text of a file: the middle of the interval it stands
    for, given the other bands of the same file.
        Top bands ("≥95%", ">96%"): the band of threshold t runs from t to the
        next higher threshold (not included), and the highest runs to 100. With
        ">95%" and ">96%": 95.5 and 98.0; with ">95%" alone: 97.5.
        Bottom bands ("≤1%", "≤5%"): the band of threshold t runs from the next
        lower threshold (not included; 0 for the lowest) to t. With "≤1%" and
        "≤5%": 0.5 and 3.0; with "≤5%" alone: 2.5.
    Returns {text: midpoint} for every text that is a band (others are left out).
    """
    parsed = {}
    for text in set(texts):
        match = BAND.match(text) if isinstance(text, str) else None
        if match:
            parsed[text] = (match.group(1), float(match.group(2)))
    upper = sorted({t for op, t in parsed.values() if op in "≥>"})
    lower = sorted({t for op, t in parsed.values() if op in "≤<"})
    middle = {}
    for text, (op, t) in parsed.items():
        if op in "≥>":
            higher = [u for u in upper if u > t]
            top = higher[0] if higher else 100.0
            middle[text] = (t + top) / 2
        else:
            below = [u for u in lower if u < t]
            bottom = below[-1] if below else 0.0
            middle[text] = (bottom + t) / 2
    return middle


def parse_number(value):
    """A float from a number or a text such as "97.3%", "  22"; None otherwise."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (
            None
            if isinstance(value, float) and np.isnan(value)
            else float(value)
        )
    text = str(value).strip().rstrip("%").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_rates(values):
    """
    A column of rates as text or numbers, in percent. Returns (rates, bands):
    `rates` has the number of each value (None for a band or a suppressed
    value) and `bands` the band text of each (None for a number).
    """
    rates, bands = [], []
    for value in values:
        number = parse_number(value)
        text = value.strip() if isinstance(value, str) else None
        is_band = text is not None and BAND.match(text) is not None
        rates.append(None if is_band else number)
        bands.append(text if is_band else None)
    return rates, bands


def scale_factor(rates):
    """100 if the rates are fractions (median at most 1.5), else 1 (read_chen_bento's rule)."""
    known = [r for r in rates if r is not None]
    if not known:
        return 1.0
    return 100.0 if float(np.median(known)) <= 1.5 else 1.0


# --- counties ---------------------------------------------------------------


def county_key(name):
    """
    A county name as letters only, lower case, 'saint' as 'st', without a final
    'county' ('city' stays: Baltimore City and Baltimore County differ).
    """
    text = str(name).lower().replace("saint", "st")
    text = re.sub(r"[^a-z]", "", text)
    if text.endswith("county") and len(text) > len("county"):
        text = text[: -len("county")]
    return text


# The county polygons name Baltimore city and Baltimore County both "Baltimore"
# (24510 and 24005); a file's "Baltimore City" is the city, its "Baltimore" or
# "Baltimore County" the county. (state, county_key of the file's name): fips
COUNTY_OVERRIDES = {
    ("MD", "baltimorecity"): "24510",
    ("MD", "baltimore"): "24005",
    # misspelt county labels of the New York file, read from the school's city and zip code:
    (
        "NY",
        "chatauqua",
    ): "36013",  # 2 private schools in Mayville 14701: Chautauqua
    ("NY", "eriephysica"): "36029",  # a school in Hamburg 14075: Erie
    ("NY", "warne"): "36117",  # a school in Clyde 14433: Wayne
    (
        "IA",
        "pottawattami",
    ): "19155",  # Pottawattamie: one row each in the 2024-25 and 2025-26 files
}


def map_counties(names, state, counties):
    """
    {file county name: (county_fips, county_name)} for the names of one state.
    `counties` has county_fips, county_name, county_state. Raises ValueError
    for a name that matches no county of the state or more than one (the
    names of COUNTY_OVERRIDES are looked up by their FIPS).
    """
    own = counties.filter(pl.col("county_state") == state).with_columns(
        key=pl.col("county_name").map_elements(
            county_key, return_dtype=pl.String
        )
    )
    by_key = {}
    for fips, name, key in own.select(
        "county_fips", "county_name", "key"
    ).iter_rows():
        by_key.setdefault(key, []).append((fips, name))
    by_fips = {
        fips: (fips, name)
        for fips, name, _ in own.select(
            "county_fips", "county_name", "key"
        ).iter_rows()
    }
    found, problems = {}, []
    for name in sorted({n for n in names if n is not None}):
        override = COUNTY_OVERRIDES.get((state, county_key(name)))
        matches = (
            [by_fips[override]]
            if override in by_fips
            else by_key.get(county_key(name), [])
        )
        if len(matches) == 1:
            found[name] = matches[0]
        else:
            problems.append((name, [m[1] for m in matches]))
    if problems:
        raise ValueError(
            f"{state}: county names without exactly one match: {problems}"
        )
    return found


# --- common finish ----------------------------------------------------------


def finish(df, state):
    """Row ids, source, the county of the label, and the common layout."""
    df = df.with_row_index("rid").with_columns(
        row_id=pl.format(
            "{}:{}:{}:{}",
            pl.lit(STATE_FILE),
            pl.lit(state),
            pl.col("year_start"),
            pl.col("rid"),
        ),
        source=pl.lit(STATE_FILE),
        state=pl.lit(state),
        year=pl.format(
            "{}-{}", pl.col("year_start"), pl.col("year_start") + 1
        ),
        year_end=pl.col("year_start") + 1,
        county_join=pl.lit("label"),
        copied_year=pl.lit(False),
        # a known enrollment says where it came from (the school file checks it)
        enrollment_source=pl.when(pl.col("enrollment").is_not_null()).then(
            pl.lit(STATE_FILE)
        ),
    )
    return standardize(df)


def log_rows(log, df, state, step, action, reason):
    """Log rows of a reader's own frame (its columns are enough for the QC log)."""
    if df.height == 0:
        return
    rows = df.with_columns(
        row_id=pl.lit(f"{STATE_FILE}:{state}:"),
        source=pl.lit(STATE_FILE),
        state=pl.lit(state),
        year=pl.col("year_start").cast(pl.String),
        lon=pl.lit(None, dtype=pl.Float64),
        lat=pl.lit(None, dtype=pl.Float64),
    )
    log.add(rows, step, action, reason)


def school_type(text):
    text = None if text is None else str(text).strip().lower()
    return {"public": "PUBLIC", "private": "PRIVATE"}.get(text)


def header_row(frame, first_cell, limit=15):
    """The position of the row whose first cell reads `first_cell` (case and spaces ignored)."""
    for i in range(min(limit, frame.height)):
        cell = frame.row(i)[0]
        if (
            cell is not None
            and str(cell).strip().lower() == first_cell.lower()
        ):
            return i
    raise ValueError(
        f"no header row starting with {first_cell!r} in the first {limit} rows"
    )


def with_header(frame, at):
    """The rows below row `at` of a headerless frame, with that row's text as column names."""
    names = [
        str(v).strip() if v is not None else f"column_{j}"
        for j, v in enumerate(frame.row(at))
    ]
    unique = []
    for j, name in enumerate(names):
        unique.append(name if name not in unique else f"{name}_{j}")
    body = frame[at + 1 :]
    return body.rename(dict(zip(body.columns, unique)))


# --- Maryland -----------------------------------------------------------------


def parse_maryland(frame, year_start, log=None):
    """
    One Maryland kindergarten sheet (headerless frame). The header row is the
    one that starts with "School Name" (it moves between files). Enrollment is
    "Total K Students" (before 2024-25) or "Total Number of Enrolled Students";
    the rate is "% MMR"; exemptions are "% Medical Exemption" and "% Religious
    Exemption". Returns (frame, info): the rows of schools with the columns
    school_name, county_label, school_type, enrollment, rate, exemption_*,
    year_start, and the scale found.
    """
    table = with_header(frame, header_row(frame, "School Name"))
    lower = {c.lower().strip(): c for c in table.columns}

    def pick(*starts, exact=False):
        for key, column in lower.items():
            if any((key == s) if exact else key.startswith(s) for s in starts):
                return column
        raise ValueError(f"no column {starts} in {list(table.columns)}")

    name_c = pick("school name", exact=True)
    county_c = pick("county", exact=True)
    type_c = pick("type of school")
    enroll_c = (
        lower["total k students"]
        if "total k students" in lower
        else pick("total number of enrolled")
    )
    mmr_c = pick("% mmr", exact=True)
    med_c, rel_c = pick("% medical"), pick("% religious")
    table = table.filter(pl.col(name_c).is_not_null())
    totals = table.filter(
        pl.col(name_c)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .str.contains(r"^(total|state|maryland)\b")
    )
    table = (
        table.filter(~pl.col(name_c).is_in(totals[name_c].to_list()))
        if totals.height
        else table
    )
    rate, _ = parse_rates(table[mmr_c].to_list())
    factor = scale_factor(rate)
    med, _ = parse_rates(table[med_c].to_list())
    rel, _ = parse_rates(table[rel_c].to_list())
    enroll = [parse_number(v) for v in table[enroll_c].to_list()]
    out = pl.DataFrame(
        {
            "school_name": table[name_c].cast(pl.String).str.strip_chars(),
            "county_label": table[county_c].cast(pl.String).str.strip_chars(),
            "school_type": [school_type(v) for v in table[type_c].to_list()],
            "enrollment": enroll,
            "rate": [None if r is None else r * factor for r in rate],
            "exemption_medical": [
                None if r is None else r * factor for r in med
            ],
            "exemption_religious": [
                None if r is None else r * factor for r in rel
            ],
        },
        schema_overrides={
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "exemption_medical": pl.Float64,
            "exemption_religious": pl.Float64,
        },
    ).with_columns(year_start=pl.lit(year_start, dtype=pl.Int64))
    # a negative rate (such as -583%) is an error of the file, not a low rate: clipping it to 0 would
    # make a 0% school, so it is missing (a rate above 100 is left to the pipeline's clip to 100)
    negative = out.filter(pl.col("rate") < 0)
    out = out.with_columns(
        rate=pl.when(pl.col("rate") < 0).then(None).otherwise(pl.col("rate"))
    )
    censored = out.filter(pl.col("enrollment").is_null())
    suppressed = out.filter(pl.col("rate").is_null())
    if log is not None:
        log_rows(
            log,
            negative,
            "MD",
            "maryland values",
            "changed",
            "negative rate (error in the file): missing",
        )
        log_rows(
            log,
            totals.select(
                school_name=pl.col(name_c), year_start=pl.lit(year_start)
            ),
            "MD",
            "maryland rows",
            "dropped",
            "total or footer row",
        )
        log_rows(
            log,
            censored,
            "MD",
            "maryland values",
            "changed",
            "enrollment censored (<10): missing",
        )
        log_rows(
            log,
            suppressed,
            "MD",
            "maryland values",
            "changed",
            "rate suppressed (**): missing",
        )
    info = {
        "year_start": year_start,
        "rows": out.height,
        "dropped_totals": totals.height,
        "scale": "fraction x100" if factor == 100 else "percent",
        "enrollment_censored": censored.height,
        "rate_missing": suppressed.height,
        "rate_negative": negative.height,
    }
    return out, info


# --- New York -----------------------------------------------------------------

POINT = re.compile(r"POINT \((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)")


def parse_new_york(table, log=None):
    """
    The New York school immunization survey (one row per school and school year,
    all grades, no enrollment). "School District" rows are district totals and
    are dropped. The rate is "Percent Immunized Measles" (the measles vaccine,
    for the whole school); "Percent Medical Exemptions" is kept (religious
    exemptions only in the 2012-2018 file). The point is read from "Location"
    where it exists. A school listed twice with the same School ID, school
    year and measles rate is one school (20 rows of the two files; the 4
    School ID and year pairs whose rates differ are kept). Returns (frame,
    info).
    """
    table = table.with_columns(pl.col("Type").cast(pl.String))
    districts = table.filter(pl.col("Type") == "School District")
    nameless = table.filter(
        (pl.col("Type") != "School District") & pl.col("School Name").is_null()
    )
    schools = table.filter(
        (pl.col("Type") != "School District")
        & pl.col("School Name").is_not_null()
    ).with_columns(
        year_start=pl.col("Report Period").str.slice(0, 4).cast(pl.Int64)
    )
    first = pl.struct(
        "School ID", "Report Period", "Percent Immunized Measles"
    ).is_first_distinct()
    repeats = schools.filter(~first)
    schools = schools.filter(first)
    rate, _ = parse_rates(schools["Percent Immunized Measles"].to_list())
    med, _ = parse_rates(schools["Percent Medical Exemptions"].to_list())
    rel = (
        parse_rates(schools["Percent Religious Exemptions"].to_list())[0]
        if "Percent Religious Exemptions" in schools.columns
        else [None] * schools.height
    )
    points = (
        [POINT.search(v) if v else None for v in schools["Location"].to_list()]
        if "Location" in schools.columns
        else [None] * schools.height
    )
    out = pl.DataFrame(
        {
            "school_name": schools["School Name"],
            "address": schools["Street"],
            "county_label": schools["County"]
            .cast(pl.String)
            .str.strip_chars(),
            "school_type": [
                {
                    "public school": "PUBLIC",
                    "charter school": "PUBLIC",
                    "private school": "PRIVATE",
                }.get(str(v).lower())
                for v in schools["Type"].to_list()
            ],
            "ny_type": schools["Type"],
            "enrollment": [None] * schools.height,
            "rate": rate,
            "exemption_medical": med,
            "exemption_religious": rel,
            "lon": [float(m.group(1)) if m else None for m in points],
            "lat": [float(m.group(2)) if m else None for m in points],
            "year_start": schools["year_start"],
            "school_id": schools["School ID"],
        },
        schema_overrides={
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "exemption_medical": pl.Float64,
            "exemption_religious": pl.Float64,
            "lon": pl.Float64,
            "lat": pl.Float64,
        },
    )
    if log is not None:
        log_rows(
            log,
            districts.select(
                school_name=pl.col("School Name"),
                year_start=pl.col("Report Period")
                .str.slice(0, 4)
                .cast(pl.Int64),
            ),
            "NY",
            "new york rows",
            "dropped",
            "district total row (Type = School District)",
        )
        log_rows(
            log,
            nameless.select(
                school_name=pl.col("School Name"),
                year_start=pl.col("Report Period")
                .str.slice(0, 4)
                .cast(pl.Int64),
            ),
            "NY",
            "new york rows",
            "dropped",
            "row without a school name",
        )
        log_rows(
            log,
            repeats.select(
                school_name=pl.col("School Name"),
                year_start=pl.col("year_start"),
            ),
            "NY",
            "new york rows",
            "dropped",
            "the same school (School ID, year and measles rate) listed twice",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "NY",
            "new york values",
            "changed",
            "measles rate missing in the file",
        )
    info = {
        "rows": out.height,
        "district_rows_dropped": districts.height,
        "nameless_rows_dropped": nameless.height,
        "rate_missing": out.filter(pl.col("rate").is_null()).height,
        "rate_below_10": out.filter(pl.col("rate") < 10).height,
        "duplicate_school_years": out.select("school_id", "year_start")
        .is_duplicated()
        .sum(),
    }
    return out, info


# --- California ---------------------------------------------------------------


def parse_california(frame, year_start=None, log=None):
    """
    One California sheet "Enrollment 20 or More" (headerless frame; schools with
    19 or fewer kindergartners are on another sheet and have no rates). Three
    layouts: (2017-2020) a header row "SCHOOL CODE ..." with "#" and "%" in the
    row below; (2020-22) the same with "School Year" and "Grade" columns; (2022
    on) one header row with "MMR‡ Percent"; the last two also have a "Grade"
    column (only "Kindergarten" is kept). The rate is the MMR percent: a number,
    a band ("≥95%", "≥98%", "≥99%", "≤1%", "≤5%": the middle of its interval,
    band_midpoints, over the kindergarten rows of the file) or missing ("." or
    "--*"). Returns (frame, info); `rate_source` is "mmr" (a band's text
    is in `band`).
    """
    at = header_row(frame, "School Code")
    top = [str(v).strip() if v is not None else "" for v in frame.row(at)]
    below = [
        str(v).strip() if v is not None else "" for v in frame.row(at + 1)
    ]
    two_level = any(v in ("%", "#") for v in below)
    first = at + 2 if two_level else at + 1
    low = [t.lower() for t in top]

    def column(*needles):
        for j, t in enumerate(low):
            if all(n in t for n in needles):
                return j
        raise ValueError(f"no column {needles} in {top}")

    mmr = next(j for j, t in enumerate(low) if t.startswith("mmr"))
    if two_level:
        mmr_pct = mmr + 1 if below[mmr + 1] == "%" else mmr
    else:
        mmr_pct = column("mmr", "percent")
    j_school, j_county = column("school name"), column("county")
    j_type = next(
        j for j, t in enumerate(low) if t.startswith(("pub/", "public/"))
    )
    j_enroll = column("enroll")
    j_city = column("city")
    j_grade = next((j for j, t in enumerate(low) if t == "grade"), None)
    j_year = next((j for j, t in enumerate(low) if t == "school year"), None)
    body = frame[first:]
    rows = body.rows()
    keep = [r for r in rows if r[j_school] is not None]
    grades = [
        r[j_grade] if j_grade is not None else "Kindergarten" for r in keep
    ]
    kinder = [str(g).strip().lower() == "kindergarten" for g in grades]
    other_grades = sum(1 for k in kinder if not k)
    keep = [r for r, k in zip(keep, kinder) if k]
    texts = [
        None if r[mmr_pct] is None else str(r[mmr_pct]).strip() for r in keep
    ]
    rates, bands = parse_rates(texts)
    middle = band_midpoints([b for b in bands if b])
    out = pl.DataFrame(
        {
            "school_name": [str(r[j_school]).strip() for r in keep],
            "county_label": [str(r[j_county]).strip() for r in keep],
            "city": [
                None if r[j_city] is None else str(r[j_city]).strip()
                for r in keep
            ],
            "school_type": [school_type(r[j_type]) for r in keep],
            "enrollment": [parse_number(r[j_enroll]) for r in keep],
            "rate": [
                r if b is None else middle[b] for r, b in zip(rates, bands)
            ],
            "rate_source": [
                None if (r is None and b is None) else "mmr"
                for r, b in zip(rates, bands)
            ],
            "band": bands,
            "year_start": [
                int(str(r[j_year])[:4]) if j_year is not None else year_start
                for r in keep
            ],
        },
        schema_overrides={
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "band": pl.String,
            "year_start": pl.Int64,
        },
    )
    missing = out.filter(pl.col("rate").is_null())
    if log is not None:
        log_rows(
            log,
            out.filter(pl.col("band").is_not_null()),
            "CA",
            "california values",
            "changed",
            "banded rate replaced by the middle of its interval",
        )
        log_rows(
            log,
            missing,
            "CA",
            "california values",
            "changed",
            "rate missing or suppressed in the file",
        )
    info = {
        "rows": out.height,
        "other_grade_rows_dropped": other_grades,
        "banded": out.filter(pl.col("band").is_not_null()).height,
        "bands": sorted(middle.items()),
        "rate_missing": missing.height,
        "layout": "two header rows" if two_level else "one header row",
    }
    return out, info


# --- South Carolina -----------------------------------------------------------

SC_COUNTIES = [
    "Abbeville", "Aiken", "Allendale", "Anderson", "Bamberg", "Barnwell", "Beaufort", "Berkeley", "Calhoun",
    "Charleston", "Cherokee", "Chester", "Chesterfield", "Clarendon", "Colleton", "Darlington", "Dillon",
    "Dorchester", "Edgefield", "Fairfield", "Florence", "Georgetown", "Greenville", "Greenwood", "Hampton",
    "Horry", "Jasper", "Kershaw", "Lancaster", "Laurens", "Lee", "Lexington", "Marion", "Marlboro",
    "McCormick", "Newberry", "Oconee", "Orangeburg", "Pickens", "Richland", "Saluda", "Spartanburg",
    "Sumter", "Union", "Williamsburg", "York",
]  # fmt: skip
# region, county (longest names first: Chesterfield before Chester; two counties
# joined by "/" are a school on a county line, which has no single county), the
# type (the 2025-26 report sometimes glues it to the county), the school, city
# and grades, the number of students and the rate
SC_REGION = r"(?:Upstate|Midlands|Pee\s?Dee|Low\s?Count\w*)"
SC_COUNTY = "(?:" + "|".join(sorted(SC_COUNTIES, key=len, reverse=True)) + ")"
SC_LINE = re.compile(
    r"^\s*(?P<region>" + SC_REGION + r")\s+"
    r"(?P<county>" + SC_COUNTY + r"(?:/" + SC_COUNTY + r")?)\s*"
    r"(?P<type>Public|Private|Charter)\s+"
    r"(?P<rest>.+?)\s+"
    r"(?P<students><\s*10|\d[\d,]*)\s+"
    r"(?P<rate>[<>≥≤]?\s*\d+(?:\.\d+)?%)\s*$",
    re.IGNORECASE,
)
SC_SCHOOL_START = re.compile(r"^\s*" + SC_REGION + r"\s", re.IGNORECASE)
# the 2025-26 report cuts long county names ("Spartanbur"): any word, resolved below
SC_LINE_CUT = re.compile(
    r"^\s*(?P<region>" + SC_REGION + r")\s+"
    r"(?P<county>[A-Za-z]{4,}(?:/[A-Za-z]{4,})?)\s+"
    r"(?P<type>Public|Private|Charter)\s+"
    r"(?P<rest>.+?)\s+"
    r"(?P<students><\s*10|\d[\d,]*)\s+"
    r"(?P<rate>[<>≥≤]?\s*\d+(?:\.\d+)?%)\s*$",
    re.IGNORECASE,
)


def full_sc_county(word):
    """A county named by its full name or by a cut name (a prefix of exactly one county); None otherwise."""
    parts = []
    for part in word.split("/"):
        hits = [c for c in SC_COUNTIES if c.lower() == part.lower()] or [
            c for c in SC_COUNTIES if c.lower().startswith(part.lower())
        ]
        if len(hits) != 1:
            return None
        parts.append(hits[0])
    return "/".join(parts)


def includes_kindergarten(grades):
    """
    Does a grade range such as "K5-12", "PK-5th", "Infants - K", "3K-5th grade"
    include kindergarten? It does when the range starts below first grade and
    does not end before kindergarten (a range of 3K/4K/PK only does not).
    """
    text = str(grades).strip().lower().replace(" ", "")
    if not text:
        return None
    parts = re.split(r"-|to", text)
    start, end = parts[0], parts[-1]
    starts_below_first = bool(
        re.match(r"^(infants?|pre-?k|pk|\d?k)", start)
    ) or start in ("k",)
    only_preschool = end in ("pk", "3k", "4k", "pre-k", "prek") or text in (
        "pk",
        "3k",
        "4k",
        "pre-k",
        "prek",
    )
    return bool(starts_below_first and not only_preschool)


def parse_south_carolina(text, log=None):
    """
    The text of an SC "45-Day Report of Schools with Required Immunization
    Certification" (pdftotext -layout). One row per school: region, county, type,
    school, city, grade range, total students and the percent of students with
    the required immunizations (all grades; not MMR). "<10" students is a missing
    enrollment (the rate is kept); ">96%" is a band (band_midpoints: 98.0).
    Lines that start like a school but cannot be read are returned in `unparsed`.
    """
    rows, unparsed = [], []
    for line in text.split("\n"):
        if not line.strip():
            continue
        match = SC_LINE.match(line)
        county = match.group("county") if match else None
        if not match:
            match = SC_LINE_CUT.match(line)
            county = full_sc_county(match.group("county")) if match else None
            if county is None:
                if SC_SCHOOL_START.match(line):
                    unparsed.append(line.strip())
                continue
        parts = re.split(r"\s{2,}", match.group("rest").strip())
        name, city, grades = (
            (parts + [None, None, None])[:3]
            if len(parts) >= 3
            else (parts[0], None, parts[-1] if len(parts) == 2 else None)
        )
        rows.append(
            {
                "region": match.group("region").replace(" ", ""),
                "county_label": county,
                "school_type": school_type(match.group("type")),
                "school_name": name.strip(),
                "city": city,
                "grade_range": grades,
                "students_text": match.group("students")
                .replace(" ", "")
                .replace(",", ""),
                "rate_text": match.group("rate").replace(" ", ""),
            }
        )
    df = pl.DataFrame(rows)
    rates, bands = parse_rates(df["rate_text"].to_list())
    middle = band_midpoints([b for b in bands if b])
    df = df.with_columns(
        enrollment=pl.col("students_text").map_elements(
            parse_number, return_dtype=pl.Float64
        ),
        rate=pl.Series(
            [r if b is None else middle[b] for r, b in zip(rates, bands)],
            dtype=pl.Float64,
        ),
        band=pl.Series(bands, dtype=pl.String),
        rate_source=pl.lit("overall"),
        includes_k=pl.Series(
            [
                includes_kindergarten(g) if g else None
                for g in df["grade_range"].to_list()
            ],
            dtype=pl.Boolean,
        ),
    )
    censored = df.filter(pl.col("students_text").str.starts_with("<"))
    if log is not None:
        un = pl.DataFrame(
            {"school_name": unparsed, "year_start": [None] * len(unparsed)},
            schema={"school_name": pl.String, "year_start": pl.Int64},
        )
        log_rows(
            log,
            un,
            "SC",
            "south carolina rows",
            "dropped",
            "line could not be read",
        )
        log_rows(
            log,
            censored.with_columns(year_start=pl.lit(None, dtype=pl.Int64)),
            "SC",
            "south carolina values",
            "changed",
            "enrollment censored (<10): missing",
        )
        log_rows(
            log,
            df.filter(pl.col("band").is_not_null()).with_columns(
                year_start=pl.lit(None, dtype=pl.Int64)
            ),
            "SC",
            "south carolina values",
            "changed",
            "banded rate replaced by the middle of its interval",
        )
    return df, unparsed


# --- files ----------------------------------------------------------------------


# --- Ohio ---------------------------------------------------------------------


def _cell(value):
    """A cell as clean text, or None."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def parse_ohio(frame, year_start, log=None):
    """
    One Ohio sheet (headerless frame; Ohio publishes one Excel file a year and
    the layout moves). The header row is the one with a "School Name" cell in
    the first 12 rows. Columns are found by their header text: the school name,
    "City", the "County" (not a county ID), the school IRN (a row whose IRN
    text starts with "Ohio" is a statewide total and is dropped), the number
    of kindergarten students enrolled ("...Enrolled..." with "Total" in the
    header, or "PupilsTotal"), and the MMR column:
        a count of students up to date or meeting the MMR requirement: the rate
        is count / enrollment x 100 (a count above the enrollment gives a rate
        above 100, which the assembler makes missing);
        a "%" column (a fraction in 2019-20): scale_factor decides;
        2015-16 has no MMR column: "PupilsWithReqImmu" / "PupilsTotal", all
        required vaccines (rate_source "overall"), and the sheet has every
        grade, so only rows with Type "Kindergarten" are kept.
    A cell that holds a text instead of a number ("All data that represents
    less than 10 students..." is masked) is a missing value; the school stays.
    Returns (frame, info): school_name, county_label, city, enrollment, rate,
    rate_source, year_start, and the layout found.
    """
    header_at = next(
        (
            i
            for i in range(min(12, frame.height))
            if any(
                re.fullmatch(r"(?i)school ?name", _cell(v) or "")
                for v in frame.row(i)
            )
        ),
        None,
    )
    if header_at is None:
        raise ValueError(
            "no header row with a School Name cell in the first 12 rows"
        )
    header = [_cell(v) or "" for v in frame.row(header_at)]
    lower = [h.lower() for h in header]

    def column(test, what):
        found = [j for j, h in enumerate(lower) if test(h)]
        if not found:
            raise ValueError(f"no {what} column in {header}")
        return found[0]

    name_c = column(
        lambda h: re.fullmatch(r"school ?name", h) is not None, "school name"
    )
    city_c = column(lambda h: h == "city", "city")
    county_c = column(
        lambda h: h.startswith("county") and "id" not in h, "county"
    )
    irn_c = column(
        lambda h: "irn" in h or h.startswith("school information retrieval"),
        "IRN",
    )
    type_c = next((j for j, h in enumerate(lower) if h == "type"), None)
    enroll_c = column(
        lambda h: ("enrolled" in h and "total" in h) or h == "pupilstotal",
        "enrollment",
    )
    if type_c is not None:
        layout, rate_c = (
            "overall",
            column(lambda h: h == "pupilswithreqimmu", "PupilsWithReqImmu"),
        )
    else:
        rate_c = column(
            lambda h: "mmr" in h
            and ("up-to-date" in h or "met the measles" in h)
            and "exemption" not in h,
            "MMR",
        )
        layout = "percent" if header[rate_c].startswith("%") else "count"
    kept, totals, other_grades = [], [], 0
    for row in frame[header_at + 1 :].iter_rows():
        if type_c is not None and _cell(row[type_c]) != "Kindergarten":
            other_grades += 1
            continue
        name = _cell(row[name_c])
        if name is None or (_cell(row[irn_c]) or "").lower().startswith(
            "ohio"
        ):
            totals.append(name or "")
            continue
        kept.append(row)
    enrollment = [parse_number(_cell(r[enroll_c])) for r in kept]
    given = [parse_number(_cell(r[rate_c])) for r in kept]
    if layout == "percent":
        factor = scale_factor(given)
        rate = [None if g is None else g * factor for g in given]
    else:
        rate = [
            None if g is None or not e else g / e * 100
            for g, e in zip(given, enrollment)
        ]
    out = pl.DataFrame(
        {
            "school_name": [_cell(r[name_c]) for r in kept],
            "county_label": [_cell(r[county_c]) for r in kept],
            "city": [_cell(r[city_c]) for r in kept],
            "enrollment": enrollment,
            "rate": rate,
            "rate_source": ["overall" if layout == "overall" else "mmr"]
            * len(kept),
        },
        schema_overrides={"enrollment": pl.Float64, "rate": pl.Float64},
    ).with_columns(year_start=pl.lit(year_start, dtype=pl.Int64))
    if log is not None:
        log_rows(
            log,
            pl.DataFrame(
                {
                    "school_name": totals,
                    "year_start": [year_start] * len(totals),
                }
            ),
            "OH",
            "ohio rows",
            "dropped",
            "statewide total or nameless row",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "OH",
            "ohio values",
            "changed",
            "rate masked or enrollment missing: missing",
        )
    info = {
        "year_start": year_start,
        "rows": out.height,
        "layout": layout,
        "dropped_totals": len(totals),
        "other_grade_rows": other_grades,
        "rate_missing": int(out["rate"].null_count()),
    }
    return out, info


# --- Iowa -----------------------------------------------------------------------

IA_NUM = r"\d[\d,]*|S"
IA_PCT = r"\d+(?:\.\d+)?|S"
IA_COUNTY = r"[A-Za-z][A-Za-z'.& ]*?"
IA_FULL = re.compile(
    rf"^(?P<county>{IA_COUNTY})\s+(?P<district>\d+)\s+(?:(?P<name>\S.*?)\s+)?(?P<cert>{IA_NUM})\s+(?P<total>{IA_NUM})\s+(?P<pct>{IA_PCT})\s*$"
)
IA_HEAD = re.compile(
    rf"^(?P<county>{IA_COUNTY})\s+(?P<district>\d+)\s+(?P<name>\S.*?)\s*$"
)  # layout C, first line
IA_NUMS = re.compile(
    rf"^\s+(?P<cert>{IA_NUM})\s+(?P<total>{IA_NUM})\s+(?P<pct>{IA_PCT})\s*$"
)  # layout C, second line
IA_TITLE = re.compile(r"(?i)immunization audit (\d{4})-(\d{2}) school year")
IA_FOOTER = re.compile(r"(?i)page \d+ of \d+|^\s*[a-z]+ \d{1,2}, \d{4}\s*$")
IA_HEADER_END = re.compile(r"^\s*Law\s*\*\s*$")
IA_DISTRICT_NAME = re.compile(r"(?i)school district$")
IA_TOTAL_NAME = re.compile(
    r"(?i)^(total\b|statewide\b|state total|iowa total|state of iowa)"
)
IA_NAME_INDENT = 8  # a name fragment is indented at least this much; footnotes start at column 0


def _ia_core(m):
    """A data row from the groups of a regex match (a dict or a match)."""
    m = m if isinstance(m, dict) else m.groupdict()
    return {
        "county": m["county"].strip(),
        "district": m["district"],
        "name": m.get("name") or "",
        "cert": m["cert"],
        "total": m["total"],
        "pct": m["pct"],
        "head": [],
        "tail": [],
    }


def _ia_page_items(page):
    """The lines of one page body as [("core", dict) | ("text", str)], and its footnote lines."""
    lines = page.split("\n")
    ends = [i for i, line in enumerate(lines) if IA_HEADER_END.match(line)]
    if len(ends) != 1:
        raise ValueError(
            f"expected one header end ('Law*') per page, found {len(ends)}"
        )
    items, footnotes = [], []
    for line in lines[ends[0] + 1 :]:
        if not line.strip() or IA_FOOTER.search(line):
            continue
        if (m := IA_FULL.match(line)) is not None:
            items.append(("core", _ia_core(m)))
        elif (h := IA_HEAD.match(line)) is not None:
            items.append(("head", h))
        elif (n := IA_NUMS.match(line)) is not None:
            items.append(("nums", n))
        elif len(line) - len(line.lstrip()) >= IA_NAME_INDENT:
            items.append(("text", line.strip()))
        else:
            footnotes.append(line.strip())
    # layout C: a head line followed by a numbers line is one row (name on the head line)
    merged, i = [], 0
    while i < len(items):
        kind, value = items[i]
        if kind == "head":
            if i + 1 >= len(items) or items[i + 1][0] != "nums":
                raise ValueError(
                    f"line with county and name but no numbers: {value['name']!r}"
                )
            merged.append(
                (
                    "core",
                    _ia_core(
                        {**value.groupdict(), **items[i + 1][1].groupdict()}
                    ),
                )
            )
            i += 2
        elif kind == "nums":
            raise ValueError("numbers without a county, district and name")
        else:
            merged.append((kind, value))
            i += 1
    return merged, footnotes


def _ia_attach_text(items):
    """
    Puts the text lines of a page into the head (above) or the tail (below) of the data
    line they belong to. For the lines between the data lines X and Y ("open" = the
    data line has no name text of its own):
        X open and Y closed or the page ends: the lines are the tail of X
        X closed (or the page starts): the lines are the head of Y (2021-22: a long name
            has its first line above the data line and its end on it; 2024-25: a
            nameless data line has its first line above)
        X and Y both open: a wrapped name has two lines, so X takes as many as it still
            lacks to reach two (a head of one line -> one more) and Y the rest
    """
    cores = [(i, v) for i, (k, v) in enumerate(items) if k == "core"]
    if not cores and any(k == "text" for k, _ in items):
        raise ValueError("name lines on a page without any data line")
    bounds = [(-1, None)] + cores + [(len(items), None)]
    for (i0, x), (i1, y) in zip(bounds[:-1], bounds[1:]):
        text = [v for k, v in items[i0 + 1 : i1] if k == "text"]
        if not text:
            continue
        x_open = x is not None and not x["name"]
        y_open = y is not None and not y["name"]
        if x_open and y_open:
            n = min(len(text), max(0, 2 - len(x["head"])))
            x["tail"], y["head"] = text[:n], text[n:]
        elif x_open:
            x["tail"] = text
        elif y is not None:
            y["head"] = text
        else:
            raise ValueError(f"name line(s) that belong to no row: {text}")
    return [v for _, v in cores]


def _ia_join(parts):
    """The parts of a wrapped name as one name ("Elementary-" + "Centerville" has no space)."""
    out = ""
    for part in parts:
        out = (
            part
            if not out
            else (out + part if out.endswith("-") else f"{out} {part}")
        )
    return out


def parse_iowa(text, year_start=None, log=None):
    """
        One Iowa audit PDF as the text of `pdftotext -layout`.

    Iowa school-level immunization audit ("Kindergarten - 12th Grade Summary Report by
    School", Iowa HHS Immunization Program), one PDF per school year (2021-22 to 2025-26),
    read as the text of `pdftotext -layout` (the parser takes that text, not a file).

    WHAT THE FILES ARE (read this before using the numbers)
        * NOT kindergarten MMR. A row is one school, all grades it serves (K-12): "Total
          Enrollment" is the school's all-grades enrollment, not the kindergarten class.
          There is no breakdown by grade and no MMR column.
        * The rate is "Percent Vaccinated with Vaccines Required by Law" = valid certificates
          of immunization / total enrollment x 100, all required vaccines together. So
          rate_source is "overall", never "mmr", and a school's rate is lower than its
          MMR rate would be.
        * The unit is the school, except about 40 rows a year named "... School District"
          ("Mid-Prairie Comm School District", enrollment 2 to 650): these are entries of
          the district itself, not of a building (students not assigned to a school; the file
          does not say). unit = "district" for them; they are kept, and the lead decides
          whether to use them (their rates, when not suppressed, are often low: 45-85).
        * A school with a total enrollment of 100 or less is suppressed: certificate count
          and rate are "S" (the footnote says "less than 100" in 2021-22 and 2022-23 and
          "100 or less" later; the enrollment is still printed).

    Layout of the text (the same in all years up to widths and wrapped names)
        County | District number | School name | Certificates | Total enrollment | Percent
        Each page starts with a header block that ends in a line "Law*" and ends with a
        footer "<date> ... Page n of N"; the last page has three footnote lines. Rows:
          A  one line: county, district number, name, three numbers (2021-22 to 2023-24
             almost always; a name too long for the column wraps).
          B  the name wrapped over several lines: the data line has NO name, the first part
             of the name is on the line(s) above it and the rest below (2024-25 and
             2025-26: about a third of the rows; a few in the earlier years).
          C  (a handful of rows) county, district number and the name on one line and the
             three numbers alone on the next line.
          A2 (2021-22, a few rows): the name wrapped, its first line above the data line and
             the end of it ("School") on the data line.
        A text-only line (indented to the name column) is attached by `_attach_text`: it is
        the head of the next data line, except below a data line that has no name of its own
        (then it is its tail); between two nameless data lines each wrapped name is taken to
        have two lines (a ONE-line gap between two nameless rows is therefore the tail of
        the first). The one such row of the real files (2025-26 Albert City-Truesdale
        Elementary School, Buena Vista) is read that way, and by the text it is right. This
        rests on the order of the lines only; a name can be wrong, the numbers cannot.

    Rules of the parser
        * year_start: the first year of "IMMUNIZATION AUDIT 2022-23 SCHOOL YEAR" in the
          title of page 1 (2022-23 -> 2022). If `year_start` is given and differs from the
          title it is an error (a mismatch means the wrong file was passed).
        * Numbers may have thousands commas ("1,115"). A rate is read as printed (percent,
          0-100, a few are printed as integers that differ by up to 3 points from
          certificates/enrollment; kept as printed). The scale check is scale_factor (these
          files are in percent, so it is 1).
        * "S" in the rate -> rate missing (row kept, log row). "S" in the total ->
          enrollment missing (row kept; one row of 2025-26 has certificates S, total S and
          a rate of 50.0: the rate is kept).
        * No band text ("≥95%") exists in the files; the band column is filled by the same
          band rule as the other readers if one ever appears (it is empty here).
        * Total/statewide rows do not exist in these files; a row whose name starts with
          "total", "statewide", "state total" or "iowa total" is dropped and logged, in case one appears.
        * The county is the county name printed (Iowa spells "Pottawattami" once: COUNTY_OVERRIDES maps it).
        * city is not in the files.
    Returns (frame, info): school_name, county_label, enrollment, rate, band, rate_source,
    unit, year_start, plus a dict with the counts found.
    """
    text = html.unescape(text)  # a text made with -bbox shows O&apos;Brien
    title = IA_TITLE.search(text)
    if title is None:
        raise ValueError("no 'IMMUNIZATION AUDIT yyyy-yy SCHOOL YEAR' title")
    found = int(title.group(1))
    if year_start is not None and year_start != found:
        raise ValueError(
            f"year_start {year_start} but the title says {found}-{title.group(2)}"
        )
    rows, footnotes = [], []
    for page in text.split("\f"):
        if not page.strip():
            continue
        items, notes = _ia_page_items(page)
        footnotes += notes
        rows += _ia_attach_text(items)
    names = [
        _ia_join(r["head"] + ([r["name"]] if r["name"] else []) + r["tail"])
        for r in rows
    ]
    pct_given = [None if r["pct"] == "S" else float(r["pct"]) for r in rows]
    rates, bands = parse_rates([r["pct"] for r in rows])
    assert rates == pct_given and not any(bands)
    factor = scale_factor(rates)
    out = pl.DataFrame(
        {
            "school_name": names,
            "county_label": [r["county"] for r in rows],
            "enrollment": [
                None
                if r["total"] == "S"
                else float(r["total"].replace(",", ""))
                for r in rows
            ],
            "rate": [None if g is None else g * factor for g in rates],
            "band": bands,
            "rate_source": ["overall"] * len(rows),
            "unit": [
                "district" if IA_DISTRICT_NAME.search(n) else "school"
                for n in names
            ],
        },
        schema_overrides={
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "band": pl.String,
        },
    ).with_columns(year_start=pl.lit(found, dtype=pl.Int64))
    totals = out.filter(
        pl.col("school_name").str.contains(IA_TOTAL_NAME.pattern)
    )
    out = out.filter(
        ~pl.col("school_name").str.contains(IA_TOTAL_NAME.pattern)
    )
    suppressed = out.filter(pl.col("rate").is_null())
    no_enrollment = out.filter(pl.col("enrollment").is_null())
    if log is not None:
        log_rows(
            log, totals, "IA", "iowa rows", "dropped", "total or statewide row"
        )
        log_rows(
            log,
            suppressed,
            "IA",
            "iowa values",
            "changed",
            "rate suppressed (S, enrollment of 100 or less): missing",
        )
        log_rows(
            log,
            no_enrollment,
            "IA",
            "iowa values",
            "changed",
            "total enrollment printed as S: missing",
        )
    info = {
        "year_start": found,
        "rows": out.height,
        "dropped_totals": totals.height,
        "scale": "fraction x100" if factor == 100 else "percent",
        "rate_missing": suppressed.height,
        "enrollment_missing": no_enrollment.height,
        "district_rows": out.filter(pl.col("unit") == "district").height,
        "wrapped_names": sum(1 for r in rows if r["head"] or r["tail"]),
        "footnotes": footnotes,
    }
    return out, info


# --- Maine ----------------------------------------------------------------------
# Maine school-level kindergarten MMR (Maine CDC "School Vaccination Rates",
# 2014-15 to 2024-25), in the layout of school_state_files.py.
#
# Two kinds of file, one parser each; both take a table that is already read, so
# that they are tested without files, and both return (frame, info):
#     parse_me_excel(frame, year_start, log=None): an Excel sheet "Sheet1" (2018-19
#         to 2024-25) as a headerless frame (ssf.read_sheet);
#     parse_me_pdf(bbox, year_start, log=None): the output of `pdftotext -bbox` of a
#         PDF (2014-15 to 2017-18; the table is only in the PDF).
#
# Frame columns: school_name, county_label, enrollment (float), rate (percent
# 0-100), rate_source, unit, year_start. `city` is not in any Maine file: it is
# left out.
#
# What every file looks like (one table; 3 blocks of columns side by side:
# Kindergarten, First Grade (PDF 2014-2017 only) or none, Seventh Grade, Twelfth
# Grade in the Excel files). A row is a school or a reporting unit with the
# grades it reports; a school with no kindergarten has an empty or "-"
# kindergarten block. Only the first block (Kindergarten) is read.
#
# Assumptions (each is checked in the code unless it says "taken"):
#     * The rate is the "2MMR" column of the kindergarten block: the share of
#       the students assessed with two doses of MMR. rate_source is "mmr".
#     * The enrollment is the "Num Assessed" / "Assessed" column of the
#       kindergarten block: the kindergartners assessed (the school's
#       kindergarten enrollment as reported to the survey).
#     * Rates in the Excel files are fractions (0.906), in the PDFs percents with
#       a "%" sign ("90.6%"). Excel: ssf.scale_factor decides (median at most 1.5
#       is a fraction). PDF: the "%" sign says so. A rate above 100 after scaling
#       is an error (ValueError).
#     * Rows without kindergarten students ("-", blank, or 0 assessed: middle and
#       high schools, schools with grades 6-12) are not kindergarten rows: dropped
#       and logged. A kindergarten row with a rate cell that is blank or text is
#       kept with a missing rate; a kindergarten assessed cell of text such as
#       "<10" or "*" is a censored count: missing enrollment, rate kept. (No such
#       cell exists in the files today; the code is there for a later file.)
#     * Total rows ("Statewide", "Statewide Totals", "<County> County Totals",
#       "State") are dropped and logged. The printed statewide total of
#       kindergartners assessed is checked against the rows: the sum of the
#       enrollments of ALL rows read (rows with kindergartners) must equal it,
#       or ValueError. This holds for every PDF year and for every Excel year
#       (so no school is lost, none is counted twice, and the reporting units
#       are NOT double counted with their schools). The county totals rows
#       (Excel) are checked the same way against the rows of their county.
#       The check is skipped, and said so in info["total_check"], only when an
#       enrollment is censored (the sum is then unknown).
#     * County: the county of the file's own "County" column (Excel) or of the
#       county word after the school name in the PDF row. Every row of every
#       file has one; no municipality lookup is needed (the earlier scratch work
#       did not use one either). The label is the county name; ssf.attach_counties
#       raises for a name that is not a county of Maine.
#     * A row can be a reporting unit instead of one school ("MSAD #20", "RSU 25",
#       "Auburn School Department", "Union Elementary"). The file does not say
#       so; `unit` is "district" when the name is nothing but a district label
#       (`district_unit`), else "school". It is a name heuristic, taken as it is.
#       The row is kept either way (its students are in the printed totals).
#     * The year of the file is the year_start given; the title of the file
#       ("... 2017-2018") is checked against it when the title is in the table.
#     * PDF layout, from the coordinates of `pdftotext -bbox`: words of one row
#       share a yMin (within 2 points); the school name and the county word
#       (the last word, one of the 16 counties of Maine; a long name can run
#       into it: "SchoolAroostook") are left of the "Num" header, the
#       kindergarten numbers right of it and left of the next block (the second
#       "Num Assessed"); a number belongs to the column of the
#       header word whose right edge is nearest (numbers are right aligned); two
#       numbers in the same column of a row is an error. A row of the table that
#       has no county word is a total row if it is "State"/"Statewide" or contains
#       "Total", else a title or footnote row (dropped, logged). A school whose
#       county word is misread would be lost as such a row; the printed-total
#       check then fails.

ME_STATE = "ME"
ME_COUNTIES = [
    "Androscoggin", "Aroostook", "Cumberland", "Franklin", "Hancock", "Kennebec", "Knox", "Lincoln",
    "Oxford", "Penobscot", "Piscataquis", "Sagadahoc", "Somerset", "Waldo", "Washington", "York",
]  # fmt: skip

ME_COUNTY_END = re.compile(r"^(.*?)\s*(" + "|".join(ME_COUNTIES) + r")$")
ME_TOTALS_WORD = re.compile(r"(?i)^[^•*·]*\btotals?$")
ME_COUNTY_TOTAL_NAME = re.compile(r"(?i)^(.*?)\s+county\s+totals?$")
ME_STATEWIDE_NAME = re.compile(r"(?i)^state(wide)?(\s+totals?)?$")
ME_TITLE_YEAR = re.compile(r"(20\d\d)\s*-\s*(20)?(\d\d)")
# a name that is only a district label: "MSAD #20", "RSU 25", "MSAD/RSU 32", "AOS 91",
# "Auburn School Department", "Gorham School District", "Union 60"
ME_UNIT_ONLY = re.compile(
    r"(?ix)^(?:(?:m?sad|rsu|aos|csd|cssd|union|rsu/msad|msad/rsu)\s*\#?\s*\d+[a-z]?(?:\s*[/,&-]\s*(?:m?sad|rsu|aos|csd)?\s*\#?\s*\d+)*"
    r"|(?:[a-z.' ]+?\s+)?school\s+(?:department|district)(?:\s+of\s+[a-z ]+)?)$"
)


ME_DISTRICT_SCHOOLS = re.compile(
    r"(?i)\bdistrict schools\b"
)  # "Van Buren District Schools" (not "Ashland District School")


def me_district_unit(name):
    """'district' when the name is nothing but a district label (see ME_UNIT_ONLY) or "... District Schools", else 'school'."""
    name = re.sub(r"\s+", " ", name).strip()
    return (
        "district"
        if ME_UNIT_ONLY.match(name) or ME_DISTRICT_SCHOOLS.search(name)
        else "school"
    )


def _me_text(value):
    """A cell as clean text, or None."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def _me_kind_of_count(text):
    """'none' (blank, '-': no kindergarten), 'censored' (text such as '<10', '*') or 'number'."""
    if text is None or text in ("-", "–", "—"):
        return "none"
    return (
        "number"
        if parse_number(text.replace(",", "")) is not None
        else "censored"
    )


def me_is_total(name):
    """A total row: "Statewide", "Statewide Totals", "State", "York County Totals"."""
    return bool(ME_STATEWIDE_NAME.match(name) or ME_TOTALS_WORD.search(name))


def _me_check_title_year(texts, year_start, where):
    """The first title text with a school year ("... Results 2017-2018") must name the year given; returns it, or None."""
    for text in texts:
        match = ME_TITLE_YEAR.search(text or "")
        if match:
            if int(match.group(1)) != year_start:
                raise ValueError(
                    f"{where}: the title {text!r} is not the school year starting {year_start}"
                )
            return int(match.group(1))
    return None


def _me_log(log, rows, step, action, reason, year_start):
    """Log a list of names (rows is a list of school names)."""
    if log is None or not rows:
        return
    log_rows(
        log,
        pl.DataFrame(
            {"school_name": rows, "year_start": [year_start] * len(rows)},
            schema={"school_name": pl.String, "year_start": pl.Int64},
        ),
        ME_STATE,
        step,
        action,
        reason,
    )


def _me_frame(rows, year_start):
    """The output frame from a list of dicts (school_name, county_label, enrollment, rate)."""
    return pl.DataFrame(
        {
            "school_name": [r["school_name"] for r in rows],
            "county_label": [r["county_label"] for r in rows],
            "enrollment": [r["enrollment"] for r in rows],
            "rate": [r["rate"] for r in rows],
            "rate_source": ["mmr"] * len(rows),
            "unit": [me_district_unit(r["school_name"]) for r in rows],
        },
        schema={
            "school_name": pl.String,
            "county_label": pl.String,
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "rate_source": pl.String,
            "unit": pl.String,
        },
    ).with_columns(year_start=pl.lit(year_start, dtype=pl.Int64))


def _me_check_totals(rows, printed, county_totals, year_start, where):
    """
    The printed statewide total, and each county's printed total, against the sum
    of the rows read. Returns the text for info["total_check"]. Raises ValueError.
    """
    if any(r["censored"] for r in rows):
        return "skipped: a kindergarten count is censored"
    if printed is None:
        raise ValueError(
            f"{where} {year_start}: no printed statewide total row to check the sum of students against"
        )
    total = sum(r["enrollment"] for r in rows)
    if total != printed:
        raise ValueError(
            f"{where} {year_start}: the students assessed in the rows sum to {total:g}, the printed statewide total is {printed:g}"
        )
    for county, value in county_totals.items():
        own = sum(r["enrollment"] for r in rows if r["county_label"] == county)
        if own != value:
            raise ValueError(
                f"{where} {year_start}: {county} rows sum to {own:g}, the printed county total is {value:g}"
            )
    return f"ok: {total:g} = printed statewide total" + (
        f"; {len(county_totals)} county totals ok" if county_totals else ""
    )


# --- Excel (2018-19 on) -------------------------------------------------------------


def parse_me_excel(frame, year_start, log=None):
    """
    One Excel sheet "Sheet1" (headerless frame). Layout: a title in column 3 of
    row 1, a group row with "Kindergarten" (and then "Seventh Grade",
    "Twelfth Grade") above the header row, which starts with "School" then
    "County". The kindergarten block runs from the "Kindergarten" cell to the next group
    label; in it "Num Assessed" (or "Assessed") is the enrollment and the column
    whose header contains "MMR" ("2MMR") is the rate, a fraction (scale_factor).
    Returns (frame, info).
    """
    at = header_row(frame, "School")
    group_at = next(
        (
            i
            for i in range(at)
            if any(_me_text(v) == "Kindergarten" for v in frame.row(i))
        ),
        None,
    )
    if group_at is None:
        raise ValueError("no 'Kindergarten' group cell above the header row")
    group = [_me_text(v) for v in frame.row(group_at)]
    start = group.index("Kindergarten")
    end = next(
        (j for j in range(start + 1, len(group)) if group[j]), len(group)
    )
    header = [_me_text(v) or "" for v in frame.row(at)]
    assessed_c = [
        j
        for j in range(start, end)
        if header[j].lower() in ("num assessed", "assessed")
    ]
    mmr_c = [j for j in range(start, end) if "mmr" in header[j].lower()]
    if len(assessed_c) != 1 or len(mmr_c) != 1:
        raise ValueError(
            f"kindergarten block columns {start}..{end - 1} {header[start:end]}: need one Assessed and one MMR column"
        )
    assessed_c, mmr_c = assessed_c[0], mmr_c[0]
    if header[1].lower() != "county":
        raise ValueError(
            f"column 2 of the header row is {header[1]!r}, not County"
        )
    title_year = _me_check_title_year(
        [_me_text(v) for i in range(at) for v in frame.row(i)],
        year_start,
        "Excel",
    )

    kept, totals, nokinder, footnotes, blanks = [], [], [], [], 0
    printed, county_totals = None, {}
    for row in frame[at + 1 :].iter_rows():
        name, county = _me_text(row[0]), _me_text(row[1])
        if name is None:
            blanks += 1
            continue
        kind = _me_kind_of_count(_me_text(row[assessed_c]))
        if county is None and not me_is_total(
            name
        ):  # a footnote line ("• ...", "*** Data Source", "Regarding ...")
            footnotes.append(name)
            continue
        if me_is_total(name):
            totals.append(name)
            if kind == "number":
                value = parse_number(
                    _me_text(row[assessed_c]).replace(",", "")
                )
                if ME_STATEWIDE_NAME.match(name):
                    printed = value
                else:  # "York County Totals"; the county cell of the row is empty in the 2022-23 file
                    named = ME_COUNTY_TOTAL_NAME.match(name)
                    county_totals[
                        county or (named.group(1) if named else name)
                    ] = value
            continue
        count = (
            None
            if kind != "number"
            else parse_number(_me_text(row[assessed_c]).replace(",", ""))
        )
        if kind == "none" or count == 0:
            nokinder.append(name)
            continue
        kept.append(
            {
                "school_name": name,
                "county_label": county,
                "enrollment": count,
                "censored": kind == "censored",
                "given": parse_number(_me_text(row[mmr_c])),
            }
        )
    factor = scale_factor([r["given"] for r in kept])
    for r in kept:
        r["rate"] = None if r["given"] is None else r["given"] * factor
    if any(r["rate"] is not None and r["rate"] > 100.0 + 1e-6 for r in kept):
        raise ValueError(
            f"a rate above 100 in the {year_start} file: wrong scale (factor {factor})"
        )
    check = _me_check_totals(kept, printed, county_totals, year_start, "Excel")
    out = _me_frame(kept, year_start)
    _me_log(
        log,
        totals,
        "maine rows",
        "dropped",
        "statewide or county total row",
        year_start,
    )
    _me_log(
        log,
        nokinder,
        "maine rows",
        "dropped",
        "no kindergarten students (blank, '-' or 0 assessed)",
        year_start,
    )
    _me_log(
        log, footnotes, "maine rows", "dropped", "footnote line", year_start
    )
    if log is not None:
        log_rows(
            log,
            out.filter(pl.col("enrollment").is_null()),
            ME_STATE,
            "maine values",
            "changed",
            "kindergarten count censored: enrollment missing",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            ME_STATE,
            "maine values",
            "changed",
            "MMR rate blank or suppressed: missing",
        )
    info = {
        "year_start": year_start,
        "kind": "excel",
        "rows": out.height,
        "dropped_totals": len(totals),
        "dropped_no_kindergarten": len(nokinder),
        "dropped_footnotes": len(footnotes),
        "blank_rows": blanks,
        "scale": "fraction x100" if factor == 100 else "percent",
        "enrollment_censored": int(out["enrollment"].null_count()),
        "rate_missing": int(out["rate"].null_count()),
        "units_district": int((out["unit"] == "district").sum()),
        "total_check": check,
        "printed_total": printed,
        "title_year": title_year,
        "columns": f"assessed={header[assessed_c]!r} (col {assessed_c}), mmr={header[mmr_c]!r} (col {mmr_c})",
    }
    return out, info


# --- PDF (2014-15 to 2017-18) ---------------------------------------------------------

ME_WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="[\d.]+">(.*?)</word>'
)
ME_NUMBER = re.compile(r"^[\d,]+(?:\.\d+)?%?$")
ME_ROW_TOLERANCE = (
    2.0  # points of yMin within one row (rows are about 7 points apart)
)


def _me_pdf_words(bbox):
    """[[(y, x0, x1, text), ...] for each page] from the text of `pdftotext -bbox`."""
    pages = bbox.split("<page ")[1:]
    return [
        sorted(
            (float(y), float(x0), float(x1), html.unescape(w))
            for x0, y, x1, w in ME_WORD.findall(page)
        )
        for page in pages
    ]


def _me_pdf_rows(words):
    """Words of one page grouped in rows: [(y, [(x0, x1, text), ...])], each row sorted by x."""
    rows = []
    for y, x0, x1, text in words:
        if rows and y - rows[-1][0] <= ME_ROW_TOLERANCE:
            rows[-1][1].append((x0, x1, text))
            rows[-1][0] = y
        else:
            rows.append([y, [(x0, x1, text)]])
    return [(y, sorted(ws)) for y, ws in rows]


def parse_me_pdf(bbox, year_start, log=None):
    """
    One PDF, as the text of `pdftotext -bbox`. See the module docstring for the
    geometry. The header (on page 1) is the row of the word "Assessed": its
    words "School", "County" and "Num" are checked, the K numbers are right of
    the first "Num", and between the first and the second "Assessed" are the labels of
    the kindergarten block ("TOTExempt Religious Medical Philosophical ... 2MMR
    3Polio"). Rates are percents ("94.5%"). Returns (frame, info). The sum of
    the students assessed over the rows must equal the printed statewide total
    (the row "State" or "Statewide"), else ValueError.
    """
    pages = _me_pdf_words(bbox)
    if not pages or not pages[0]:
        raise ValueError("no words on page 1 of the PDF text")
    first = pages[0]
    head_y = min(y for y, _, _, w in first if w == "Assessed")
    header = sorted(
        [
            (x0, x1, w)
            for y, x0, x1, w in first
            if abs(y - head_y) <= ME_ROW_TOLERANCE
        ],
        key=lambda t: t[0],
    )
    assessed = [h for h in header if h[2] == "Assessed"]
    nums = [h for h in header if h[2] == "Num"]
    has_county = any(w == "County" for _, _, w in header)
    if not has_county or not nums or not assessed:
        raise ValueError(
            f"header row not found (County/Num/Assessed) in {[h[2] for h in header][:12]}"
        )
    num_x0 = nums[0][0]  # K numbers start right of this
    if len(assessed) >= 2:
        next_num_x0 = next(n[0] for n in nums if n[0] > assessed[0][1])
        labels = [
            h
            for h in header
            if assessed[0][0] <= h[0] < next_num_x0 and h[2] != "Num"
        ]
        bound = (
            (labels[-1][1] + assessed[1][1]) / 2
        )  # between the last kindergarten column and the next block's Num Assessed
    else:
        labels = [h for h in header if h[0] >= assessed[0][0]]
        bound = float("inf")
    label_names = [h[2] for h in labels]
    mmr_j = [j for j, n in enumerate(label_names) if "MMR" in n.upper()]
    if label_names[0] != "Assessed" or len(mmr_j) != 1:
        raise ValueError(
            f"kindergarten block labels {label_names}: need Assessed first and one MMR"
        )
    mmr_j = mmr_j[0]

    title_texts = [
        " ".join(w for _, _, w in ws)
        for y, ws in _me_pdf_rows(first)
        if y < head_y
    ]
    title_year = _me_check_title_year(title_texts, year_start, "PDF")

    kept, totals, nokinder, other, max_offset = [], [], [], [], 0.0
    printed, county_totals, no_percent = None, {}, 0
    for pi, words in enumerate(pages):
        for y, ws in _me_pdf_rows(words):
            if pi == 0 and y <= head_y + 3:  # the title and the header lines
                continue
            # name and county are one text (a long name runs into the county: "SchoolAroostook")
            left = _me_text(" ".join(w for x0, x1, w in ws if x0 < num_x0 - 2))
            found = ME_COUNTY_END.match(left or "")
            name, county = (
                (_me_text(found.group(1)), found.group(2))
                if found
                else (left, None)
            )
            k_words = [
                (x0, x1, w)
                for x0, x1, w in ws
                if x0 >= num_x0 - 2 and x1 <= bound and ME_NUMBER.match(w)
            ]
            if name is None:
                other.append(f"p{pi + 1} y{y:.0f}: no name")
                continue
            if county is None and not me_is_total(name):  # a footnote line
                other.append(name)
                continue
            # which column each number is in
            cells = {}
            for x0, x1, w in k_words:
                j = min(
                    range(len(labels)), key=lambda j: abs(labels[j][1] - x1)
                )
                max_offset = max(max_offset, abs(labels[j][1] - x1))
                if j in cells:
                    raise ValueError(
                        f"{name!r} ({year_start}): two numbers in column {label_names[j]}: {cells[j]!r}, {w!r}"
                    )
                cells[j] = w
            if 0 in cells and not re.fullmatch(r"[\d,]+", cells[0]):
                raise ValueError(
                    f"{name!r} ({year_start}): students assessed {cells[0]!r} is not a whole number"
                )
            if county is None:  # a total row
                totals.append(name)
                if ME_STATEWIDE_NAME.match(name) and 0 in cells:
                    printed = parse_number(cells[0].replace(",", ""))
                continue
            if 0 not in cells or parse_number(cells[0].replace(",", "")) in (
                None,
                0,
            ):
                nokinder.append(name)
                continue
            rate_text = cells.get(mmr_j)
            if rate_text is not None and not rate_text.endswith("%"):
                no_percent += 1
            given = None if rate_text is None else parse_number(rate_text)
            kept.append(
                {
                    "school_name": name,
                    "county_label": county,
                    "enrollment": parse_number(cells[0].replace(",", "")),
                    "censored": False,
                    "rate": given,
                }
            )
    if any(r["rate"] is not None and r["rate"] > 100.0 + 1e-6 for r in kept):
        raise ValueError(f"a rate above 100 in the {year_start} PDF")
    check = _me_check_totals(kept, printed, county_totals, year_start, "PDF")
    out = _me_frame(kept, year_start)
    _me_log(
        log,
        totals,
        "maine rows",
        "dropped",
        "statewide or county total row",
        year_start,
    )
    _me_log(
        log,
        nokinder,
        "maine rows",
        "dropped",
        "no kindergarten students (blank or 0 assessed)",
        year_start,
    )
    _me_log(
        log,
        other,
        "maine rows",
        "dropped",
        "footnote or unreadable line",
        year_start,
    )
    if log is not None:
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            ME_STATE,
            "maine values",
            "changed",
            "MMR rate blank: missing",
        )
    info = {
        "year_start": year_start,
        "kind": "pdf",
        "rows": out.height,
        "dropped_totals": len(totals),
        "dropped_no_kindergarten": len(nokinder),
        "dropped_other_lines": len(other),
        "other_lines": other[:8],
        "scale": "percent (% sign)",
        "rate_without_percent_sign": no_percent,
        "enrollment_censored": 0,
        "rate_missing": int(out["rate"].null_count()),
        "units_district": int((out["unit"] == "district").sum()),
        "total_check": check,
        "printed_total": printed,
        "title_year": title_year,
        "columns": f"labels={label_names}, mmr={label_names[mmr_j]!r}, max column offset {max_offset:.1f} pt",
    }
    return out, info


# --- Wisconsin ---------------------------------------------------------------------
# Wisconsin Department of Health Services (P-01892): the "By School" sheet of p01892.xlsx.

WI_TOTAL_NAME = re.compile(
    r"(?i)^\s*(total\b|statewide\b|state total\b|state of wisconsin\b|wisconsin( state)? total\b|wisconsin\s*$|all schools\b)"
)
WI_NA = {"", "n/a", "na", "nan"}


def _wi_text(value):
    """A cell as clean text, or None."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def _wi_key(name):
    """A header as lower-case text without footnote stars ("...Requirements**")."""
    return re.sub(r"\s+", " ", str(name).replace("*", "")).strip().lower()


def _wi_band_text(text):
    """A band cell (">95", "<5") as ">95%" / "<5%": the form band_midpoints reads; None for any other cell."""
    match = BAND.match(text)
    return f"{match.group(1)}{float(match.group(2)):g}%" if match else None


def parse_wisconsin(frame, year_start=None, log=None):
    """
    The "By School" sheet of the Wisconsin file (headerless frame): one row per
    school and school year (2018-19 on), all grades and "Kindergarten:" columns.
    Returns (frame, info): school_name, county_label, city, enrollment, rate,
    band, rate_source, year_start, unit. `year_start` keeps one school year
    (None: all years of the sheet).

    Assumptions
        * The header row is the one that starts with "County" (first 15 rows).
          Columns are found by their header text, footnote stars ignored; only
          the "Kindergarten:" columns are used, never the all-grades ones.
        * The school year is the first four digits of "School Year" ("2024– 2025"
          -> 2024, the first year of the school year).
        * Rate: "Kindergarten: % Met Minimum MMR Requirements" (two doses of MMR)
          in a school year in which that column has any value (2024-25 on):
          rate_source "mmr". Before that the column is "N/A" for every school
          and the rate is "Kindergarten: % Met Minimum Requirements", students
          who met ALL the minimum requirements of the year (DTaP, polio, hepatitis
          B, MMR, varicella), not MMR: rate_source "overall". The choice is per
          school year, never per row (a row of an MMR year whose MMR cell is "N/A"
          has a missing rate, even if its all-vaccine cell is given).
        * The values are integer percents of the kindergartners of the school (not
          of the school: "% ... in all grades" is another column). A file in
          fractions (median at most 1.5 in a year) would be multiplied by 100
          (scale_factor); the real file is in percent.
        * ">95" is a band (more than 95 percent; the highest the file prints, 95
          is the highest plain number): the text goes to `band` (">95%") and the
          middle of its interval to `rate` (band_midpoints: 97.5 with ">95%"
          alone). Only ">" bands get a midpoint.
        * "<5" (13 cells, all in the all-vaccine column, 2018-2023) is NOT given
          a value: it would read as "fewer than 5 percent of the kindergartners
          met all requirements", but the same schools are mostly at 40-95 in
          other years (median 77) and in 2024-25 rows an all-vaccine "<5" sits
          next to MMR values of 75-94. It is treated as a suppressed cell:
          missing rate, band missing, row kept, logged (info["bottom_band_missing"]).
        * "N/A" (cells of fewer than 10 kindergartners are suppressed with "N/A"
          according to the file's notes) is a missing rate; the row is kept.
        * A row with "N/A" in BOTH kindergarten rate columns has no kindergarten
          data (a middle or high school, a school with no report, or fewer than 10
          students): it is not a kindergarten row and is dropped, logged by the
          file's Comment ("Report received", "No report received", "Report
          received, <10 students"). No such row has another kindergarten column
          filled in the real file.
        * Total rows (a school name such as "Total", "Statewide") are dropped
          and logged; the real sheet has none.
        * No enrollment is in the sheet: enrollment is missing for every row
          (never taken from the district tab). unit is "school".
        * Rows with the same county, city, name and year are kept: schools share
          names (and the file says several locations may share one code).
    """
    at = header_row(frame, "County", limit=15)
    table = with_header(frame, at)
    by_key = {_wi_key(c): c for c in table.columns}

    def col(key, required=True):
        if key in by_key:
            return table[by_key[key]].to_list()
        if required:
            raise ValueError(f"no column {key!r} in {list(table.columns)}")
        return [None] * table.height

    names, counties, cities = (
        col("school name"),
        col("county"),
        col("city", False),
    )
    comments, years = col("comment", False), col("school year")
    overall, mmr = (
        col("kindergarten: % met minimum requirements"),
        col("kindergarten: % met minimum mmr requirements"),
    )

    rows = []  # one dict per row that has a school name and a school year
    nameless = 0
    for i in range(table.height):
        name, year_text = _wi_text(names[i]), _wi_text(years[i])
        if (
            name is None
            and year_text is None
            and _wi_text(counties[i]) is None
        ):
            nameless += 1  # an empty row at the end of the sheet
            continue
        match = re.match(r"\s*(\d{4})", year_text or "")
        if name is None or match is None:
            raise ValueError(
                f"row {at + 2 + i}: school name {name!r}, school year {year_text!r}"
            )
        rows.append(
            {
                "school_name": name,
                "county_label": _wi_text(counties[i]),
                "city": _wi_text(cities[i]),
                "year_start": int(match.group(1)),
                "comment": _wi_text(comments[i]) or "no Comment",
                "overall": _wi_text(overall[i]),
                "mmr": _wi_text(mmr[i]),
            }
        )
    other_years = 0
    if year_start is not None:
        other_years = sum(r["year_start"] != year_start for r in rows)
        rows = [r for r in rows if r["year_start"] == year_start]

    def logged(selected, action, reason):
        if log is not None:
            frame_ = pl.DataFrame(
                {
                    "school_name": [r["school_name"] for r in selected],
                    "year_start": [r["year_start"] for r in selected],
                },
                schema={"school_name": pl.String, "year_start": pl.Int64},
            )
            log_rows(
                log,
                frame_,
                "WI",
                "wisconsin rows"
                if action == "dropped"
                else "wisconsin values",
                action,
                reason,
            )

    is_na = lambda t: t is None or t.lower() in WI_NA  # noqa: E731
    totals = [r for r in rows if WI_TOTAL_NAME.match(r["school_name"])]
    logged(totals, "dropped", "total or statewide row")
    rows = [r for r in rows if not WI_TOTAL_NAME.match(r["school_name"])]
    empty = [r for r in rows if is_na(r["overall"]) and is_na(r["mmr"])]
    for comment in sorted({r["comment"] for r in empty}):
        logged(
            [r for r in empty if r["comment"] == comment],
            "dropped",
            f"no kindergarten data (both kindergarten rate columns N/A); Comment: {comment}",
        )
    rows = [r for r in rows if not (is_na(r["overall"]) and is_na(r["mmr"]))]

    # the rate column of each school year: MMR if the year has any MMR value
    mmr_years = {r["year_start"] for r in rows if not is_na(r["mmr"])}
    for r in rows:
        r["rate_source"] = "mmr" if r["year_start"] in mmr_years else "overall"
        r["raw"] = r["mmr"] if r["rate_source"] == "mmr" else r["overall"]
        r["raw"] = None if is_na(r["raw"]) else r["raw"]
    scale = {}
    rates, bands = parse_rates([r["raw"] for r in rows])
    for r, rate, band in zip(rows, rates, bands):
        text = None if band is None else _wi_band_text(band)
        r["bottom"] = text is not None and text[0] in "<≤"
        r["number"], r["band"] = rate, None if r["bottom"] else text
    for year in sorted({r["year_start"] for r in rows}):
        scale[year] = scale_factor(
            [r["number"] for r in rows if r["year_start"] == year]
        )
    middle = band_midpoints([r["band"] for r in rows if r["band"]])
    for r in rows:
        r["rate"] = (
            r["number"] * scale[r["year_start"]]
            if r["number"] is not None
            else middle.get(r["band"])
        )

    out = pl.DataFrame(
        {
            "school_name": [r["school_name"] for r in rows],
            "county_label": [r["county_label"] for r in rows],
            "city": [r["city"] for r in rows],
            "enrollment": [None] * len(rows),
            "rate": [r["rate"] for r in rows],
            "band": [r["band"] for r in rows],
            "rate_source": [r["rate_source"] for r in rows],
            "year_start": [r["year_start"] for r in rows],
            "unit": ["school"] * len(rows),
        },
        schema={
            "school_name": pl.String,
            "county_label": pl.String,
            "city": pl.String,
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "band": pl.String,
            "rate_source": pl.String,
            "year_start": pl.Int64,
            "unit": pl.String,
        },
    )
    logged(
        [r for r in rows if r["band"]],
        "changed",
        "banded rate (>95) replaced by the middle of its interval",
    )
    logged(
        [r for r in rows if r["bottom"]],
        "changed",
        "rate '<5' (implausible as a rate, treated as suppressed): missing",
    )
    logged(
        [r for r in rows if r["rate"] is None and not r["bottom"]],
        "changed",
        "rate N/A (suppressed) or not in the column of the school year: missing",
    )
    info = {
        "rows": out.height,
        "years": sorted({r["year_start"] for r in rows}),
        "rate_source_by_year": {
            y: ("mmr" if y in mmr_years else "overall")
            for y in sorted({r["year_start"] for r in rows})
        },
        "scale_by_year": {
            y: ("fraction x100" if f == 100 else "percent")
            for y, f in scale.items()
        },
        "dropped_totals": len(totals),
        "dropped_no_kindergarten": len(empty),
        "dropped_nameless": nameless,
        "other_year_rows_skipped": other_years,
        "banded": sum(1 for r in rows if r["band"]),
        "bands": sorted(middle.items()),
        "bottom_band_missing": sum(1 for r in rows if r["bottom"]),
        "rate_missing": int(out["rate"].null_count()),
    }
    return out, info


# --- Colorado ------------------------------------------------------------
# Colorado school and child care immunization files (CDPHE "School and Child Care Immunization, Facility
# Data", two long-format csv files in the zip folder `Colorado/`: 2017/18-2022/23 and 2023/24-2025/26).
#
# One row per facility x school year x survey x vaccine x metric. Columns used: ID, Site_Name, District_Name,
# Facility_Type, County, Survey_Type, Vaccine, Metric, Value_Percent, Enrollment, Year ("2025/2026").
# The parser takes the table of one file read WITH its header (every column as text; a BOM on the first
# column name is ignored; it may be pre-filtered to the Kindergarten/MMR rows, as read_co.py does to save
# memory) and one year_start.
#
# Rows kept: Survey_Type == "Kindergarten", Vaccine == "MMR", Metric == "Fully Immunized" (the percentage of
# the facility's kindergarten enrollment fully immunized for MMR; the other metrics are In Process, Incomplete
# Record, Medical / Nonmedical Exemption, No Record and are not used), Year == "<y>/<y+1>".
#
# Assumptions (the lead should know them)
#     * Enrollment is the "Enrollment" of the kindergarten record (the number of kindergarten students).
#     * Only Facility_Type "Public School" and "Private School" are kept. Kindergarten records of "Child
#       Care/Preschool" facilities (about 100-150 a year) are dropped and logged: they are not schools.
#     * The values are percentages already (0-100) and numeric in every row (no suppressed values, no bands);
#       ssf.scale_factor still guards against fractions. A value that is not a number would be a missing rate.
#     * County "Unknown" (one school a year from 2019-20) has no county: county_label is None (the row is
#       kept and logged); its county_fips will be empty. District_Name is returned in info for a manual fix.
#     * Every row is a school. A "(ID, year)" appears once (checked: info["duplicate_ids"]).


CO_KEEP_FACILITIES = ("Public School", "Private School")


def parse_colorado(table, year_start, log=None):
    """One Colorado file (frame with the csv header, text cells) -> (frame, info)."""
    table = table.rename({c: c.lstrip("﻿").strip() for c in table.columns})
    need = [
        "ID",
        "Site_Name",
        "District_Name",
        "Facility_Type",
        "County",
        "Survey_Type",
        "Vaccine",
        "Metric",
        "Value_Percent",
        "Enrollment",
        "Year",
    ]
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(
            f"Colorado table without columns {missing}: {table.columns}"
        )
    wanted_year = f"{year_start}/{year_start + 1}"
    mmr = table.filter(
        (pl.col("Survey_Type") == "Kindergarten")
        & (pl.col("Vaccine") == "MMR")
        & (pl.col("Metric") == "Fully Immunized")
        & (pl.col("Year") == wanted_year)
    )
    school = mmr.filter(pl.col("Facility_Type").is_in(CO_KEEP_FACILITIES))
    childcare = mmr.filter(~pl.col("Facility_Type").is_in(CO_KEEP_FACILITIES))
    rows = school.select(
        "ID",
        "Site_Name",
        "District_Name",
        "County",
        "Value_Percent",
        "Enrollment",
    ).to_dicts()
    given = [parse_number(r["Value_Percent"]) for r in rows]
    factor = scale_factor(given)
    unknown = [
        r
        for r in rows
        if (r["County"] or "").strip().lower() in ("unknown", "")
    ]
    out = pl.DataFrame(
        {
            "school_name": [
                (r["Site_Name"] or "").strip() or None for r in rows
            ],
            "county_label": [
                None if r in unknown else r["County"].strip() for r in rows
            ],
            "enrollment": [parse_number(r["Enrollment"]) for r in rows],
            "rate": [None if g is None else g * factor for g in given],
            "rate_source": ["mmr"] * len(rows),
            "unit": ["school"] * len(rows),
        },
        schema_overrides={"enrollment": pl.Float64, "rate": pl.Float64},
    ).with_columns(year_start=pl.lit(year_start, dtype=pl.Int64))
    if log is not None:
        log_rows(
            log,
            childcare.select(
                school_name=pl.col("Site_Name"), year_start=pl.lit(year_start)
            ),
            "CO",
            "colorado rows",
            "dropped",
            "kindergarten record of a child care / preschool facility, not a school",
        )
        log_rows(
            log,
            out.filter(pl.col("county_label").is_null()),
            "CO",
            "colorado values",
            "changed",
            "county 'Unknown': county label missing, school kept",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "CO",
            "colorado values",
            "changed",
            "rate not a number: missing",
        )
    info = {
        "year_start": year_start,
        "rows": out.height,
        "dropped_childcare": childcare.height,
        "duplicate_ids": school.height - school["ID"].n_unique(),
        "enrollment_zero": int((out["enrollment"] == 0).sum()),
        "rate_missing": int(out["rate"].null_count()),
        "unknown_county": [
            (r["Site_Name"], r["District_Name"]) for r in unknown
        ],
        "scale": factor,
    }
    return out, info


# --- Indiana ------------------------------------------------------------
# Indiana school immunization files (Indiana Department of Health, one Excel file a year; the zip folder
# `Indiana/` has 2023-24, 2024-25 and 2025-26). One sheet per file, all grades (K, 1, 6, 7, 12), one row
# per school and grade; only Grade == "K" is kept.
#
# Assumptions (the lead should know them)
#     * Enrollment is "Reported in SNM (CHIRP)" (the kindergarten students the school reported to the
#       state registry): the denominator of the MMR rate (|MMR / reported - MMR rate| < 1e-9 where both are
#       known). "DOE Enrollment" is a different count (the Department of Education's) and is not used.
#     * The rate is "MMR Rate" (a fraction, x100 by ssf.scale_factor), rate_source "mmr", no bands.
#       "School_Reporting_To_CHIRP" is the share of the school's DOE enrollment that was reported: not a rate.
#     * A count of "<=10" is censored. A censored enrollment is missing (the school stays). When the MMR
#       count is censored the file still gives MMR Rate, computed from at most 10 students: those rates are
#       0, 1 or a coarse fraction (the smallest uncensored rate of a file is about 0.43-0.52) and mostly
#       belong to schools that reported few students. They are kept as given, and logged, by default;
#       `keep_censored_count_rates=False` makes them missing.
#     * A rate above 100 (2 schools in 2025-26: the file has more MMR doses than reported students) is kept
#       and logged; the assembler makes it missing.
#     * Header spellings change between years ("Reported in SNM (CHIRP)" / "Reported_in_SNM_CHIRP",
#       "MMR Rate" / "MMR_Rate"; 2023-24 has no County_Code): headers are lower-cased and every run of
#       other characters becomes "_".
#     * Every row is a school (no totals row, no district rows; a same-name school can appear twice in a
#       year, e.g. 72 repeated names in 2023-24, and is kept).


IN_LABELS = (
    "school_name",
    "grade",
    "reported_in_snm_chirp",
    "mmr",
    "mmr_rate",
    "county",
)


def _in_normal(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _in_cell(value):
    """A cell as clean text, or None."""
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip() or None


def parse_indiana(frame, year_start, log=None, keep_censored_count_rates=True):
    """
    One Indiana sheet (headerless frame, cells as text or numbers) -> (frame, info).
    frame: school_name, county_label, enrollment, rate, rate_source, year_start, unit.
    """
    at = header_row(frame, "Corporation_Name")
    table = with_header(frame, at)
    table = table.rename({c: _in_normal(c) for c in table.columns})
    missing = [c for c in IN_LABELS if c not in table.columns]
    if missing:
        raise ValueError(
            f"Indiana sheet without columns {missing}: {table.columns}"
        )
    rows = [
        dict(zip(IN_LABELS, [_in_cell(v) for v in r]))
        for r in table.select(*IN_LABELS).iter_rows()
    ]
    kindergarten = [r for r in rows if (r["grade"] or "").upper() == "K"]
    nameless = [r for r in kindergarten if r["school_name"] is None]
    kept = [r for r in kindergarten if r["school_name"] is not None]

    enrollment = [parse_number(r["reported_in_snm_chirp"]) for r in kept]
    given = [parse_number(r["mmr_rate"]) for r in kept]
    factor = scale_factor(given)
    rate = [None if g is None else g * factor for g in given]
    count_censored = [
        parse_number(r["mmr"]) is None and r["mmr"] is not None for r in kept
    ]
    if not keep_censored_count_rates:
        rate = [None if c else x for x, c in zip(rate, count_censored)]
    out = pl.DataFrame(
        {
            "school_name": [r["school_name"] for r in kept],
            "county_label": [r["county"] for r in kept],
            "enrollment": enrollment,
            "rate": rate,
            "rate_source": ["mmr"] * len(kept),
            "unit": ["school"] * len(kept),
        },
        schema_overrides={"enrollment": pl.Float64, "rate": pl.Float64},
    ).with_columns(year_start=pl.lit(year_start, dtype=pl.Int64))
    flags = out.with_columns(
        count_censored=pl.Series(count_censored, dtype=pl.Boolean),
        enrollment_censored=pl.col("enrollment").is_null(),
    )
    if log is not None:

        def say(df, step, action, reason):
            log_rows(log, df, "IN", step, action, reason)

        say(
            pl.DataFrame(
                {
                    "school_name": [r["school_name"] for r in nameless],
                    "year_start": [year_start] * len(nameless),
                }
            ),
            "indiana rows",
            "dropped",
            "kindergarten row without a school name",
        )
        say(
            flags.filter(pl.col("enrollment_censored")),
            "indiana values",
            "changed",
            "enrollment '<=10' (censored): missing",
        )
        say(
            flags.filter(
                pl.col("count_censored") & pl.col("rate").is_not_null()
            ),
            "indiana values",
            "kept",
            "MMR count '<=10' (censored), rate given by the file from at most 10 students: kept",
        )
        say(
            flags.filter(pl.col("count_censored") & pl.col("rate").is_null()),
            "indiana values",
            "changed",
            "MMR count '<=10' (censored): rate made missing",
        )
        say(
            flags.filter(pl.col("rate") > 100),
            "indiana values",
            "kept",
            "rate above 100: kept (the assembler makes it missing)",
        )
        say(
            flags.filter(pl.col("rate").is_null() & ~pl.col("count_censored")),
            "indiana values",
            "changed",
            "rate missing in the file",
        )
    rated = flags.filter(pl.col("rate").is_not_null())
    info = {
        "year_start": year_start,
        "rows": out.height,
        "other_grade_rows": len(rows) - len(kindergarten),
        "nameless_dropped": len(nameless),
        "enrollment_censored": int(flags["enrollment_censored"].sum()),
        "mmr_count_censored": int(flags["count_censored"].sum()),
        "censored_count_rate_zero": int(
            rated.filter(
                pl.col("count_censored") & (pl.col("rate") == 0)
            ).height
        ),
        "censored_count_rate_100": int(
            rated.filter(
                pl.col("count_censored") & (pl.col("rate") == 100)
            ).height
        ),
        "rate_above_100": int(flags.filter(pl.col("rate") > 100).height),
        "rate_missing": int(out["rate"].null_count()),
        "scale": factor,
    }
    return out, info


# --- North Carolina ------------------------------------------------------------
# North Carolina school immunization files ("School Specific Download Page", North Carolina DHHS; the zip
# folder `North Carolina/` has one folder per school year 2020-21 .. 2025-26, each with the same table as
# an .xlsx and as a UTF-16 tab-separated .csv, plus a stacked combined.csv).
#
# One row per school: County, School Name, School Type (Public / Charter / Private), Total Enrollment, and a
# count and a % for Up to Date, Provisionally Enrolled, Not Up to Date, Medical Exemption, Religious Exemption.
#
# Assumptions (the lead should know them)
#     * The file never says "kindergarten". Total Enrollment is taken as the kindergarten enrollment: the
#       NC kindergarten immunization assessment is the source and the sums are one cohort (see read_nc.py).
#     * There is no MMR column. The rate is "Up to Date (%)", the share of the school's students who are up
#       to date on ALL required vaccines (count / enrollment, checked): rate_source is "overall" in every year.
#     * The .xlsx holds the full-precision fractions ("Up to Date (%)" = 0.9677...), merged County cells (the
#       county is only on the first row of each county: it is carried forward in this parser); the .csv has
#       the percentages as text ("96.8%") and the county on every row. Both go through this parser: a text
#       with "%" is a percentage (x1), a number is a fraction when ssf.scale_factor says so.
#     * No city and no censored values (counts are given for any size). A row without a school name or one
#       called "total"/"state" is dropped and logged (none in the files).
#     * A school with enrollment 0 would have no rate (missing).


def _nc_cell(value):
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip() or None


def parse_north_carolina(frame, year_start, log=None):
    """One NC sheet or csv (headerless frame; cells as text or numbers) -> (frame, info)."""
    at = header_row(frame, "County")
    table = with_header(frame, at)
    names = {c: _nc_cell(c).lower() for c in table.columns}
    by_name = {v: k for k, v in names.items()}
    need = ["county", "school name", "total enrollment", "up to date (%)"]
    missing = [n for n in need if n not in by_name]
    if missing:
        raise ValueError(
            f"North Carolina sheet without columns {missing}: {list(names.values())}"
        )
    columns = [by_name[n] for n in need]
    county, school, enroll, pct = (
        [_nc_cell(v) for v in table[c].to_list()] for c in columns
    )
    # merged county cells: the county is on the first row of its block only
    carried, last = [], None
    for c in county:
        last = c or last
        carried.append(last)
    is_total = [
        s is None or re.search(r"(?i)\b(total|state ?wide)\b", s) is not None
        for s in school
    ]
    keep = [i for i, t in enumerate(is_total) if not t]
    given = [
        v if isinstance(v, float) else None
        for v in (parse_number(pct[i]) for i in keep)
    ]
    # a text with "%" is already a percentage, a bare number is a fraction or a percentage by its median
    percent_text = [pct[i] is not None and "%" in pct[i] for i in keep]
    factor = 1.0 if all(percent_text) and keep else scale_factor(given)
    rate = [None if g is None else g * factor for g in given]
    enrollment = [parse_number(enroll[i]) for i in keep]
    rate = [None if e == 0 else r for r, e in zip(rate, enrollment)]
    out = pl.DataFrame(
        {
            "school_name": [school[i] for i in keep],
            "county_label": [carried[i] for i in keep],
            "enrollment": enrollment,
            "rate": rate,
            "rate_source": ["overall"] * len(keep),
            "unit": ["school"] * len(keep),
        },
        schema_overrides={"enrollment": pl.Float64, "rate": pl.Float64},
    ).with_columns(year_start=pl.lit(year_start, dtype=pl.Int64))
    dropped = [i for i, t in enumerate(is_total) if t]
    if log is not None:
        log_rows(
            log,
            pl.DataFrame(
                {
                    "school_name": [school[i] or "" for i in dropped],
                    "year_start": [year_start] * len(dropped),
                }
            ),
            "NC",
            "north carolina rows",
            "dropped",
            "nameless or total row",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "NC",
            "north carolina values",
            "changed",
            "rate missing (enrollment 0 or no rate in the file)",
        )
    info = {
        "year_start": year_start,
        "rows": out.height,
        "dropped_totals": len(dropped),
        "enrollment_sum": float(out["enrollment"].sum()),
        "enrollment_zero": int((out["enrollment"] == 0).sum()),
        "rate_missing": int(out["rate"].null_count()),
        "county_carried_rows": sum(1 for c in county if c is None),
        "scale": factor,
    }
    return out, info


# --- Massachusetts ------------------------------------------------------------
# Massachusetts kindergarten immunization survey by school (Massachusetts Department of
# Public Health, Immunization Division), one Excel file per school year. Pure parsing of an
# already-read, headerless sheet (see read_ma.py for the files).
#
# parse_ma(frame, year_start, log=None) -> (frame, info)
# frame: school_name, county_label, city, enrollment, rate, rate_source, year_start, unit.
#
# Assumptions
#     * The sheet is the "Kindergarten Rates by School" one (the file also has a Notes
#       sheet and, in 2021-22, three unused scratch sheets). One row per school: public,
#       private, charter, collaborative and special-education programs ("School Type" is
#       not carried). All of them are kept: unit = "school".
#     * The header row is the first of the first 5 rows whose first cell starts with
#       "School" ("SCHOOL", "School Name", "SCHOOL NAME"); header cells may contain line
#       breaks ("2\\r\\nMMR "), so they are compared with white space collapsed. Columns are
#       found by header text: "City" or "Municipality", "County", and the first header
#       that contains "MMR" ("2 MMR": two doses of MMR; the file's other columns are DTaP,
#       polio, hepatitis B, varicella, the series, exemptions, no record, non-compliance).
#     * rate = "2 MMR", rate_source "mmr". The year's scale comes from scale_factor
#       (2014-2020 and 2022-2025 are fractions, 2021-22 is percent).
#     * There is NO enrollment column (enrollment is always missing). The file says that
#       schools with fewer than 30 students are suppressed; the number of students is
#       never given.
#     * A symbol in the rate cell is a missing rate and the school stays (log row):
#       "†" fewer than 30 students (suppressed), "*" the school did not respond,
#       "¶" rates withheld because of unresolved discrepancies, "N/A" zero students,
#       "‡" negative non-compliance. No band exists in this file.
#     * Rows without a school name, or without city and county (the footnotes below the
#       table in 2014-2018), are dropped; so is a row whose name is a statewide total.
#       (None of the latter exists in the real files.)
#     * `year_start` is given by the caller (the file name carries it).
MA_TOTAL_NAME = re.compile(
    r"(?i)(statewide|state total|total|massachusetts( total)?|all schools)"
)
MA_SYMBOLS = {
    "†": "fewer than 30 students: suppressed",
    "*": "school did not respond",
    "¶": "rates withheld (unresolved discrepancies)",
    "N/A": "zero students reported",
    "‡": "negative non-compliance rate",
}


def _ma_text(value):
    """A cell as clean text (white space collapsed), or None."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def parse_massachusetts(frame, year_start, log=None):
    at = next(
        (
            i
            for i in range(min(5, frame.height))
            if (_ma_text(frame.row(i)[0]) or "").lower().startswith("school")
        ),
        None,
    )
    if at is None:
        raise ValueError(
            "no header row whose first cell starts with 'School' in the first 5 rows"
        )
    header = [(_ma_text(v) or "").lower() for v in frame.row(at)]

    def column(test, what):
        found = [j for j, h in enumerate(header) if test(h)]
        if not found:
            raise ValueError(f"no {what} column in {header}")
        return found[0]

    city_c = column(lambda h: h in ("city", "municipality"), "city")
    county_c = column(lambda h: h == "county", "county")
    rate_c = column(lambda h: "mmr" in h, "MMR")
    kept, no_name, totals = [], 0, []
    for row in frame[at + 1 :].iter_rows():
        name, city, county = (
            _ma_text(row[0]),
            _ma_text(row[city_c]),
            _ma_text(row[county_c]),
        )
        if name is None or (city is None and county is None):
            no_name += 1  # a footnote or an empty row
        elif MA_TOTAL_NAME.fullmatch(name):
            totals.append(name)
        else:
            kept.append((name, county, city, _ma_text(row[rate_c])))
    given, _ = parse_rates([k[3] for k in kept])
    symbols = [k[3] if g is None else None for k, g in zip(kept, given)]
    factor = scale_factor(given)
    out = pl.DataFrame(
        {
            "school_name": [k[0] for k in kept],
            "county_label": [k[1] for k in kept],
            "city": [k[2] for k in kept],
            "enrollment": [None] * len(kept),
            "rate": [None if g is None else g * factor for g in given],
            "rate_source": ["mmr"] * len(kept),
            "year_start": [year_start] * len(kept),
            "unit": ["school"] * len(kept),
        },
        schema={
            "school_name": pl.String,
            "county_label": pl.String,
            "city": pl.String,
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "rate_source": pl.String,
            "year_start": pl.Int64,
            "unit": pl.String,
        },
    )
    if log is not None:
        log_rows(
            log,
            pl.DataFrame(
                {
                    "school_name": totals,
                    "year_start": [year_start] * len(totals),
                },
                schema={"school_name": pl.String, "year_start": pl.Int64},
            ),
            "MA",
            "massachusetts rows",
            "dropped",
            "statewide total row",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "MA",
            "massachusetts values",
            "changed",
            "rate is a symbol (†, *, ¶, N/A, ‡) or blank: missing",
        )
    counts = {}
    for s in symbols:
        if s is not None:
            counts[s] = counts.get(s, 0) + 1
    info = {
        "year_start": year_start,
        "rows": out.height,
        "header_at": at,
        "scale": "fraction x100" if factor == 100.0 else "percent",
        "dropped_footnote_or_empty": no_name,
        "dropped_totals": len(totals),
        "rate_missing": int(out["rate"].null_count()),
        "symbols": counts,
    }
    return out, info


# --- Minnesota ------------------------------------------------------------
# Minnesota Department of Health, "Percent of Kindergarteners Vaccinated by School": the
# "K_School" sheet of one Excel workbook per school year. Pure parsing of an already-read,
# headerless sheet (see read_mn.py for the files).
#
# parse_mn(frame, year_start, log=None) -> (frame, info)
# frame: school_name, county_label, enrollment, rate, band, rate_source, year_start, unit
# (plus city: not in the file, always missing).
#
# Layout handled (2019-20 and 2021-22 to 2025-26): row 0 is a title ("Percent of
# Kindergarteners Vaccinated by School 2025-26"), row 1 the header with School District,
# School Name, County, Kindergarten Enrollment and, per vaccine, "<vaccine> % Vaccinated",
# "... % Partially Vaccinated or No Doses", "... % non-medical", "... % medical". The older
# workbooks (2013-14 to 2018-19 and 2020-21) have no County column and a two-row grouped
# header (the vaccine in a group row, an unlabelled "% Vaccinated" under it); they are
# refused with MNOldLayout (a school without a county cannot be used, and there is no
# rule to cut a county out of them).
#
# Assumptions
#     * rate = "MMR % Vaccinated": the share of the kindergartners with the full MMR
#       series (two doses); partially vaccinated, no doses and the medical and
#       non-medical exemptions are other columns, so exempt children count as not
#       vaccinated. rate_source "mmr". The year's scale comes from scale_factor
#       (fractions in all real files).
#     * enrollment = "Kindergarten Enrollment" (number of kindergartners).
#     * A school is one row (unit "school"); the row "Statewide" (school district, school
#       name and county all "Statewide") is dropped (log row).
#     * Censored cells, a missing value (rows kept, log row):
#         school did not report: enrollment "Did not report", rate "*";
#         fewer than 10 (5 up to 2022-23) kindergartners: enrollment "Enrollment < 10"
#         ("Enrollment <10" in 2024-25), rate "**" (a censored enrollment is missing, as in
#         read_chen_bento);
#         2021-22 only, 3 schools: rate "***" (the school reported a data error after
#         submitting; redacted), the enrollment is given.
#     * Bands (2024-25 and 2025-26, schools with high coverage): the rate cell reads
#       ">95%" or ">98%": text in `band`, rate = the middle of its interval from
#       ssf.band_midpoints of the file (">95%" runs to 98: 96.5; ">98%" runs to 100: 99.0).
#       The enrollment of a banded school is given.
#     * The title's school year (in the title text) must agree with `year_start`.
class MNOldLayout(ValueError):
    """A workbook without a County column or an "MMR % Vaccinated" header."""


def _mn_text(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def parse_minnesota(frame, year_start, log=None):
    at = next(
        (
            i
            for i in range(min(8, frame.height))
            if (_mn_text(frame.row(i)[0]) or "").lower() == "school district"
        ),
        None,
    )
    if at is None:
        raise MNOldLayout(
            "no header row starting with 'School District' in the first 8 rows"
        )
    header = [(_mn_text(v) or "").lower() for v in frame.row(at)]

    def column(text):
        found = [j for j, h in enumerate(header) if h == text]
        if not found:
            raise MNOldLayout(f"no {text!r} column in {header}")
        return found[0]

    name_c, county_c = column("school name"), column("county")
    enroll_c, rate_c = (
        column("kindergarten enrollment"),
        column("mmr % vaccinated"),
    )
    title = _mn_text(frame.row(0)[0]) or ""
    match = re.search(r"(20\d\d)-(\d\d)\b", title)
    if match and int(match.group(1)) != year_start:
        raise ValueError(
            f"title says school year {match.group(1)}-{match.group(2)}, year_start is {year_start}"
        )

    kept, totals, empty = [], [], 0
    for row in frame[at + 1 :].iter_rows():
        name = _mn_text(row[name_c])
        if name is None:
            empty += 1
        elif name.lower() == "statewide":
            totals.append(name)
        else:
            kept.append(
                (
                    name,
                    _mn_text(row[county_c]),
                    _mn_text(row[enroll_c]),
                    _mn_text(row[rate_c]),
                )
            )
    numbers, bands = parse_rates([k[3] for k in kept])
    factor = scale_factor(numbers)
    middle = band_midpoints([b for b in bands if b])
    rate = [
        n * factor if n is not None else middle.get(b)
        for n, b in zip(numbers, bands)
    ]
    enrollment = [parse_number(k[2]) for k in kept]
    out = pl.DataFrame(
        {
            "school_name": [k[0] for k in kept],
            "county_label": [k[1] for k in kept],
            "city": [None] * len(kept),
            "enrollment": enrollment,
            "rate": rate,
            "band": bands,
            "rate_source": ["mmr"] * len(kept),
            "year_start": [year_start] * len(kept),
            "unit": ["school"] * len(kept),
        },
        schema={
            "school_name": pl.String,
            "county_label": pl.String,
            "city": pl.String,
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "band": pl.String,
            "rate_source": pl.String,
            "year_start": pl.Int64,
            "unit": pl.String,
        },
    )
    if log is not None:
        log_rows(
            log,
            pl.DataFrame(
                {
                    "school_name": totals,
                    "year_start": [year_start] * len(totals),
                },
                schema={"school_name": pl.String, "year_start": pl.Int64},
            ),
            "MN",
            "minnesota rows",
            "dropped",
            "statewide total row",
        )
        log_rows(
            log,
            out.filter(pl.col("band").is_not_null()),
            "MN",
            "minnesota values",
            "changed",
            "banded rate (>95% / >98%) replaced by the middle of its interval",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "MN",
            "minnesota values",
            "changed",
            "rate censored (* did not report, ** small enrollment) or blank: missing",
        )
        log_rows(
            log,
            out.filter(pl.col("enrollment").is_null()),
            "MN",
            "minnesota values",
            "changed",
            "enrollment censored (Did not report, Enrollment < n) or blank: missing",
        )
    tokens = {}
    for k, n, b in zip(kept, numbers, bands):
        for cell, number in ((k[2], parse_number(k[2])), (k[3], n)):
            if (
                number is None
                and cell is not None
                and not (cell == k[3] and b)
            ):
                tokens[cell] = tokens.get(cell, 0) + 1
    info = {
        "year_start": year_start,
        "rows": out.height,
        "header_at": at,
        "scale": "fraction x100" if factor == 100.0 else "percent",
        "dropped_totals": len(totals),
        "dropped_empty": empty,
        "banded": sum(1 for b in bands if b),
        "bands": sorted(middle.items()),
        "rate_missing": int(out["rate"].null_count()),
        "enrollment_missing": int(out["enrollment"].null_count()),
        "censored_cells": tokens,
    }
    return out, info


# --- Pennsylvania ------------------------------------------------------------
# Pennsylvania Department of Health, "school immunization rates" (school year 2025-26): one CSV
# of all schools and three grades. Pure parsing of the already-read, headerless file (all
# cells text; see read_pa.py).
#
# parse_pa(frame, year_start, log=None) -> (frame, info)
# frame: school_name, county_label, enrollment, rate, rate_source, year_start, unit
# (plus city: not in the file, always missing).
#
# Assumptions
#     * The header row (first 5 rows) starts with "County" and has "School", "Grade",
#       "Total Students Enrolled" and "MMR Percent" (columns found by their text).
#     * The file has one row per school AND grade: Kindergarten, 7th Grade and 12th Grade. Only
#       "Kindergarten" rows are kept (the others are dropped and logged); a school has at most
#       one Kindergarten row.
#     * enrollment = "Total Students Enrolled" of the Kindergarten row (kindergarten students
#       enrolled; "1,012" is read as 1012).
#     * rate = "MMR Percent" ("95.0%": already a percent; scale_factor decides, as for every
#       file), rate_source "mmr". The file does not say how many doses; the school-level
#       percentage is of the enrolled students of the grade.
#     * The MMR cell (and every other rate cell) of a school with fewer than 20 kindergartners
#       is blank (suppressed; all 580 blank kindergarten rates have enrollment 1 to 19,
#       all rated ones 20 or more): a missing rate, the school stays (log row). Enrollment is
#       given for them.
#     * unit "school" for every row: private schools and child-care centres included.
#     * A row named like a statewide or county total is dropped (none in the real file).
PA_TOTAL_NAME = re.compile(
    r"(?i)(statewide|state total|total|pennsylvania( total)?|all schools|county total)"
)


def _pa_text(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def parse_pennsylvania(frame, year_start, log=None):
    at = next(
        (
            i
            for i in range(min(5, frame.height))
            if (_pa_text(frame.row(i)[0]) or "").lower() == "county"
        ),
        None,
    )
    if at is None:
        raise ValueError(
            "no header row starting with 'County' in the first 5 rows"
        )
    header = [(_pa_text(v) or "").lower() for v in frame.row(at)]

    def column(text):
        found = [j for j, h in enumerate(header) if h == text]
        if not found:
            raise ValueError(f"no {text!r} column in {header}")
        return found[0]

    county_c, name_c, grade_c = (
        column("county"),
        column("school"),
        column("grade"),
    )
    enroll_c, rate_c = column("total students enrolled"), column("mmr percent")
    kept, other_grades, totals = [], 0, []
    for row in frame[at + 1 :].iter_rows():
        if _pa_text(row[grade_c]) != "Kindergarten":
            other_grades += 1
            continue
        name = _pa_text(row[name_c])
        if name is None or PA_TOTAL_NAME.fullmatch(name):
            totals.append(name or "")
            continue
        kept.append(
            (
                name,
                _pa_text(row[county_c]),
                _pa_text(row[enroll_c]),
                _pa_text(row[rate_c]),
            )
        )
    given, bands = parse_rates([k[3] for k in kept])
    if any(bands):
        raise ValueError(
            "a band in the MMR column: " + str([b for b in bands if b][:3])
        )
    factor = scale_factor(given)
    out = pl.DataFrame(
        {
            "school_name": [k[0] for k in kept],
            "county_label": [k[1] for k in kept],
            "city": [None] * len(kept),
            "enrollment": [
                parse_number(k[2].replace(",", "") if k[2] else None)
                for k in kept
            ],
            "rate": [None if g is None else g * factor for g in given],
            "rate_source": ["mmr"] * len(kept),
            "year_start": [year_start] * len(kept),
            "unit": ["school"] * len(kept),
        },
        schema={
            "school_name": pl.String,
            "county_label": pl.String,
            "city": pl.String,
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "rate_source": pl.String,
            "year_start": pl.Int64,
            "unit": pl.String,
        },
    )
    if log is not None:
        yr = [year_start]
        log_rows(
            log,
            pl.DataFrame(
                {
                    "school_name": ["(grade 7 or 12 row)"] * other_grades,
                    "year_start": yr * other_grades,
                },
                schema={"school_name": pl.String, "year_start": pl.Int64},
            ),
            "PA",
            "pennsylvania rows",
            "dropped",
            "grade is not Kindergarten (7th or 12th grade row)",
        )
        log_rows(
            log,
            pl.DataFrame(
                {"school_name": totals, "year_start": yr * len(totals)},
                schema={"school_name": pl.String, "year_start": pl.Int64},
            ),
            "PA",
            "pennsylvania rows",
            "dropped",
            "total row or nameless row",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "PA",
            "pennsylvania values",
            "changed",
            "MMR rate blank (fewer than 20 kindergartners, suppressed): missing",
        )
    info = {
        "year_start": year_start,
        "rows": out.height,
        "header_at": at,
        "scale": "fraction x100" if factor == 100.0 else "percent",
        "dropped_other_grades": other_grades,
        "dropped_totals": len(totals),
        "rate_missing": int(out["rate"].null_count()),
        "enrollment_missing": int(out["enrollment"].null_count()),
        "max_enrollment_of_blank_rate": out.filter(pl.col("rate").is_null())[
            "enrollment"
        ].max(),
        "min_enrollment_of_rated": out.filter(pl.col("rate").is_not_null())[
            "enrollment"
        ].min(),
    }
    return out, info


# --- Texas ------------------------------------------------------------
# Texas Department of State Health Services, "School Vaccination Coverage Levels - Kindergarten":
# sheet "Coverage by District" of one Excel workbook per school year. Pure parsing of an
# already-read, headerless sheet (see read_tx.py for the files).
#
# parse_tx(frame, year_start, log=None) -> (frame, info)
# frame: school_name, county_label, city, enrollment, rate, rate_source, year_start, unit.
#
# Assumptions
#     * Rows: row 0 is a title ("Vaccination Coverage Rates among Kindergarteners in Texas,
#       by School, 2019-2020"), row 1 a note ("This list does not include schools with fewer
#       than 5 students in kindergarten"; "equal or fewer than 5" from 2024-25), the header
#       row is the one that starts with "Facility Number" (row 2 in all files). Columns are
#       found by header text: "School Type", "Facility Name", "Facility Address", "County",
#       "MMR". The workbook's second sheet ("Coverage by County") is a county aggregate and is
#       not read.
#     * THE ROWS ARE NOT ALL SCHOOLS. "School Type" is "Public ISD" (2019-20, 2023-24) or
#       "Public School" (the other years) for the rows of PUBLIC DISTRICTS (independent school
#       districts and charter operators, listed once each; the facility number ends in 000,
#       e.g. 9057829000 "A+ ACADEMY", 9109901000 "ABBOTT ISD"): the coverage of the district's
#       kindergartners, not of a building. unit = "district" for them. "Private School" rows
#       are single private schools (facility numbers 72xxxxxxxx): unit = "school".
#       About 1,100 district rows and 600 private-school rows a year. A school type that is
#       neither is an error.
#     * rate = "MMR" (fraction of the kindergartners with the MMR requirement met, x100 by
#       scale_factor), rate_source "mmr". There is no enrollment column: enrollment is
#       always missing (the file lists only units with more than 5 kindergartners).
#     * A blank MMR cell (2019-22) and "NR" (2022-23, 2023-24: not reported) are missing
#       rates, the row stays (log row).
#     * city = the token before "TX" of "Facility Address" ("5701 RED BIRD CTR DR, DALLAS, TX,
#       75237" -> "DALLAS"); the county label is the file's "County" (the county of the
#       address).
#     * The title's school year must agree with `year_start`. A row named like a statewide
#       total is dropped (none in the real files).
TX_TOTAL_NAME = re.compile(
    r"(?i)(statewide|state total|total|texas( total)?|all schools)"
)
TX_CITY = re.compile(r",\s*([^,]+?)\s*,\s*TX\b", re.IGNORECASE)


def _tx_text(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def parse_texas(frame, year_start, log=None):
    at = next(
        (
            i
            for i in range(min(8, frame.height))
            if (_tx_text(frame.row(i)[0]) or "").lower() == "facility number"
        ),
        None,
    )
    if at is None:
        raise ValueError(
            "no header row starting with 'Facility Number' in the first 8 rows"
        )
    header = [(_tx_text(v) or "").lower() for v in frame.row(at)]

    def column(text):
        found = [j for j, h in enumerate(header) if h == text]
        if not found:
            raise ValueError(f"no {text!r} column in {header}")
        return found[0]

    type_c, name_c, addr_c = (
        column("school type"),
        column("facility name"),
        column("facility address"),
    )
    county_c, rate_c = column("county"), column("mmr")
    match = re.search(r"(20\d\d)-(20\d\d)", _tx_text(frame.row(0)[0]) or "")
    if match and int(match.group(1)) != year_start:
        raise ValueError(
            f"title says school year {match.group(1)}-{match.group(2)}, year_start is {year_start}"
        )

    kept, totals, empty = [], [], 0
    for row in frame[at + 1 :].iter_rows():
        name = _tx_text(row[name_c])
        if name is None:
            empty += 1
            continue
        if TX_TOTAL_NAME.fullmatch(name):
            totals.append(name)
            continue
        kind = (_tx_text(row[type_c]) or "").lower()
        if kind.startswith("public"):
            unit = "district"
        elif kind.startswith("private"):
            unit = "school"
        else:
            raise ValueError(
                f"school type {row[type_c]!r} of {name!r} is neither public nor private"
            )
        city = TX_CITY.search(_tx_text(row[addr_c]) or "")
        kept.append(
            (
                name,
                _tx_text(row[county_c]),
                city.group(1) if city else None,
                _tx_text(row[rate_c]),
                unit,
                _tx_text(row[0]),
            )
        )
    given, bands = parse_rates([k[3] for k in kept])
    if any(bands):
        raise ValueError(
            "a band in the MMR column: " + str([b for b in bands if b][:3])
        )
    factor = scale_factor(given)
    out = pl.DataFrame(
        {
            "school_name": [k[0] for k in kept],
            "county_label": [k[1] for k in kept],
            "city": [k[2] for k in kept],
            "enrollment": [None] * len(kept),
            "rate": [None if g is None else g * factor for g in given],
            "rate_source": ["mmr"] * len(kept),
            "year_start": [year_start] * len(kept),
            "unit": [k[4] for k in kept],
        },
        schema={
            "school_name": pl.String,
            "county_label": pl.String,
            "city": pl.String,
            "enrollment": pl.Float64,
            "rate": pl.Float64,
            "rate_source": pl.String,
            "year_start": pl.Int64,
            "unit": pl.String,
        },
    )
    if log is not None:
        log_rows(
            log,
            pl.DataFrame(
                {
                    "school_name": totals,
                    "year_start": [year_start] * len(totals),
                },
                schema={"school_name": pl.String, "year_start": pl.Int64},
            ),
            "TX",
            "texas rows",
            "dropped",
            "statewide total row",
        )
        log_rows(
            log,
            out.filter(pl.col("rate").is_null()),
            "TX",
            "texas values",
            "changed",
            "MMR blank or NR (not reported): missing",
        )
    tokens = {}
    for k, g in zip(kept, given):
        if g is None and k[3] is not None:
            tokens[k[3]] = tokens.get(k[3], 0) + 1
    info = {
        "year_start": year_start,
        "rows": out.height,
        "header_at": at,
        "scale": "fraction x100" if factor == 100.0 else "percent",
        "district_rows": sum(1 for k in kept if k[4] == "district"),
        "school_rows": sum(1 for k in kept if k[4] == "school"),
        "district_facility_not_000": sum(
            1
            for k in kept
            if k[4] == "district" and not (k[5] or "").endswith("000")
        ),
        "dropped_totals": len(totals),
        "dropped_empty": empty,
        "rate_missing": int(out["rate"].null_count()),
        "tokens": tokens,
        "blank_rates": sum(
            1 for k, g in zip(kept, given) if g is None and k[3] is None
        ),
    }
    return out, info


# --- the school year of a state file, from its name ---------------------------


def file_year_massachusetts(name):
    """First year of the school year in a Massachusetts file name ("...-14-15.xlsx", "... 2020- 2021.xlsx"); None for the undated file."""
    match = re.search(r"(\d{4}|\d{2})\s*-\s*(\d{4}|\d{2})\.xlsx?$", name)
    if match is None:
        return None
    first = int(match.group(1))
    return first + 2000 if first < 100 else first


def file_year_minnesota(name):
    """First year of the school year from the last four digits of a Minnesota file name ("...1920.xlsx" -> 2019)."""
    match = re.search(r"(\d\d)(\d\d)\.xlsx$", name)
    if match is None or int(match.group(2)) != (int(match.group(1)) + 1) % 100:
        raise ValueError(f"cannot read a school year from {name!r}")
    return 2000 + int(match.group(1))


def file_year_pennsylvania(name):
    """First year of the school year in a Pennsylvania file name ("... 2025-2026 ..." -> 2025)."""
    match = re.search(r"(20\d\d)-(20\d\d)", name)
    if match is None or int(match.group(2)) != int(match.group(1)) + 1:
        raise ValueError(f"cannot read a school year from {name!r}")
    return int(match.group(1))


def file_year_texas(name):
    """First year of the school year from a Texas file name ("2019-2020-..." or "22-23-...")."""
    match = re.search(
        r"(?:^|[^\d])(20\d\d|\d\d)-(20\d\d|\d\d)(?!\d)",
        name.rsplit("/", 1)[-1],
    )
    if match is None:
        raise ValueError(f"cannot read a school year from {name!r}")
    first = int(match.group(1))
    return first + 2000 if first < 100 else first


def read_sheet(source, sheet=None):
    """A sheet of an Excel file (path or bytes) as a headerless frame of text/numbers."""
    source = (
        io.BytesIO(source)
        if isinstance(source, (bytes, bytearray))
        else source
    )
    kwargs = {"read_options": {"header_row": None}}
    return (
        pl.read_excel(source, sheet_name=sheet, **kwargs)
        if sheet
        else pl.read_excel(source, **kwargs)
    )


def attach_counties(df, state, counties):
    """
    Adds county_fips and county_name from the county label (map_counties raises
    on a mismatch). A label of two counties ("Lexington/Richland": a school on a
    county line) has no single county: its county_fips is empty.
    """
    found = map_counties(
        [n for n in df["county_label"].to_list() if n is None or "/" not in n],
        state,
        counties,
    )
    return df.with_columns(
        county_fips=pl.col("county_label").map_elements(
            lambda n: found[n][0] if n in found else None,
            return_dtype=pl.String,
        ),
        county_name=pl.col("county_label").map_elements(
            lambda n: found[n][1] if n in found else None,
            return_dtype=pl.String,
        ),
    )
