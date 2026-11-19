"""Write the hashed data files for the reviewer to outputs/reviewer_data/ (and nothing else).

The five files: analysis_dataset.csv, outbreaks_vaccinations_cleaned.csv, processed_vaccinations_data.csv,
county_adjacencies.txt, outbreak_records_master.csv. County names, FIPS codes, region names and geoids are replaced by a keyed
hash (HMAC-SHA256, key in .secrets/hash_key, never printed or shipped). Row order is never changed (the models index
counties by first appearance). Originals are only read; the unhashed master comes from outputs/reviewer_data_unhashed/
(written by 12_export_reviewer_data.py).

    uv run analysis/13_build_hashed_reviewer_bundle.py
"""

import hashlib
import hmac
import os

import polars as pl

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo, "data")
out_dir = os.path.join(repo, "outputs", "reviewer_data")
secrets_dir = os.path.join(repo, ".secrets")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(secrets_dir, "hash_key"), "rb") as f:
    key = f.read().strip()
assert len(key) >= 32, "hash key looks too short"

crosswalk: dict[str, tuple[str, str]] = {}  # hashed -> (original, kind)


def H(value: str, kind: str) -> str:
    """keyed hash, 'h_' + 12 hex characters (a letter first so nothing reads it as a number)"""
    h = "h_" + hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:12]
    assert crosswalk.setdefault(h, (value, kind))[0] == value, "hash collision"
    return h


def hash_county(df: pl.DataFrame) -> pl.Series:
    # county + state together, so e.g. every 'Washington' county does not share one hash
    return pl.Series(
        "county",
        [
            None if c is None else H(f"{c}|{s}", "county|state")
            for c, s in zip(df["county"], df["state"])
        ],
        dtype=pl.String,
    )


def hash_plain(s: pl.Series, kind: str) -> pl.Series:
    return pl.Series(
        s.name, [None if v is None else H(v, kind) for v in s], dtype=pl.String
    )


def hash_list(
    s: pl.Series,
) -> pl.Series:  # 'a;b;c' -> hashed elements joined by ';'
    return pl.Series(
        s.name,
        [
            None if v is None else ";".join(H(x, "fips") for x in v.split(";"))
            for v in s
        ],
        dtype=pl.String,
    )


def read_text(
    path, sep=","
):  # everything as text, so values round-trip exactly
    return pl.read_csv(path, separator=sep, infer_schema_length=0)


FILES = [  # (source path, output name, separator, columns to hash)
    (
        os.path.join(data_dir, "analysis_dataset.csv"),
        "analysis_dataset.csv",
        ",",
        ["county", "fips", "geoid"],
    ),
    (
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv"),
        "outbreaks_vaccinations_cleaned.csv",
        ",",
        ["county", "fips", "geoid"],
    ),
    (
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        "processed_vaccinations_data.csv",
        ",",
        ["county", "fips"],
    ),
    (
        os.path.join(data_dir, "county_adjacencies.txt"),
        "county_adjacencies.txt",
        "|",
        ["County Name", "County GEOID", "Neighbor Name", "Neighbor GEOID"],
    ),
    (
        os.path.join(
            repo,
            "outputs",
            "reviewer_data_unhashed",
            "outbreak_records_master.csv",
        ),
        "outbreak_records_master.csv",
        ",",
        ["county", "fips", "fips_counties_in_region"],
    ),
]

checks = []
for src, name, sep, cols in FILES:
    df = read_text(src, sep)
    for c in cols:  # refuse input that is already hashed (a rerun must never hash the hashes)
        assert not df[c].drop_nulls().str.starts_with("h_").any(), (
            f"{src}: column {c} looks already hashed"
        )
    hashed = df
    for c in cols:
        if c == "county":
            hashed = hashed.with_columns(hash_county(df))
        elif c == "fips_counties_in_region":
            hashed = hashed.with_columns(hash_list(df[c]))
        else:
            hashed = hashed.with_columns(
                hash_plain(
                    df[c], "fips" if "fips" in c.lower() or "GEOID" in c else c
                )
            )
    hashed.write_csv(os.path.join(out_dir, name), separator=sep)
    print(f"wrote {name}: {hashed.shape}")
    checks.append((name, cols, df, hashed))

# --- checks: shape/order preserved, every hashed cell is a hash or empty, un-hashing returns the original text -------
for name, cols, df, hashed in checks:
    assert hashed.shape == df.shape and hashed.columns == df.columns, name
    for c in cols:
        for v in hashed[c].drop_nulls():
            assert all(p.startswith("h_") for p in v.split(";")), (name, c, v)
        assert hashed[c].null_count() == df[c].null_count(), (name, c)
        assert hashed[c].str.contains(" ").sum() == 0, (
            name,
            c,
        )  # no name text left
    untouched = [c for c in df.columns if c not in cols]
    assert hashed.select(untouched).equals(df.select(untouched)), name
    for c in cols:
        if c == "county":  # stored as 'county|state'
            back = [
                None if v is None else crosswalk[v][0].split("|")[0]
                for v in hashed[c]
            ]
        else:
            back = [
                None
                if v is None
                else ";".join(crosswalk[x][0] for x in v.split(";"))
                for v in hashed[c]
            ]
        assert back == df[c].to_list(), (name, c)
    print(
        f"[ok] {name}: shape/order kept, {len(cols)} columns hashed, round trip exact, other columns unchanged"
    )

# the folder must hold exactly the five files (hidden entries such as .claude are tool artifacts, not ours)
expected = {name for _, name, _, _ in FILES}
found = {f for f in os.listdir(out_dir) if not f.startswith(".")}
assert found == expected, (
    f"unexpected files in outputs/reviewer_data/: {sorted(found ^ expected)}"
)
print(
    f"[ok] outputs/reviewer_data/ holds exactly the {len(expected)} hashed files"
)

# local lookup (original <-> hash) for you; .secrets/ is gitignored and never shipped. Written last.
pl.DataFrame(
    {
        "hashed": list(crosswalk),
        "original": [v[0] for v in crosswalk.values()],
        "kind": [v[1] for v in crosswalk.values()],
    }
).write_csv(os.path.join(secrets_dir, "crosswalk.csv"))
print(
    f"distinct hashed values: {len(crosswalk)}; lookup written to .secrets/crosswalk.csv (not shipped)"
)
