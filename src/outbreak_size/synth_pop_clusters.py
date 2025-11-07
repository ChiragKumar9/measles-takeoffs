import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def fips_to_state(fips: list[str]) -> list[str]:
    fips_to_state = {
        "01": "AL",
        "02": "AK",
        "04": "AZ",
        "05": "AR",
        "06": "CA",
        "08": "CO",
        "09": "CT",
        "10": "DE",
        "11": "DC",
        "12": "FL",
        "13": "GA",
        "15": "HI",
        "16": "ID",
        "17": "IL",
        "18": "IN",
        "19": "IA",
        "20": "KS",
        "21": "KY",
        "22": "LA",
        "23": "ME",
        "24": "MD",
        "25": "MA",
        "26": "MI",
        "27": "MN",
        "28": "MS",
        "29": "MO",
        "30": "MT",
        "31": "NE",
        "32": "NV",
        "33": "NH",
        "34": "NJ",
        "35": "NM",
        "36": "NY",
        "37": "NC",
        "38": "ND",
        "39": "OH",
        "40": "OK",
        "41": "OR",
        "42": "PA",
        "44": "RI",
        "45": "SC",
        "46": "SD",
        "47": "TN",
        "48": "TX",
        "49": "UT",
        "50": "VT",
        "51": "VA",
        "53": "WA",
        "54": "WV",
        "55": "WI",
        "56": "WY",
    }
    states = [fips_to_state[fip[:2]] for fip in fips]
    return states


def get_synth_pop_fips(fips: list[str]) -> pl.DataFrame:
    # get the states
    states = fips_to_state(fips)
    # unique
    unique_states = set(states)
    # query the relevant parquet files
    synth_pops = []
    for state in unique_states:
        synth_pop = pl.read_parquet(
            f"~/Projects/rti_synth_pop/data/processed/2019/{state}_2019_households.parquet"
        )
        synth_pops.append(synth_pop)
    synth_pop = pl.concat(synth_pops)
    # filter to only the fips in question
    # but only do this if the fips provided are at the county level
    synth_pops = []
    for fip in fips:
        if len(fip) == 5:
            synth_pops.append(synth_pop.filter(pl.col("county_fips") == fip))
        else:
            assert len(fip) == 2, (
                "FIPS codes must be either state (2) or county (5) level"
            )
            synth_pops.append(synth_pop.filter(pl.col("state_fips") == fip))
    synth_pop = pl.concat(synth_pops)
    return synth_pop


def summarize_household_size(fips: list[str]) -> tuple[float, float]:
    synth_pop = get_synth_pop_fips(fips)
    # get average household size
    avg_size = synth_pop.select(pl.col("size")).mean()[0, 0]
    std_size = synth_pop.select(pl.col("size")).std()[0, 0]
    return avg_size, std_size


def cluster_fips(
    fips: list[str], n_clusters=10
) -> tuple[float, MiniBatchKMeans]:
    # get the synthetic population for these fips
    synth_pop = get_synth_pop_fips(fips)

    # get the variables we need
    synth_pop = synth_pop.select(
        [
            "hh_age",
            "hh_income",
            "hh_race",
            "size",
            "lat_4326",
            "lon_4326",
        ]
    ).drop_nulls()
    # turn race into a one-hot encoded feature
    synth_pop = synth_pop.to_dummies(
        columns=["hh_race"], separator="_", drop_first=True
    )
    synth_pop = synth_pop.to_numpy()

    scaler = StandardScaler()
    synth_pop = scaler.fit_transform(synth_pop)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=108,
        reassignment_ratio=0.2,
        batch_size=1024,
    )

    kmeans.fit(synth_pop)
    # need a measure of cluster spread
    # that's the variance across cluster centroids
    cluster_centers = kmeans.cluster_centers_
    spread = cluster_centers.var(axis=0).mean()

    # more spread out clusters mean that the data is more clumpy
    # but this clumpiness is moderated by how tight the clusters are
    # if the clusters are super tight (small inertia) then they are even more
    # clumpy
    # we need to normalize the inertia by the number of samples to make it comparable
    clumpiness = spread / (kmeans.inertia_ / synth_pop.shape[0])

    return clumpiness, kmeans
