## Data cleaning
### Outbreaks
- aggregated Detroit up to Wayne County
- at some point, need to do special handling for:
    - East Tennessee
    - Louisiana Health Region 1
    - Fulton and Gwinnett counties being grouped together
    - Delaware providing vaccine data by state, not county
    - Hawaii having whole state data in some years
    - Indiana having whole state data until 2023
    - Kansas -- data by school district
    - Nebraska -- data by health district
    - Alaska -- data by economic zones
    - Delaware providing state level data

- Places that still concern me:
    - Kansas -- what is up with these FIPS codes
    - Hawaii -- it seems like the coverage data must be there?
    - Indiana -- same deal with the coverage data must be there?
    - Alaska -- can we get the exemption data
- Less concerning:
    - What is North Central Oregon?




    ax.hist(coverage_outbreak, bins=np.linspace(70, 100, 1000), alpha=0.5, label=r"Outbreak ($\geq$10 cases)",
            density=True, cumulative=True, color="red")
    ax.hist(coverage_no_outbreak, bins=np.linspace(70, 100, 1000), alpha=0.5, label="No reported transmission",
            density=True, cumulative=True, color="blue")
    # add a dashed line at the herd immunity threshold
    ax.axvline(vaccination_threshold, color="black", linestyle="--")
    ax.set_xlabel("Estimated MMR vaccine coverage (%)")
    ax.set_ylabel("Cumulative density")
    ax.legend()



        # add an inset with a confusion matrix for threshold of 95% and outbreaks
    left, bottom, width, height = [0.3, 0.35, 0.3, 0.3]
    inset_ax = fig.add_axes([left, bottom, width, height])
    # outbreak is 1 if cases >= cases_threshold, 0 if cases are 0, and NA otherwise
    df = df.with_columns(
        pl.when(pl.col("cases_outbreak") >= cases_threshold)
        .then(1)
        .when(pl.col("cases_outbreak") == 0)
        .then(0)
        .otherwise(None)
        .alias("outbreak")
    )
    df_cm = df.select(['outbreak', 'estimated_mmr_cov']).filter(pl.col("outbreak").is_not_null())
    c = ConfusionMatrixDisplay.from_predictions(df_cm['outbreak'].to_list(),
                         (df_cm['estimated_mmr_cov'] >= vaccination_threshold).to_list(),
                         ax=inset_ax,
                         cmap="Greys",
    )
    # chi square test on the confusion matrix
    chi2, p, _, _ = stats.chi2_contingency(c.confusion_matrix) # type: ignore
    print(f"Chi-square statistic: {chi2}, p-value: {p}")
    # make the labels meaningful
    inset_ax.set_xlabel(r"Coverage $\geq$ 95%")
    inset_ax.set_ylabel("Outbreak size")
    inset_ax.set_xticklabels(["No", "Yes"])
    inset_ax.set_yticklabels(["0", r"$\geq$10"])
    cbar = c.im_.colorbar
    cbar.set_ticks([0, 50, 100]) # type: ignore
    cbar.set_label("County-years", # type: ignore
                   rotation=270,
                   labelpad=23)






def plot_validation(df, import_df, fig, ax):
    # we have validation data for 2023, 2024, and 2025
    projection_years = [2023, 2024, 2025]
    dfs_by_year = {}
    method = 'dense'
    for year in projection_years:
        df_year = df.filter(pl.col("year") == year)
        # calculate the rank order of the samples
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{year}.pkl",
            ),
            "rb",
        ) as file:
            projections = pickle.load(file)
        # multiply the projections against the import samples
        # extract the import samples for this year
        df_year = df_year.join(
            import_df.select(
                [
                    "county", # rather than fips because we have some regions
                    'state',
                    'year',
                    "importation_probability_samples",
                ]
            ),
            on=["county", 'state', "year"],
            how="left",
        )
        import_probability_samples = np.vstack(
            df_year["importation_probability_samples"].to_numpy()
        )  # type: ignore
        projections = projections * import_probability_samples.T

        ranked_projections = rankdata(projections, method=method, axis=1) - 1
        percentile_projections = 100 * ranked_projections / (ranked_projections.max(axis = 1, keepdims=True) + 1)

        df_year = df_year.with_columns(
            median_risk_rank=pl.Series(
                np.median(percentile_projections, axis=0).__array__()
            ),
            lower_risk_rank=pl.Series(
                np.percentile(percentile_projections, 2.5, method = 'lower', axis=0).__array__()
            ),
            upper_risk_rank=pl.Series(
                np.percentile(percentile_projections, 97.5, method = 'higher', axis=0).__array__()
            ),
        )

        df_year = df_year.with_columns(
            year_specific_rank_order=pl.col("cases").rank(
                method,
                descending=False
            ) - 1
        ).with_columns(
            year_specific_percentile=pl.col("year_specific_rank_order")
            / (pl.max("year_specific_rank_order") + 1)
            * 100
        )

        # ask whether the 2.5th to 97.5th percentile contains the observed rank
        df_year = df_year.with_columns(
            contains_observed_rank=pl.when(
                (pl.col("lower_risk_rank") <= pl.col("year_specific_percentile"))
                & (pl.col("upper_risk_rank") >= pl.col("year_specific_percentile"))
            )
            .then(True)
            .otherwise(False),
            observed_rank_above = pl.when(pl.col("year_specific_percentile") > pl.col("upper_risk_rank")).then(True).otherwise(False),
            observed_rank_below = pl.when(pl.col("year_specific_percentile") < pl.col("lower_risk_rank")).then(True).otherwise(False),
        )
        # # if this is true, then we have a successful prediction
        # # we want to circumvent the issue of rank ordering biasing towards the
        # # center for values that are tied
        df_year = df_year.with_columns(
            adjusted_risk_rank = pl.when("contains_observed_rank")
            .then(pl.col("year_specific_percentile"))
            .otherwise(pl.when("observed_rank_above").then(pl.col("upper_risk_rank")).otherwise(pl.col("lower_risk_rank")))
        )

        dfs_by_year[year] = df_year
        print(df_year.filter(~pl.col("contains_observed_rank"))[['county', 'state', 'cases', 'median_risk_rank', 'lower_risk_rank', 'upper_risk_rank', 'year_specific_percentile']])

    # boxplots by chunk of risk quintiles
    num_chunks = 5
    outbreaks = {2023: [], 2024: [], 2025: []}
    for tile in range(num_chunks):
        # want the outbreak size data for each chunk
        percentile_lower = tile * (100 / num_chunks)
        percentile_upper = (tile + 1) * (100 / num_chunks)

        # take the data for each year from that chunk
        for year in projection_years:
            outbreaks[year].append(
                dfs_by_year[year].filter(
                    (pl.col("adjusted_risk_rank") >= percentile_lower)
                    & (pl.col("adjusted_risk_rank") < percentile_upper)
                )['cases'].to_list()
            )

    offsets = np.linspace(-0.35, 0.35, num_chunks)
    cmap = plt.get_cmap("Reds")
    cmap = [cmap(i) for i in np.linspace(0.3, 1.0, num_chunks)]
    for tile in range(num_chunks):
        data = [outbreaks[year][tile] for year in outbreaks.keys()]
        ax.boxplot(
            data,
            positions=np.array([2023, 2024, 2025]) + offsets[tile],
            orientation='vertical',
            whis=(2.5, 97.5),
            widths=0.15,
            showmeans=True,
            boxprops=dict(color=cmap[tile], linewidth = 2),
            medianprops=dict(color='black', linewidth = 2),
            meanprops=dict(markeredgecolor=cmap[tile], markerfacecolor=cmap[tile], marker='*', markersize=12),
            whiskerprops=dict(color=cmap[tile]),
            capprops=dict(color=cmap[tile]),
            flierprops=dict(markeredgecolor=cmap[tile], markersize=8),
        )

    ax.set_yscale("log")
    ax.set_ylim(1)

    minor_y_ticks = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        200,
        300,
        400,
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))
    ax.xaxis.set_minor_locator(tck.FixedLocator([year + offset for year in projection_years for offset in offsets]))

    ax.set_yticks([1, 10, 100], labels=[1, 10, 100])
    ax.set_ylabel("Reported cases (N)")
    ax.set_xlabel("Year")

    patches = [mpatches.Patch(edgecolor=color, linewidth=2, fill=False, label=label)
           for color, label in zip(cmap, [f"{i*100/num_chunks:.0f}%-{(i+1)*100/num_chunks:.0f}%" for i in range(num_chunks)])]

    ax.legend(handles=patches)
    ax.set_xticks([2023, 2024, 2025], labels=[2023, 2024, 2025])

    return fig, ax
