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
