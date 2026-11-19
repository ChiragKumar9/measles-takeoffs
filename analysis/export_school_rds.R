# Export the per-state school files released with Chen & Bento (Nature Medicine
# 2026; PMID 42613442; Zenodo 10.5281/zenodo.19430274 and 10.5281/zenodo.20174071)
# to one CSV that Python reads: data/schools/school_points_raw.csv.
#
# Nothing is cleaned here. Each row of a state's file becomes a row with the
# state (from the file name), the year label, the enrollment and the MMR value as
# they are (text), the point of the school (x, y) and its coordinate system.
# Rows without a point cannot be placed and are not exported (the count is
# printed). Python does the rest (analysis/assemble_school_vaccination.py); the R
# package sf does not load on this machine, so this script uses base R only.
#
# Run from the repo root:  Rscript --vanilla analysis/export_school_rds.R

data_dir <- "data/schools"

read_state_file <- function(path) {
  state <- sub("_school\\.rds$", "", basename(path))
  x <- readRDS(path)
  crs <- attr(x$geometry, "crs")$input # read before subsetting, which drops attributes
  has_point <- vapply(x$geometry, length, integer(1)) == 2
  xy <- do.call(rbind, lapply(x$geometry[has_point], function(g) as.numeric(g)[1:2]))
  cat(sprintf("%-3s rows %6d  without a point (not exported) %d  crs %s\n", state, nrow(x), sum(!has_point), crs))
  data.frame(
    state = state,
    year = gsub("/", "-", as.character(x$year[has_point])),
    enrollment = as.character(x$enrollment[has_point]),
    mmr = as.character(x$MMR[has_point]),
    x = xy[, 1], y = xy[, 2], crs = crs,
    stringsAsFactors = FALSE
  )
}

files <- sort(list.files(data_dir, pattern = "_school\\.rds$", full.names = TRUE))
d <- do.call(rbind, lapply(files, read_state_file))
path <- file.path(data_dir, "school_points_raw.csv")
write.csv(d, path, row.names = FALSE, na = "")
cat("wrote", path, nrow(d), "rows\n")
