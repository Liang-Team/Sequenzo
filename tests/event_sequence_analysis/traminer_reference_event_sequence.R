# TraMineR reference for event sequence analysis (LSOG / dyadic_children)
#
# Usage: Rscript traminer_reference_event_sequence.R <path_to_dyadic_children.csv> [nrows] [outdir]
# Example: Rscript traminer_reference_event_sequence.R ../../../sequenzo/datasets/dyadic_children.csv 20 .
#
# Writes:
#   ref_eseq_meta.csv          - n_sequences, n_events
#   ref_eseq_alphabet.csv      - Event alphabet (order for Python alignment)
#   ref_eseq_fsub_support.csv - Support and Count for frequent subsequences (seqefsub)
#   ref_eseq_applysub.csv      - Presence matrix (seqeapplysub method="presence")

if (!requireNamespace("TraMineR", quietly = TRUE)) {
  stop("TraMineR required: install.packages('TraMineR')")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript traminer_reference_event_sequence.R <csv_path> [nrows=20] [outdir=.]")
csv_path <- args[1]
nrows <- if (length(args) >= 2) as.integer(args[2]) else 20L
outdir <- if (length(args) >= 3) args[3] else "."

dat <- read.csv(csv_path, nrows = nrows, check.names = FALSE)
time_cols <- names(dat)[sapply(names(dat), function(x) suppressWarnings(!is.na(as.numeric(x))))]
time_cols <- time_cols[order(as.numeric(time_cols))]
id_col <- "dyadID"
states <- c(1, 2, 3, 4, 5, 6)

# State sequence object (same as other lsog tests)
seqdata <- TraMineR::seqdef(dat[, time_cols, drop = FALSE], alphabet = states, id = dat[[id_col]])

# Event sequence from state sequence (transition method)
eseq <- TraMineR::seqecreate(seqdata, tevent = "transition")

# Frequent subsequences with min.support = 2
fsub <- TraMineR::seqefsub(eseq, min.support = 2)

# Export Support and Count
if (length(fsub$subseq) > 0) {
  subseq_str <- as.character(fsub$subseq)
  df_support <- data.frame(
    Subseq = subseq_str,
    Support = fsub$data$Support,
    Count = fsub$data$Count,
    stringsAsFactors = FALSE
  )
  write.csv(df_support, file.path(outdir, "ref_eseq_fsub_support.csv"), row.names = FALSE)

  # Presence matrix (seqeapplysub method="presence")
  pres <- TraMineR::seqeapplysub(fsub, method = "presence")
  write.csv(pres, file.path(outdir, "ref_eseq_applysub.csv"), row.names = TRUE)
} else {
  # No subsequences found
  write.csv(data.frame(Subseq = character(), Support = double(), Count = integer()),
            file.path(outdir, "ref_eseq_fsub_support.csv"), row.names = FALSE)
  write.csv(matrix(0, nrow = length(eseq), ncol = 0),
            file.path(outdir, "ref_eseq_applysub.csv"), row.names = TRUE)
}

# Metadata for test
meta <- data.frame(
  n_sequences = length(eseq),
  n_events = length(TraMineR::alphabet(eseq))
)
write.csv(meta, file.path(outdir, "ref_eseq_meta.csv"), row.names = FALSE)

# Event alphabet (order matters for matching Python dictionary)
alph <- TraMineR::alphabet(eseq)
write.csv(data.frame(event = alph), file.path(outdir, "ref_eseq_alphabet.csv"), row.names = FALSE)
