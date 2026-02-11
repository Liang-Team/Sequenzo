#!/usr/bin/env Rscript
# Generate TraMineR reference results for compare_differences comparison
# Usage: Rscript tests/compare_differences/traminer_reference.R

library(TraMineR)
library(TraMineRextras)

# Set working directory to project root
setwd("/Users/lei/Documents/Sequenzo_all_folders/Sequenzo")

# Load lsog data (dyadic_children)
# Use check.names=FALSE to preserve numeric column names (15, 16, etc.)
df <- read.csv("sequenzo/datasets/dyadic_children.csv", stringsAsFactors = FALSE, check.names = FALSE)
df <- df[1:20, ]  # First 20 rows for testing

# Extract time columns (numeric columns)
time_cols <- names(df)[sapply(names(df), function(x) {
    clean_x <- sub("^X", "", x)
    num_val <- suppressWarnings(as.numeric(clean_x))
    !is.na(num_val) && clean_x != "" && x != "dyadID" && x != "sex"
})]

# Extract numeric values and sort
time_nums <- sort(as.numeric(sub("^X", "", time_cols)))
time_cols <- as.character(time_nums)

# If R added "X" prefix, use those column names
if (any(paste0("X", time_cols) %in% names(df))) {
    time_cols <- paste0("X", time_cols)
}

cat(sprintf("Found %d time columns: %s ... %s\n", 
            length(time_cols), time_cols[1], time_cols[length(time_cols)]))

# Create sequence object
seqdata <- seqdef(df[, time_cols], alphabet = 1:6, states = c("1", "2", "3", "4", "5", "6"))

cat("=== TraMineR Reference Results for Compare Differences ===\n\n")

# Create grouping variable (split into two groups)
group <- factor(rep(c("A", "B"), each = 10))

# ============================================================================
# Test 1: seqdiff (compare_groups_across_positions)
# ============================================================================
cat("Test 1: seqdiff with cmprange=(0, 1), method='LCS', norm='auto'\n")
set.seed(12345)  # For reproducibility
diff_result_1 <- seqdiff(seqdata, group = group, cmprange = c(0, 1),
                         seqdist.args = list(method = "LCS", norm = "auto"),
                         with.missing = FALSE, weighted = TRUE, squared = FALSE)

# Extract statistics
stat_df <- diff_result_1$stat
write.csv(stat_df, "tests/compare_differences/ref_seqdiff_stat_cmprange01.csv", row.names = FALSE)

# Extract discrepancy
discrepancy_df <- diff_result_1$discrepancy
write.csv(discrepancy_df, "tests/compare_differences/ref_seqdiff_discrepancy_cmprange01.csv", row.names = FALSE)

cat("Saved: ref_seqdiff_stat_cmprange01.csv, ref_seqdiff_discrepancy_cmprange01.csv\n")

# Test with different cmprange
cat("\nTest 1b: seqdiff with cmprange=(-2, 2), method='LCS', norm='auto'\n")
diff_result_2 <- seqdiff(seqdata, group = group, cmprange = c(-2, 2),
                         seqdist.args = list(method = "LCS", norm = "auto"),
                         with.missing = FALSE, weighted = TRUE, squared = FALSE)

stat_df_2 <- diff_result_2$stat
write.csv(stat_df_2, "tests/compare_differences/ref_seqdiff_stat_cmprange-22.csv", row.names = FALSE)

discrepancy_df_2 <- diff_result_2$discrepancy
write.csv(discrepancy_df_2, "tests/compare_differences/ref_seqdiff_discrepancy_cmprange-22.csv", row.names = FALSE)

cat("Saved: ref_seqdiff_stat_cmprange-22.csv, ref_seqdiff_discrepancy_cmprange-22.csv\n")

# Test with different method
cat("\nTest 1c: seqdiff with cmprange=(0, 1), method='OM', norm='auto'\n")
# OM method requires sm (substitution matrix), use TRATE method to generate it
diff_result_3 <- seqdiff(seqdata, group = group, cmprange = c(0, 1),
                         seqdist.args = list(method = "OM", norm = "auto", sm = "TRATE"),
                         with.missing = FALSE, weighted = TRUE, squared = FALSE)

stat_df_3 <- diff_result_3$stat
write.csv(stat_df_3, "tests/compare_differences/ref_seqdiff_stat_OM.csv", row.names = FALSE)

discrepancy_df_3 <- diff_result_3$discrepancy
write.csv(discrepancy_df_3, "tests/compare_differences/ref_seqdiff_discrepancy_OM.csv", row.names = FALSE)

cat("Saved: ref_seqdiff_stat_OM.csv, ref_seqdiff_discrepancy_OM.csv\n")

# Test with squared=TRUE
cat("\nTest 1d: seqdiff with squared=TRUE\n")
diff_result_4 <- seqdiff(seqdata, group = group, cmprange = c(0, 1),
                         seqdist.args = list(method = "LCS", norm = "auto"),
                         with.missing = FALSE, weighted = TRUE, squared = TRUE)

stat_df_4 <- diff_result_4$stat
write.csv(stat_df_4, "tests/compare_differences/ref_seqdiff_stat_squared.csv", row.names = FALSE)

cat("Saved: ref_seqdiff_stat_squared.csv\n")

# ============================================================================
# Test 2: seqCompare (compare_groups_overall)
# ============================================================================
cat("\nTest 2: seqCompare with method='LCS', s=100, stat='all'\n")
set.seed(36963)  # Default seed from TraMineRextras
compare_result_1 <- seqCompare(seqdata, group = group, s = 100, seed = 36963,
                                stat = "all", squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(compare_result_1, "tests/compare_differences/ref_seqCompare_LCS_all.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_LCS_all.csv\n")

# Test with stat='LRT' only
cat("\nTest 2b: seqCompare with stat='LRT' only\n")
compare_result_2 <- seqCompare(seqdata, group = group, s = 100, seed = 36963,
                                stat = "LRT", squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(compare_result_2, "tests/compare_differences/ref_seqCompare_LCS_LRT.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_LCS_LRT.csv\n")

# Test with stat='BIC' only
cat("\nTest 2c: seqCompare with stat='BIC' only\n")
compare_result_3 <- seqCompare(seqdata, group = group, s = 100, seed = 36963,
                                stat = "BIC", squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(compare_result_3, "tests/compare_differences/ref_seqCompare_LCS_BIC.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_LCS_BIC.csv\n")

# Test with method='OM'
cat("\nTest 2d: seqCompare with method='OM'\n")
# OM method requires sm (substitution matrix), use TRATE method to generate it
compare_result_4 <- seqCompare(seqdata, group = group, s = 100, seed = 36963,
                                stat = "all", squared = "LRTonly", weighted = TRUE, 
                                method = "OM", sm = "TRATE")

write.csv(compare_result_4, "tests/compare_differences/ref_seqCompare_OM_all.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_OM_all.csv\n")

# Test with s=0 (no sampling)
cat("\nTest 2e: seqCompare with s=0 (no sampling)\n")
compare_result_5 <- seqCompare(seqdata, group = group, s = 0, seed = 36963,
                                stat = "all", squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(compare_result_5, "tests/compare_differences/ref_seqCompare_s0.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_s0.csv\n")

# Test with different s value
cat("\nTest 2f: seqCompare with s=50\n")
compare_result_6 <- seqCompare(seqdata, group = group, s = 50, seed = 36963,
                                stat = "all", squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(compare_result_6, "tests/compare_differences/ref_seqCompare_s50.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_s50.csv\n")

# Test with weighted=FALSE
cat("\nTest 2g: seqCompare with weighted=FALSE\n")
compare_result_7 <- seqCompare(seqdata, group = group, s = 100, seed = 36963,
                                stat = "all", squared = "LRTonly", weighted = FALSE, method = "LCS")

write.csv(compare_result_7, "tests/compare_differences/ref_seqCompare_unweighted.csv", row.names = FALSE)
cat("Saved: ref_seqCompare_unweighted.csv\n")

# ============================================================================
# Test 3: seqLRT (compute_likelihood_ratio_test)
# ============================================================================
cat("\nTest 3: seqLRT with method='LCS', s=100\n")
set.seed(36963)
lrt_result <- seqLRT(seqdata, group = group, s = 100, seed = 36963,
                     squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(lrt_result, "tests/compare_differences/ref_seqLRT_LCS.csv", row.names = FALSE)
cat("Saved: ref_seqLRT_LCS.csv\n")

# ============================================================================
# Test 4: seqBIC (compute_bayesian_information_criterion_test)
# ============================================================================
cat("\nTest 4: seqBIC with method='LCS', s=100\n")
set.seed(36963)
bic_result <- seqBIC(seqdata, group = group, s = 100, seed = 36963,
                     squared = "LRTonly", weighted = TRUE, method = "LCS")

write.csv(bic_result, "tests/compare_differences/ref_seqBIC_LCS.csv", row.names = FALSE)
cat("Saved: ref_seqBIC_LCS.csv\n")

cat("\n=== All reference files generated successfully! ===\n")
