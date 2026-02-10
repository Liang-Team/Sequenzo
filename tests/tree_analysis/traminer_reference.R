#!/usr/bin/env Rscript
# Generate TraMineR reference results for tree analysis comparison
# Usage: Rscript tests/tree_analysis/traminer_reference.R

library(TraMineR)

# Set working directory to project root
setwd("/Users/lei/Documents/Sequenzo_all_folders/Sequenzo")

# Load lsog data (dyadic_children)
# Use check.names=FALSE to preserve numeric column names (15, 16, etc.)
df <- read.csv("sequenzo/datasets/dyadic_children.csv", stringsAsFactors = FALSE, check.names = FALSE)
df <- df[1:20, ]  # First 20 rows for testing

# Extract time columns (numeric columns)
# Columns that are pure numbers (like "15", "16", etc.)
# Handle both "15" and "X15" formats
time_cols <- names(df)[sapply(names(df), function(x) {
    # Remove "X" prefix if present (R adds this for numeric column names)
    clean_x <- sub("^X", "", x)
    # Check if column name is a pure number
    num_val <- suppressWarnings(as.numeric(clean_x))
    !is.na(num_val) && clean_x != "" && x != "dyadID" && x != "sex"
})]

# Extract numeric values and sort
time_nums <- sort(as.numeric(sub("^X", "", time_cols)))
time_cols <- as.character(time_nums)  # Use clean numeric names

# If R added "X" prefix, use those column names
if (any(paste0("X", time_cols) %in% names(df))) {
    time_cols <- paste0("X", time_cols)
}

cat(sprintf("Found %d time columns: %s ... %s\n", 
            length(time_cols), time_cols[1], time_cols[length(time_cols)]))

# Create sequence object
# Note: seqdef needs proper alphabet specification
seqdata <- seqdef(df[, time_cols], alphabet = 1:6, states = c("1", "2", "3", "4", "5", "6"))

cat("=== TraMineR Reference Results for Tree Analysis ===\n\n")

# ============================================================================
# Test 1: dissvar (compute_pseudo_variance)
# ============================================================================
cat("Test 1: dissvar (unweighted)\n")
dist_matrix <- seqdist(seqdata, method = "LCS", norm = "auto")
variance_unweighted <- dissvar(dist_matrix, weights = NULL, squared = FALSE)
cat(sprintf("Unweighted variance: %.10f\n", variance_unweighted))

cat("\nTest 1b: dissvar (weighted)\n")
weights <- rep(2.0, nrow(seqdata))
variance_weighted <- dissvar(dist_matrix, weights = weights, squared = FALSE)
cat(sprintf("Weighted variance (all weights=2): %.10f\n", variance_weighted))

cat("\nTest 1c: dissvar (squared)\n")
variance_squared <- dissvar(dist_matrix, weights = NULL, squared = TRUE)
cat(sprintf("Squared variance: %.10f\n", variance_squared))

# Save results
write.csv(data.frame(
    test = c("unweighted", "weighted", "squared"),
    variance = c(variance_unweighted, variance_weighted, variance_squared)
), "tests/tree_analysis/ref_dissvar.csv", row.names = FALSE)

# ============================================================================
# Test 2: dissassoc (compute_distance_association)
# ============================================================================
cat("\nTest 2: dissassoc\n")
groups <- factor(rep(c("A", "B"), each = 10))

# Run dissassoc
assoc_result <- dissassoc(dist_matrix, groups, weights = NULL, R = 10, 
                          weight.permutation = "none", squared = FALSE)

# Extract results from stat table
# The stat table has row names: "Pseudo F", "Pseudo Fbf", "Pseudo R2", "Bartlett", "Levene"
# Columns are: "t0" (test statistic) and "p.value"
pseudo_f <- assoc_result$stat["Pseudo F", "t0"]
pseudo_r2 <- assoc_result$stat["Pseudo R2", "t0"]
pseudo_f_pval <- assoc_result$stat["Pseudo F", "p.value"]

cat(sprintf("Pseudo F: %.10f\n", pseudo_f))
cat(sprintf("Pseudo R²: %.10f\n", pseudo_r2))
cat(sprintf("Pseudo F p-value: %.10f\n", pseudo_f_pval))

write.csv(data.frame(
    pseudo_f = pseudo_f,
    pseudo_r2 = pseudo_r2,
    pseudo_f_pval = pseudo_f_pval
), "tests/tree_analysis/ref_dissassoc.csv", row.names = FALSE)

# ============================================================================
# Test 3: disstree (build_distance_tree)
# ============================================================================
cat("\nTest 3: disstree\n")

# Set seed for reproducibility BEFORE generating random numbers
set.seed(12345)

predictors <- data.frame(
    group = groups,
    numeric_var = rnorm(20)  # Random numeric variable
)

# Convert distance matrix to dist object for disstree
dist_obj <- as.dist(dist_matrix)

tree <- disstree(dist_obj ~ group + numeric_var, data = predictors,
                 weights = NULL, min.size = 0.1, max.depth = 3, R = 10,
                 pval = 0.1, weight.permutation = "none", squared = FALSE)

cat("Tree structure:\n")
print(tree)

# Get leaf memberships
leaf_ids <- disstreeleaf(tree)
cat(sprintf("\nNumber of leaves: %d\n", length(unique(leaf_ids))))
cat(sprintf("Leaf distribution:\n"))
print(table(leaf_ids))

# Save tree structure
tree_info <- data.frame(
    n_leaves = length(unique(leaf_ids)),
    total_n = tree$info$n,
    min_size = tree$info$parameters$min.size,
    max_depth = tree$info$parameters$max.depth
)
write.csv(tree_info, "tests/tree_analysis/ref_disstree_info.csv", row.names = FALSE)

# Save leaf memberships
write.csv(data.frame(leaf_id = leaf_ids), 
          "tests/tree_analysis/ref_disstree_leaves.csv", row.names = FALSE)

# Get classification rules (disstree.get.rules)
cat("\nClassification rules:\n")
rules <- disstree.get.rules(tree, collapse = "; ")
cat(sprintf("Number of rules: %d\n", length(rules)))
for (i in 1:min(3, length(rules))) {
    cat(sprintf("Rule %d: %s\n", i, rules[i]))
}

# Save classification rules
write.csv(data.frame(
    rule_id = 1:length(rules),
    rule = rules
), "tests/tree_analysis/ref_disstree_rules.csv", row.names = FALSE)

# ============================================================================
# Test 4: seqtree (build_sequence_tree)
# ============================================================================
cat("\nTest 4: seqtree\n")
set.seed(12345)

seqtree_result <- seqtree(seqdata ~ group + numeric_var, data = predictors,
                          weighted = TRUE, min.size = 0.1, max.depth = 3, R = 10,
                          pval = 0.1, weight.permutation = "replicate",
                          seqdist.args = list(method = "LCS", norm = "auto"))

cat("Sequence tree structure:\n")
print(seqtree_result)

seqtree_leaf_ids <- seqtree_result$fitted[, 1]
cat(sprintf("\nNumber of leaves: %d\n", length(unique(seqtree_leaf_ids))))

write.csv(data.frame(leaf_id = seqtree_leaf_ids),
          "tests/tree_analysis/ref_seqtree_leaves.csv", row.names = FALSE)

# Get classification rules for seqtree (if available)
# Note: seqtree uses the same disstree structure, so we can use disstree.get.rules
cat("\nSequence tree classification rules:\n")
seqtree_rules <- disstree.get.rules(seqtree_result, collapse = "; ")
cat(sprintf("Number of rules: %d\n", length(seqtree_rules)))
for (i in 1:min(3, length(seqtree_rules))) {
    cat(sprintf("Rule %d: %s\n", i, seqtree_rules[i]))
}

write.csv(data.frame(
    rule_id = 1:length(seqtree_rules),
    rule = seqtree_rules
), "tests/tree_analysis/ref_seqtree_rules.csv", row.names = FALSE)

cat("\n=== Reference generation complete ===\n")
cat("Reference files saved to tests/tree_analysis/\n")
cat("Generated files:\n")
cat("  - ref_dissvar.csv\n")
cat("  - ref_dissassoc.csv\n")
cat("  - ref_disstree_info.csv\n")
cat("  - ref_disstree_leaves.csv\n")
cat("  - ref_disstree_rules.csv\n")
cat("  - ref_seqtree_leaves.csv\n")
cat("  - ref_seqtree_rules.csv\n")
