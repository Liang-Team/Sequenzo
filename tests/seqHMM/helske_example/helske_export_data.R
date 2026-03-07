## ============================================================================
## helske_export_data.R
## cd tests/seqHMM && Rscript helske_export_data.R
## ============================================================================
library("TraMineR"); library("seqHMM")

data("biofam", package = "TraMineR")
biofam_seq <- seqdef(biofam[, 10:25], start = 15, labels = c(
  "parent","left","married","left+marr","child",
  "left+child","left+marr+ch","divorced"))

seq_mat <- as.matrix(biofam_seq)
lmap <- setNames(attr(biofam_seq,"labels"), attr(biofam_seq,"alphabet"))
df <- as.data.frame(seq_mat, stringsAsFactors=FALSE)
for (col in names(df)) df[[col]] <- lmap[df[[col]]]
colnames(df) <- paste0("age_", 15:30)
df <- cbind(id=rownames(seq_mat), df)
write.csv(df, "biofam_seq.csv", row.names=FALSE)

data("biofam3c", package = "seqHMM")
export_ch <- function(raw, alph, fname) {
  s <- seqdef(raw, start=15, alphabet=alph)
  m <- as.matrix(s); mp <- setNames(attr(s,"labels"), attr(s,"alphabet"))
  d <- as.data.frame(m, stringsAsFactors=FALSE)
  for (col in names(d)) d[[col]] <- mp[d[[col]]]
  colnames(d) <- paste0("age_", 15:30)
  write.csv(cbind(id=rownames(m), d), fname, row.names=FALSE)
}
export_ch(biofam3c$married,  c("single","married","divorced"),  "biofam3c_married.csv")
export_ch(biofam3c$children, c("childless","children"),         "biofam3c_children.csv")
export_ch(biofam3c$left,     c("with parents","left home"),     "biofam3c_left.csv")

sc_emiss <- matrix(NA, nrow=5, ncol=8)
sc_emiss[1,] <- seqstatf(biofam_seq[, 1:4 ])[,2] + 0.1
sc_emiss[2,] <- seqstatf(biofam_seq[, 5:7 ])[,2] + 0.1
sc_emiss[3,] <- seqstatf(biofam_seq[, 8:10])[,2] + 0.1
sc_emiss[4,] <- seqstatf(biofam_seq[,11:13])[,2] + 0.1
sc_emiss[5,] <- seqstatf(biofam_seq[,14:16])[,2] + 0.1
sc_emiss <- sc_emiss / rowSums(sc_emiss)
write.csv(sc_emiss, "ref_sc_emiss_init.csv", row.names=FALSE)

sc_init <- c(0.9,0.06,0.02,0.01,0.01)
sc_trans <- matrix(c(0.80,0.10,0.05,0.03,0.02, 0.02,0.80,0.10,0.05,0.03,
  0.02,0.03,0.80,0.10,0.05, 0.02,0.03,0.05,0.80,0.10,
  0.02,0.03,0.05,0.05,0.85), nrow=5, byrow=TRUE)
sc_mod <- build_hmm(observations=biofam_seq, initial_probs=sc_init,
  transition_probs=sc_trans, emission_probs=sc_emiss)
sc_fit <- fit_model(sc_mod)
cat("R SC logLik:", sc_fit$logLik, "\n")

marr_seq  <- seqdef(biofam3c$married,  start=15, alphabet=c("single","married","divorced"))
child_seq <- seqdef(biofam3c$children, start=15, alphabet=c("childless","children"))
left_seq  <- seqdef(biofam3c$left,     start=15, alphabet=c("with parents","left home"))
mc_init <- c(0.9,0.05,0.02,0.02,0.01)
mc_trans <- matrix(c(0.80,0.10,0.05,0.03,0.02, 0,0.90,0.05,0.03,0.02,
  0,0,0.90,0.07,0.03, 0,0,0,0.90,0.10, 0,0,0,0,1), nrow=5, byrow=TRUE)
mc_em <- list(
  matrix(c(0.90,0.05,0.05, 0.90,0.05,0.05, 0.05,0.90,0.05,
    0.05,0.90,0.05, 0.30,0.30,0.40), nrow=5, byrow=TRUE),
  matrix(c(0.9,0.1, 0.9,0.1, 0.1,0.9, 0.1,0.9, 0.5,0.5), nrow=5, byrow=TRUE),
  matrix(c(0.9,0.1, 0.1,0.9, 0.1,0.9, 0.1,0.9, 0.5,0.5), nrow=5, byrow=TRUE))
mc_mod <- build_hmm(observations=list(marr_seq, child_seq, left_seq),
  initial_probs=mc_init, transition_probs=mc_trans, emission_probs=mc_em,
  channel_names=c("Marriage","Parenthood","Residence"))
mc_fit <- fit_model(mc_mod, em_step=FALSE, local_step=TRUE, threads=4)
mc_ll <- as.numeric(logLik(mc_fit$model)); mc_bic <- BIC(mc_fit$model)
cat("R MC logLik:", mc_ll, " BIC:", mc_bic, "\n")

write.csv(data.frame(key=c("sc_loglik","mc_loglik","mc_bic"),
  value=c(sc_fit$logLik, mc_ll, mc_bic)), "ref_results.csv", row.names=FALSE)
cat("Done.\n")
