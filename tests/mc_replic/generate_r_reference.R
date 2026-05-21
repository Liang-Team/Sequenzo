# Generate reference outputs (sources developer MCseqReplic R files + TraMineR).
# Run: Rscript tests/mc_replic/generate_r_reference.R

suppressPackageStartupMessages(library(TraMineR))

pkg_root <- Sys.getenv("MCSEQREPLIC_PKG")
if (!nzchar(pkg_root)) {
  pkg_root <- file.path("developer", "MCseqReplic")
}
r_dir <- file.path(pkg_root, "R")
for (f in c("MCchgmeth.R", "MCpj.R", "MCseqReplicate.R", "MCdisslist.R", "MCseqdistSE.R", "MCmisc.R", "MCudist.R")) {
  source(file.path(r_dir, f), local = FALSE)
}

out_dir <- "tests/mc_replic/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

exdata <- read.table(text = "
                a a b b
                a a b b
                b b a a
                a c c b
                b b a c
                b b a c
                ")
weights <- rep(1, nrow(exdata))
s.exdata <- TraMineR::seqdef(exdata, weights = weights, id = paste0("id", 1:nrow(exdata)))

set.seed(25)
altseq <- MCseqReplicate(s.exdata, J = 1, R = 3, model = "keep.dss")
write.csv(as.data.frame(altseq[[1]]), file.path(out_dir, "replicate_r1.csv"), row.names = TRUE)

disslist <- MCdisslist(altseq, method = "HAM", full.matrix = TRUE)
write.csv(as.matrix(disslist[[1]]), file.path(out_dir, "diss_r1_ham.csv"), row.names = TRUE)

alt_obs <- MCseqReplicate(s.exdata, J = 1, R = 3, include.obs = TRUE)
mcd <- MCseqdistSE(dissrepl = disslist)
write.csv(as.matrix(mcd$MC.mean), file.path(out_dir, "mc_mean_ham.csv"), row.names = TRUE)
write.csv(as.matrix(mcd$MC.sd), file.path(out_dir, "mc_sd_ham.csv"), row.names = TRUE)

pj <- MCpj(Emean = 1.2, pzero = 0.4)
write.csv(data.frame(pj = pj), file.path(out_dir, "mc_pj.csv"), row.names = FALSE)

u <- s.exdata[1, ]
sd <- TraMineR::seqdur(u)[1, ]
set.seed(42)
sd2 <- ch.dur(sd, Jprob = 1)
write.csv(data.frame(dur = sd2), file.path(out_dir, "ch_dur_j1.csv"), row.names = FALSE)

message("Reference files written to ", out_dir)
