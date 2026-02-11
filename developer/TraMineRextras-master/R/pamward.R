## runs a pam from the solution in k groups of a hierarchical clustering
## author: Gilbert Ritschard

pamward <- function(diss, k=3, method="ward", dist) {

  TraMineR.check.depr.args(alist(diss = dist))

  if (!inherits(diss, "dist")) {
    if (!isSymmetric(diss)) stop("diss should be a dist object or a symmetric matrix")
    diss <- as.dist(diss)
  }
  clustw <- agnes(diss, diss=T, method=method)
  clw <- cutree(clustw, k=k)
  centers <- disscenter(diss, group=clw, medoids.index="first")
  return <- pam(diss, diss=T, k=k, medoids=centers)
}
