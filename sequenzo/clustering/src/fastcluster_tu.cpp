/*
 * fastcluster_tu.cpp — DEPRECATED
 *
 * This file previously compiled both distance_prep_utils and fastcluster_linkage
 * together WITHOUT -ffast-math.  It has been split into two separate TUs:
 *
 *   distance_prep_tu.cpp        — compiled WITHOUT -ffast-math (IEEE NaN handling)
 *   fastcluster_linkage_tu.cpp  — compiled WITH    -ffast-math (linkage performance)
 *
 * This stub remains only for backward compatibility with build systems that have
 * not yet been updated.  It includes both TUs so nothing breaks, but ideally
 * setuptools / CMake should reference the two new files instead.
 */
#include "distance_prep_tu.cpp"
#include "fastcluster_linkage_tu.cpp"
