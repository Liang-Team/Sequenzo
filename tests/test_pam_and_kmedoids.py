import numpy as np

from sequenzo.clustering import KMedoids


def test_pam_kmedoids_and_pamonce_smoke():
    diss = np.array(
        [
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 5.0, 6.0],
            [5.0, 5.0, 0.0, 1.0],
            [6.0, 6.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    for method in ["PAM", "KMedoids", "PAMonce"]:
        labels = KMedoids(
            method=method,
            k=2,
            initialclust=[0, 2],
            npass=1,
            diss=diss,
            verbose=False,
        )

        np.testing.assert_array_equal(labels, np.array([1, 1, 3, 3]))
