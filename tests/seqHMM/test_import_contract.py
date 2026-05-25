import os
import subprocess
import sys


def test_package_import_keeps_clustering_lazy():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    code = (
        "import sys\n"
        "import sequenzo\n"
        "print('sequenzo.clustering' in sys.modules)\n"
        "from sequenzo import SequenceData\n"
        "print(SequenceData.__name__)\n"
        "print('sequenzo.clustering' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stdout.strip().splitlines() == [
        "False",
        "SequenceData",
        "False",
    ]


def test_seqhmm_utilities_are_available_from_top_level_package():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    code = (
        "import sys\n"
        "from sequenzo import hidden_paths, trim_model, data_to_stslist\n"
        "print(hidden_paths.__name__)\n"
        "print(trim_model.__name__)\n"
        "print(data_to_stslist.__name__)\n"
        "print('sequenzo.clustering' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stdout.strip().splitlines() == [
        "hidden_paths",
        "trim_model",
        "data_to_stslist",
        "False",
    ]


def test_mnhmm_api_is_available_from_top_level_package():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    code = (
        "import sys\n"
        "from sequenzo import MNHMM, build_mnhmm, estimate_mnhmm, simulate_mnhmm\n"
        "from sequenzo import compute_n_parameters, compute_n_observations\n"
        "print(MNHMM.__name__)\n"
        "print(build_mnhmm.__name__)\n"
        "print(estimate_mnhmm.__name__)\n"
        "print(simulate_mnhmm.__name__)\n"
        "print(compute_n_parameters.__name__)\n"
        "print(compute_n_observations.__name__)\n"
        "print('sequenzo.clustering' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stdout.strip().splitlines() == [
        "MNHMM",
        "build_mnhmm",
        "estimate_mnhmm",
        "simulate_mnhmm",
        "compute_n_parameters",
        "compute_n_observations",
        "False",
    ]
