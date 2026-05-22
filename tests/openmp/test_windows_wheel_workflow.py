from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "python-app.yml"


def test_windows_cibuildwheel_smoke_test_uses_cmd_safe_python():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Add-Content test_sequenzo_ci.py @\"" not in workflow
    assert "from sequenzo import SequenceData, get_distance_matrix, Cluster" in workflow
    assert "import sequenzo.utils.core_distance_operations.core_distance_c_code" in workflow


def test_windows_wheel_repair_verifies_bundled_openmp_dlls():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    repair_script = (
        WORKFLOW.parents[2] / "maintenance_scripts" / "repair_windows_wheel.py"
    ).read_text(encoding="utf-8")

    assert "delvewheel show" in repair_script
    assert "LIBOMP_DLL_DIR" in workflow
    assert "sequenzo.libs" in repair_script
    assert "libomp140" in repair_script


def test_windows_conda_smoke_installs_repaired_wheel():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "conda-incubator/setup-miniconda@v3" in workflow
    assert "Windows Conda wheel smoke test" in workflow
    assert "CONDA_PREFIX=" in workflow
    assert "python -m pip install --force-reinstall \"$WHEEL\"" in workflow
    assert "[OK] Windows Conda wheel smoke test passed" in workflow
