from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib


def test_setup_does_not_clone_unpinned_xsimd_from_network():
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    source = setup_py.read_text()

    assert "git\", \"clone" not in source
    assert "https://github.com/xtensor-stack/xsimd.git" not in source
    assert "include/xsimd/xsimd.hpp" in source


def test_macos_wheel_repair_fails_instead_of_copying_unrepaired_wheel():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-app.yml"
    source = workflow.read_text()
    repo_root = Path(__file__).resolve().parents[1]

    assert "copying wheel as-is" not in source
    assert "cp {wheel} {dest_dir}/" not in source
    assert 'find . -name "*.c" -delete' not in source
    assert 'find . -name "*.so" -delete' not in source
    assert "delocate-wheel --require-archs {delocate_archs}" in source
    assert 'CIBW_TARGET_OSX_x86_64: "10.15"' in source
    assert 'CIBW_TARGET_OSX_arm64: "11.0"' in source
    assert "build_macos_ci_libomp.sh" in source
    assert "repair_windows_wheel.py" in source
    repair_script = (repo_root / "maintenance_scripts" / "repair_windows_wheel.py").read_text()
    assert "delvewheel did not emit a wheel; copying built wheel as-is" in repair_script
    assert (repo_root / "maintenance_scripts" / "repair_windows_wheel.py").is_file()
    assert "exit 1" in source
    assert (repo_root / "maintenance_scripts" / "build_macos_ci_libomp.sh").is_file()


def test_setup_respects_macos_deployment_target_from_environment():
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    source = setup_py.read_text()

    assert "os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.15')" in source
    assert "os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'" not in source
    assert "os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '14.0')" not in source


def test_setup_builds_fanny_cpp_binding_without_fast_math():
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    source = setup_py.read_text()

    assert "'sequenzo/clustering/fuzzy_clustering/src/'" in source
    assert "'sequenzo/clustering/fuzzy_clustering/src/fanny.cpp'" in source
    assert "or 'fanny.cpp' in src" in source


def test_sdist_includes_fanny_cpp_sources_required_by_setup():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = (repo_root / "MANIFEST.in").read_text()
    config = tomllib.loads((repo_root / "pyproject.toml").read_text())

    assert "recursive-include sequenzo/clustering/fuzzy_clustering/src *.cpp" in manifest
    assert "recursive-include sequenzo/clustering/fuzzy_clustering/src *.h" in manifest

    package_data = config["tool"]["setuptools"]["package-data"]["sequenzo.clustering"]
    assert "fuzzy_clustering/src/**/*.cpp" in package_data
    assert "fuzzy_clustering/src/**/*.h" in package_data


def test_pytest_collects_only_project_tests_by_default():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text())

    pytest_config = config["tool"]["pytest"]["ini_options"]
    assert pytest_config["testpaths"] == ["tests"]
