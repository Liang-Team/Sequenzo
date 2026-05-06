from sequenzo.utils import computer_performance
from sequenzo.utils import get_computer_performance


def test_get_computer_performance_returns_friendly_summary(monkeypatch):
    monkeypatch.setattr(computer_performance.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        computer_performance,
        "_get_memory_bytes",
        lambda: (16 * computer_performance.BYTES_PER_GB, 6 * computer_performance.BYTES_PER_GB),
    )
    monkeypatch.setattr(computer_performance, "_get_os_name", lambda: "TestOS 1.0")
    monkeypatch.setattr(computer_performance, "_get_machine_name", lambda: "test64")

    info = get_computer_performance()

    assert info == {
        "cpu_cores": 8,
        "total_memory_gb": 16.0,
        "available_memory_gb": 6.0,
        "os": "TestOS 1.0",
        "machine": "test64",
        "tier": "strong",
        "recommended_for": "regular Sequenzo analyses and medium-size benchmarks",
        "recommended_hardware": {
            "minimum": "4 CPU cores and 8 GB RAM for small exploratory analyses",
            "comfortable": "8 CPU cores and 16 GB RAM for regular Sequenzo workflows",
            "large_scale": "12+ CPU cores and 32+ GB RAM for larger benchmarks or repeated runs",
        },
        "suggested_threads": 8,
        "distance_matrix_memory_gb": {
            "N=1000": 0.01,
            "N=10000": 0.75,
            "N=30000": 6.71,
        },
        "advice": [
            "Start OpenMP-enabled runs with about 8 threads, then compare 4 and 8 if timing matters.",
            "Memory looks comfortable for regular workflows; be cautious when N becomes large.",
            "As a rough reference, a full 10,000 x 10,000 distance matrix needs about 0.75 GB before table overhead.",
        ],
        "summary": (
            "This computer has 8 CPU cores, 16.0 GB total memory "
            "(6.0 GB currently available), and runs TestOS 1.0 on test64. "
            "For typical Sequenzo work, this looks like a strong machine, suitable for "
            "regular Sequenzo analyses and medium-size benchmarks. "
            "A practical starting point is 8 OpenMP threads."
        ),
    }


def test_get_computer_performance_handles_unknown_memory(monkeypatch):
    monkeypatch.setattr(computer_performance.os, "cpu_count", lambda: None)
    monkeypatch.setattr(computer_performance, "_get_memory_bytes", lambda: (None, None))
    monkeypatch.setattr(computer_performance, "_get_os_name", lambda: "MysteryOS")
    monkeypatch.setattr(computer_performance, "_get_machine_name", lambda: "unknown machine")

    info = get_computer_performance()

    assert info["cpu_cores"] == 1
    assert info["total_memory_gb"] is None
    assert info["available_memory_gb"] is None
    assert info["machine"] == "unknown machine"
    assert info["tier"] == "limited"
    assert info["suggested_threads"] == 1
    assert "Memory size could not be detected" in info["advice"][1]
    assert "unknown total memory" in info["summary"]


def test_parse_macos_vm_stat_extracts_page_counts():
    text = """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             200.
Pages inactive:                           300.
Pages speculative:                         40.
"""

    pages = computer_performance._parse_macos_vm_stat(text)

    assert pages["Pages free"] == 100
    assert pages["Pages inactive"] == 300
    assert pages["Pages speculative"] == 40


def test_get_computer_performance_can_print_summary(monkeypatch, capsys):
    monkeypatch.setattr(computer_performance.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        computer_performance,
        "_get_memory_bytes",
        lambda: (8 * computer_performance.BYTES_PER_GB, 2 * computer_performance.BYTES_PER_GB),
    )
    monkeypatch.setattr(computer_performance, "_get_os_name", lambda: "TestOS 2.0")
    monkeypatch.setattr(computer_performance, "_get_machine_name", lambda: "test64")

    info = get_computer_performance(print_summary=True)
    captured = capsys.readouterr()

    assert info["tier"] == "capable"
    assert "capable machine" in captured.out


def test_get_computer_performance_is_top_level_export():
    import sequenzo

    assert callable(sequenzo.get_computer_performance)


def test_get_machine_name_uses_friendly_macos_labels(monkeypatch):
    monkeypatch.setattr(computer_performance.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(computer_performance.platform, "machine", lambda: "arm64")

    assert computer_performance._get_machine_name() == "Apple Silicon (arm64)"


def test_get_os_name_uses_macos_when_available(monkeypatch):
    monkeypatch.setattr(computer_performance.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(computer_performance, "_run_text_command", lambda command: "15.5\n")

    assert computer_performance._get_os_name() == "macOS 15.5"
