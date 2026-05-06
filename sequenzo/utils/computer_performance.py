"""
@Author  : Sequenzo contributors
@File    : computer_performance.py
@Desc    :
User-friendly computer performance helper.

This module uses only the Python standard library so users can inspect their
hardware before choosing larger Sequenzo analyses without installing extra
system packages.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from typing import Optional, Tuple


BYTES_PER_GB = 1024 ** 3


def get_computer_performance(print_summary: bool = False) -> dict:
    """
    Return a friendly, approximate summary of the current computer.

    The result is intentionally rough. It helps normal users understand whether
    their computer is likely to feel limited, comfortable, or strong for larger
    sequence analysis workloads; it is not a benchmarking tool.

    Parameters
    ----------
    print_summary : bool, default False
        If True, print the friendly summary before returning the dictionary.

    Returns
    -------
    dict
    Dictionary with CPU count, memory, platform, coarse performance tier,
        recommended hardware guidance, and a human-readable summary.

    Examples
    --------
    >>> info = get_computer_performance()
    >>> info["cpu_cores"] >= 1
    True
    >>> "summary" in info
    True
    """
    cpu_cores = os.cpu_count() or 1
    total_memory, available_memory = _get_memory_bytes()
    os_name = _get_os_name()
    machine = platform.machine() or "unknown machine"
    tier = _estimate_tier(cpu_cores, _bytes_to_gb(total_memory))

    total_memory_gb = _bytes_to_gb(total_memory)
    available_memory_gb = _bytes_to_gb(available_memory)
    recommended_for = _recommended_for(tier)
    recommended_hardware = _recommended_hardware()
    suggested_threads = _suggested_threads(cpu_cores)
    advice = _build_advice(
        cpu_cores=cpu_cores,
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        suggested_threads=suggested_threads,
    )
    summary = _build_summary(
        cpu_cores=cpu_cores,
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        os_name=os_name,
        machine=machine,
        tier=tier,
        recommended_for=recommended_for,
        suggested_threads=suggested_threads,
    )

    result = {
        "cpu_cores": cpu_cores,
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": available_memory_gb,
        "os": os_name,
        "machine": machine,
        "tier": tier,
        "recommended_for": recommended_for,
        "recommended_hardware": recommended_hardware,
        "suggested_threads": suggested_threads,
        "advice": advice,
        "summary": summary,
    }
    if print_summary:
        print(summary)
    return result


def _get_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    system = platform.system().lower()
    if system == "darwin":
        return _get_macos_memory_bytes()
    if system == "linux":
        return _get_linux_memory_bytes()
    if system == "windows":
        return _get_windows_memory_bytes()
    return _get_posix_total_memory_bytes(), None


def _get_macos_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    total = _run_int_command(["sysctl", "-n", "hw.memsize"])
    available = None

    page_size = _run_int_command(["sysctl", "-n", "hw.pagesize"])
    vm_stat = _run_text_command(["vm_stat"])
    if page_size is not None and vm_stat:
        pages = _parse_macos_vm_stat(vm_stat)
        free_pages = (
            pages.get("Pages free", 0)
            + pages.get("Pages inactive", 0)
            + pages.get("Pages speculative", 0)
        )
        if free_pages:
            available = free_pages * page_size

    return total, available


def _get_linux_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * 1024
    except OSError:
        return _get_posix_total_memory_bytes(), None

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if available is None:
        available = values.get("MemFree")
    return total, available


def _get_windows_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except AttributeError:
        return None, None
    if not ok:
        return None, None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _get_posix_total_memory_bytes() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * physical_pages)


def _parse_macos_vm_stat(text: str) -> dict:
    pages = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        digits = "".join(char for char in raw_value if char.isdigit())
        if digits:
            pages[key.strip()] = int(digits)
    return pages


def _run_int_command(command: list[str]) -> Optional[int]:
    text = _run_text_command(command)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _run_text_command(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _get_os_name() -> str:
    system = platform.system() or "Unknown OS"
    release = platform.release()
    return f"{system} {release}".strip()


def _bytes_to_gb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / BYTES_PER_GB, 1)


def _estimate_tier(cpu_cores: int, total_memory_gb: Optional[float]) -> str:
    memory = total_memory_gb or 0.0
    if cpu_cores >= 12 and memory >= 32:
        return "workstation"
    if cpu_cores >= 8 and memory >= 16:
        return "strong"
    if cpu_cores >= 4 and memory >= 8:
        return "capable"
    if cpu_cores >= 2 and memory >= 4:
        return "basic"
    return "limited"


def _recommended_for(tier: str) -> str:
    recommendations = {
        "workstation": "large distance matrices, repeated benchmarks, and heavier OpenMP runs",
        "strong": "regular Sequenzo analyses and medium-size benchmarks",
        "capable": "small to medium Sequenzo analyses and exploratory work",
        "basic": "small examples, teaching, and checking code before larger runs",
        "limited": "very small examples; use caution with distance matrices",
    }
    return recommendations.get(tier, recommendations["limited"])


def _recommended_hardware() -> dict:
    return {
        "minimum": "4 CPU cores and 8 GB RAM for small exploratory analyses",
        "comfortable": "8 CPU cores and 16 GB RAM for regular Sequenzo workflows",
        "large_scale": "12+ CPU cores and 32+ GB RAM for larger benchmarks or repeated runs",
    }


def _suggested_threads(cpu_cores: int) -> int:
    return max(1, min(cpu_cores, 8))


def _build_advice(
    cpu_cores: int,
    total_memory_gb: Optional[float],
    available_memory_gb: Optional[float],
    suggested_threads: int,
) -> list[str]:
    advice = []
    if cpu_cores >= 8:
        advice.append(
            f"Start OpenMP-enabled runs with about {suggested_threads} threads, then compare 4 and 8 if timing matters."
        )
    elif cpu_cores >= 4:
        advice.append(
            f"Use up to {suggested_threads} threads for normal runs; larger benchmarks may benefit from a stronger CPU."
        )
    else:
        advice.append("Keep thread counts low and prefer small pilot runs on this CPU.")

    if total_memory_gb is None:
        advice.append("Memory size could not be detected, so try a small run before building large distance matrices.")
    elif total_memory_gb >= 32:
        advice.append("Memory looks suitable for larger experiments, but distance matrices still grow roughly with N squared.")
    elif total_memory_gb >= 16:
        advice.append("Memory looks comfortable for regular workflows; be cautious when N becomes large.")
    elif total_memory_gb >= 8:
        advice.append("Memory is enough for smaller analyses; avoid very large full distance matrices.")
    else:
        advice.append("Memory is limited for sequence distance matrices; start with small samples.")

    if (
        available_memory_gb is not None
        and total_memory_gb is not None
        and available_memory_gb < max(2.0, total_memory_gb * 0.15)
    ):
        advice.append("Available memory is currently low; closing other apps may make Sequenzo runs more stable.")

    return advice


def _format_memory(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f} GB"


def _format_cpu_cores(cpu_cores: int) -> str:
    if cpu_cores == 1:
        return "1 CPU core"
    return f"{cpu_cores} CPU cores"


def _build_summary(
    cpu_cores: int,
    total_memory_gb: Optional[float],
    available_memory_gb: Optional[float],
    os_name: str,
    machine: str,
    tier: str,
    recommended_for: str,
    suggested_threads: int,
) -> str:
    cores = _format_cpu_cores(cpu_cores)
    total = _format_memory(total_memory_gb)
    available = _format_memory(available_memory_gb)
    return (
        f"This computer has {cores}, {total} total memory "
        f"({available} currently available), and runs {os_name} on {machine}. "
        f"For typical Sequenzo work, this looks like a {tier} machine, suitable for {recommended_for}. "
        f"A practical starting point is {suggested_threads} OpenMP thread(s)."
    )
