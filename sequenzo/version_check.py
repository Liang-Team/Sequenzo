"""
@Author  : Yuqi Liang 梁彧祺
@File    : version_check.py
@Time    : 2025-11-20 07:27
@Desc    : Version check utility for Sequenzo

This module checks if the installed version of Sequenzo is up-to-date
by comparing it with the latest version available on PyPI.
Similar to pip's version notice functionality.
"""

import sys
import warnings
from typing import Optional, Tuple

# Try to import packaging for version comparison, fallback to simple comparison
try:
    from packaging import version as packaging_version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False


def get_installed_version() -> str:
    """
    Get the currently installed version of Sequenzo.
    
    Returns:
        str: The installed version string (e.g., "0.1.24")
    """
    try:
        # Try to get version from pyproject.toml via importlib.metadata
        if sys.version_info >= (3, 8):
            from importlib.metadata import version as get_package_version
            return get_package_version("sequenzo")
        else:
            # Fallback for Python < 3.8
            import pkg_resources
            return pkg_resources.get_distribution("sequenzo").version
    except Exception:
        # If we can't get the version, return None
        return None


def get_latest_version_from_pypi(package_name: str = "sequenzo", timeout: float = 1.0) -> Optional[str]:
    """
    Check PyPI for the latest version of the package.
    
    This function queries the PyPI JSON API to get the latest version.
    It uses a timeout to avoid blocking if the network is slow or unavailable.
    
    Args:
        package_name: Name of the package on PyPI (default: "sequenzo")
        timeout: Timeout in seconds for the HTTP request (default: 1.0)
        
    Returns:
        Optional[str]: The latest version string if available, None otherwise
    """
    try:
        import urllib.request
        import json
        
        # PyPI JSON API endpoint
        url = f"https://pypi.org/pypi/{package_name}/json"
        
        # Create request with timeout
        request = urllib.request.Request(url)
        request.add_header("User-Agent", f"sequenzo/{get_installed_version() or 'unknown'}")
        
        # Make request with timeout
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            return data.get("info", {}).get("version")
    except Exception:
        # Silently fail if we can't check (network issues, etc.)
        return None


def _simple_version_compare(installed: str, latest: str) -> bool:
    """
    Simple version comparison without external dependencies.
    
    This function compares version strings by splitting on '.' and comparing
    each component numerically. This works for most standard version formats
    like "0.1.24" but may not handle all edge cases.
    
    Args:
        installed: Currently installed version
        latest: Latest available version
        
    Returns:
        bool: True if installed >= latest, False otherwise
    """
    try:
        # Split versions into components
        installed_parts = [int(x) for x in installed.split('.')]
        latest_parts = [int(x) for x in latest.split('.')]
        
        # Pad shorter version with zeros
        max_len = max(len(installed_parts), len(latest_parts))
        installed_parts.extend([0] * (max_len - len(installed_parts)))
        latest_parts.extend([0] * (max_len - len(latest_parts)))
        
        # Compare component by component
        for i, l in zip(installed_parts, latest_parts):
            if i > l:
                return True  # installed is newer
            elif i < l:
                return False  # installed is older
        
        return True  # versions are equal
    except (ValueError, AttributeError):
        # If parsing fails, do string comparison as fallback
        return installed >= latest


def compare_versions(installed: str, latest: str) -> Tuple[bool, str]:
    """
    Compare two version strings to determine if an update is available.
    
    Args:
        installed: Currently installed version
        latest: Latest available version
        
    Returns:
        Tuple[bool, str]: (is_up_to_date, message)
            - is_up_to_date: True if installed version >= latest version
            - message: Human-readable comparison message
    """
    try:
        if HAS_PACKAGING:
            # Use packaging library for accurate version comparison
            installed_ver = packaging_version.parse(installed)
            latest_ver = packaging_version.parse(latest)
            
            if installed_ver >= latest_ver:
                return True, f"Installed version {installed} is up-to-date"
            else:
                return False, f"Installed version {installed} < latest version {latest}"
        else:
            # Fallback to simple comparison
            is_up_to_date = _simple_version_compare(installed, latest)
            if is_up_to_date:
                return True, f"Installed version {installed} is up-to-date"
            else:
                return False, f"Installed version {installed} < latest version {latest}"
    except Exception:
        # If version parsing fails, assume up-to-date to avoid false positives
        return True, "Could not compare versions"


def check_version_update(
    show_notice: bool = True,
    timeout: float = 1.0
) -> Optional[str]:
    """
    Check if a newer version of Sequenzo is available on PyPI.
    
    This function compares the installed version with the latest version
    on PyPI and optionally displays a notice if an update is available.
    
    Args:
        show_notice: If True, display a notice when update is available (default: True)
        timeout: Timeout in seconds for PyPI API request (default: 1.0)
        
    Returns:
        Optional[str]: The latest version string if available, None otherwise
        
    Examples:
        >>> from sequenzo.version_check import check_version_update
        >>> latest = check_version_update(show_notice=True)
        >>> if latest:
        ...     print(f"Latest version available: {latest}")
    """
    # Get installed version
    installed = get_installed_version()
    if not installed:
        # Can't determine installed version, skip check
        return None
    
    # Get latest version from PyPI
    latest = get_latest_version_from_pypi(timeout=timeout)
    if not latest:
        # Can't get latest version (network issue, etc.), skip
        return None
    
    # Compare versions
    is_up_to_date, message = compare_versions(installed, latest)
    
    # Show notice if update is available
    if not is_up_to_date and show_notice:
        print(
            f"[notice] A new release of sequenzo is available: {installed} -> {latest}",
            file=sys.stderr
        )
        print(
            f"[notice] To update, run: pip install --upgrade sequenzo=={latest}",
            file=sys.stderr
        )
    
    return latest


def check_version_update_async():
    """
    Asynchronously check for version updates without blocking.
    
    This function runs the version check in a background thread
    to avoid blocking the main import process. It's designed to be
    called during package import.
    """
    try:
        import threading
        
        def _check_in_background():
            """Run version check in background thread."""
            try:
                check_version_update(show_notice=True, timeout=1.0)
            except Exception:
                # Silently fail - we don't want version checks to break imports
                pass
        
        # Start background thread (daemon thread so it doesn't prevent exit)
        thread = threading.Thread(target=_check_in_background, daemon=True)
        thread.start()
    except Exception:
        # If threading fails, just skip the check
        pass


if __name__ == "__main__":
    # Allow manual version check
    print("Checking for Sequenzo updates...")
    latest = check_version_update(show_notice=True)
    if latest:
        installed = get_installed_version()
        print(f"\nInstalled version: {installed}")
        print(f"Latest version: {latest}")
    else:
        print("Could not check for updates (network issue or package not found)")
