"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_basic.py
@Time    : 26/02/2025 13:13
@Desc    :
"""

import sequenzo

def test_version():
    assert sequenzo.__version__ is not None  # Ensure version is not empty
