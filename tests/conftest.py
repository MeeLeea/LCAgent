import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--provider",
        action="store",
        default=None,
        help="只检测指定供应商；未指定时检测全部供应商",
    )


@pytest.fixture(autouse=True)
def _reset_safety_cache():
    """每个测试前重置 safety 配置缓存,保证用例间隔离"""
    import tools.safety as safety
    safety._config_cache = None
    yield
    safety._config_cache = None
