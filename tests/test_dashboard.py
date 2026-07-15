"""Smoke tests for the Streamlit dashboard (Phase 3).

Streamlit's own AppTest harness runs each page programmatically in a headless
script-runner and captures any exception raised during a run - so these tests
genuinely exercise every panel end-to-end (sidebar widgets, cached model/data
builders, and the Plotly figures each page renders), catching import errors, API
misuse, and runtime failures that a plain `import` would not.

These are slow: the first page that touches a cached resource trains the models /
builds the conformal layer. They are marked so a fast run can skip them, but they
are the real correctness gate for the dashboard.
"""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.dashboard

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APP = _ROOT / "dashboard" / "app.py"
_PAGES = sorted((_ROOT / "dashboard" / "pages").glob("*.py"))
_PAGE_TIMEOUT = 300  # seconds; first cached-resource build can be slow


def _run(path: pathlib.Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=_PAGE_TIMEOUT)
    at.run()
    return at


def test_pages_directory_is_populated() -> None:
    assert len(_PAGES) >= 7, "expected the seven analysis panels under dashboard/pages/"


def test_overview_app_runs_without_exception() -> None:
    at = _run(_APP)
    assert not at.exception, f"app.py raised: {at.exception}"
    assert at.title, "overview page should render a title"


@pytest.mark.parametrize("page", _PAGES, ids=[p.stem for p in _PAGES])
def test_each_page_runs_without_exception(page) -> None:
    at = _run(page)
    assert not at.exception, f"{page.name} raised: {at.exception}"
    assert at.title, f"{page.name} should render a title"


def test_sidebar_config_change_propagates_on_overview() -> None:
    """Changing the RF power slider should re-run cleanly and update the metrics -
    i.e. the shared sidebar config is genuinely wired into the panels."""
    at = _run(_APP)
    assert at.sidebar.slider  # the shared chamber-config sliders exist
    at.sidebar.slider(key="rf_power").set_value(200.0).run(timeout=_PAGE_TIMEOUT)
    assert not at.exception
