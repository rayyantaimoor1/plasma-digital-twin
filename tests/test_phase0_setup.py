"""Phase 0 sanity check: confirms the environment and package layout are wired up
correctly before any real module logic exists. Delete once real per-module tests
(digital_twin, ai_module) make this redundant.
"""
import sys


def test_python_version_floor() -> None:
    # CLAUDE.md hard floor: XGBoost 3.x and SHAP 0.5x both require Python >= 3.12 to install.
    assert sys.version_info >= (3, 12)


def test_project_packages_importable() -> None:
    import ai_module  # noqa: F401
    import dashboard  # noqa: F401
    import digital_twin  # noqa: F401
