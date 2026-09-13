"""HealthBandhu V: AI-assisted multiclass disease screening and clinical decision support.

The package is shared by the training notebook (notebooks/) and the Streamlit
web application (App.py), so both always use exactly the same emergency rules,
rule engine, ensemble maths, explanations and report layout.
"""

from . import config  # noqa: F401

__all__ = ["config"]
