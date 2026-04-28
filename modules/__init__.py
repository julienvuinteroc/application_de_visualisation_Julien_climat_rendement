"""
Modules de l'application
"""

# ===============================
# DATA
# ===============================
from .data_loader import load_data, apply_filters, load_geojson

# ===============================
# UTILS
# ===============================
from .utils import (
    zone_sort_key,
    zone_order_from_series,
    complete_years,
    bootstrap_ci,
    format_number,
)

# ===============================
# VISUALISATION (safe import)
# ===============================
try:
    from .visualization import (
        create_rendement_chart,
        create_volume_chart,
        create_comparison_bar_chart,
        create_pie_chart,
        create_scatter_with_trend,
        create_choropleth_map,
        create_anomaly_chart,
    )
except:
    pass


# ===============================
# EXPORT
# ===============================
__all__ = [
    "load_data",
    "apply_filters",
    "load_geojson",
    "zone_sort_key",
    "zone_order_from_series",
    "complete_years",
    "bootstrap_ci",
    "format_number",
]