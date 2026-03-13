"""Primary visual plot builder application with rich interactive controls."""

import ast
import inspect
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import pygame
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except Exception:  # pragma: no cover - optional dependency in some environments
    dendrogram = None
    linkage = None

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None

try:
    from visual_plot_builder_common import COLUMN_TYPE_COLORS as SHARED_COLUMN_TYPE_COLORS
    from visual_plot_builder_common import COLUMN_TYPE_PREFIX as SHARED_COLUMN_TYPE_PREFIX
except Exception:  # pragma: no cover
    SHARED_COLUMN_TYPE_COLORS = None
    SHARED_COLUMN_TYPE_PREFIX = None


# ------------------------------
# Window and color configuration
# ------------------------------
DEFAULT_WINDOW_W = 1600
DEFAULT_WINDOW_H = 920
FPS = 60

BG = (24, 26, 32)
PANEL = (34, 37, 45)
PANEL_ALT = (42, 46, 56)
TEXT = (230, 232, 236)
MUTED = (170, 175, 185)
ACCENT = (75, 170, 255)
GOOD = (74, 186, 108)
WARN = (225, 153, 88)
BAD = (220, 84, 84)
TOOLTIP_BG = (16, 19, 24)

MENU_BG = (30, 34, 42)
REL_MODAL_BG = (18, 21, 28)

REL_JOIN_ORDER = ["inner", "left", "right", "outer"]
REL_JOIN_COLORS = {
    "inner": (74, 186, 108),
    "left": (75, 170, 255),
    "right": (225, 153, 88),
    "outer": (186, 123, 255),
}

COLUMN_TYPE_COLORS = {
    "nominal": (241, 106, 95),
    "ordinal": (244, 179, 80),
    "interval": (109, 170, 255),
    "ratio": (98, 210, 130),
}

COLUMN_TYPE_PREFIX = {
    "nominal": "N: ",
    "ordinal": "O: ",
    "interval": "I: ",
    "ratio": "R: ",
}

if SHARED_COLUMN_TYPE_COLORS:
    COLUMN_TYPE_COLORS = SHARED_COLUMN_TYPE_COLORS
if SHARED_COLUMN_TYPE_PREFIX:
    COLUMN_TYPE_PREFIX = SHARED_COLUMN_TYPE_PREFIX


# ------------------------------
# Plot and parameter registries
# ------------------------------
SEABORN_PLOT_FUNCS = [
    "scatterplot",
    "lineplot",
    "histplot",
    "kdeplot",
    "ecdfplot",
    "rugplot",
    "stripplot",
    "swarmplot",
    "boxplot",
    "violinplot",
    "boxenplot",
    "pointplot",
    "barplot",
    "countplot",
    "regplot",
    "residplot",
    "heatmap",
    "clustermap",
    "jointplot",
    "pairplot",
    "relplot",
    "displot",
    "catplot",
    "lmplot",
]

COLUMN_PARAM_CANDIDATES = {
    "x",
    "y",
    "hue",
    "size",
    "style",
    "units",
    "weights",
    "row",
    "col",
    "x_vars",
    "y_vars",
    "vars",
    "feature_column",
    "target_column",
}

GLOBAL_FIXED_OPTIONS: Dict[str, List[Any]] = {
    "legend": ["auto", "brief", "full", True, False],
    "multiple": ["layer", "dodge", "stack", "fill"],
    "element": ["bars", "step", "poly"],
    "fill": [True, False],
    "common_norm": [True, False],
    "common_bins": [True, False],
    "cbar": [True, False],
    "dodge": ["auto", True, False],
    "native_scale": [True, False],
    "orient": ["x", "y", "v", "h"],
    "corner": [True, False],
    "dropna": [True, False],
    "markers": [True, False],
    "diag_kind": ["auto", "hist", "kde", None],
    "palette": [
        "deep",
        "muted",
        "bright",
        "pastel",
        "dark",
        "colorblind",
        "viridis",
        "magma",
        "rocket",
        "mako",
        "crest",
        "flare",
    ],
    "estimator": ["mean", "median", "sum", "min", "max"],
    "errorbar": ["ci", "pi", "se", "sd", None],
    "log_scale": [True, False],
}

PLOT_PARAM_OVERRIDES: Dict[str, Dict[str, List[Any]]] = {
    "jointplot": {
        "kind": ["scatter", "kde", "hist", "hex", "reg", "resid"],
    },
    "catplot": {
        "kind": ["strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"],
    },
    "relplot": {
        "kind": ["scatter", "line"],
    },
    "displot": {
        "kind": ["hist", "kde", "ecdf"],
    },
    "rf_regression": {
        "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "oob_score": [True, False],
        "warm_start": [True, False],
    },
    "dendrogram": {
        "method": ["single", "complete", "average", "weighted", "centroid", "median", "ward"],
        "metric": ["euclidean", "cityblock", "cosine", "chebyshev", "correlation"],
        "orientation": ["top", "bottom", "left", "right"],
        "truncate_mode": [None, "lastp", "level"],
    },
    "kmeans_cluster": {
        "algorithm": ["lloyd", "elkan"],
        "init": ["k-means++", "random"],
    },
    "gaussian_mixture": {
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "init_params": ["kmeans", "k-means++", "random", "random_from_data"],
    },
    "pca_plot": {
        "svd_solver": ["auto", "full", "covariance_eigh", "arpack", "randomized"],
        "whiten": [True, False],
    },
    "tsne_plot": {
        "method": ["barnes_hut", "exact"],
        "init": ["pca", "random"],
        "metric": ["euclidean", "cosine", "manhattan"],
    },
}

PLOT_DESCRIPTIONS = {
    "scatterplot": "Scatter plot for two variables, with semantic mappings.",
    "lineplot": "Line plot for trends across one axis, with grouping semantics.",
    "histplot": "Histogram for one variable with optional semantic grouping.",
    "kdeplot": "Kernel density estimate visualization for one or two variables.",
    "ecdfplot": "Empirical cumulative distribution plot.",
    "rugplot": "Rug marks showing individual observations along an axis.",
    "stripplot": "Categorical scatter plot with jittered points.",
    "swarmplot": "Categorical scatter plot with non-overlapping points.",
    "boxplot": "Categorical box-and-whisker summary plot.",
    "violinplot": "Categorical density + summary visualization.",
    "boxenplot": "Enhanced box plot focused on tail distribution.",
    "pointplot": "Categorical estimate plot with confidence intervals.",
    "barplot": "Bar chart showing aggregated values by category.",
    "countplot": "Bar chart of count by category.",
    "regplot": "Scatter plot with fitted regression model.",
    "residplot": "Residual plot for regression diagnostics.",
    "heatmap": "Color-encoded matrix visualization.",
    "clustermap": "Heatmap with hierarchical clustering.",
    "jointplot": "Bivariate plot with marginal distributions.",
    "pairplot": "Grid of pairwise relationships across columns.",
    "relplot": "Figure-level relational plot with faceting.",
    "displot": "Figure-level distribution plot with faceting.",
    "catplot": "Figure-level categorical plot with faceting.",
    "lmplot": "Figure-level regression plot with faceting.",
    "rf_regression": "Random Forest regression with stratified 5-fold CV and feature importances.",
    "dendrogram": "Hierarchical clustering dendrogram from selected numeric columns.",
    "kmeans_cluster": "KMeans clustering on one or two columns with grouped distribution/relationship plot.",
    "gaussian_mixture": "Gaussian Mixture clustering on one or two columns with optional 2D 1-sigma contours.",
    "quiver_plot": "Vector field gradient plot built from x/y grid positions and z values.",
    "parallel_lines": "Parallel coordinates plot for multiple features with optional class grouping.",
    "pca_plot": "PCA projection to two dimensions from selected feature columns.",
    "tsne_plot": "t-SNE projection to two dimensions from selected feature columns.",
    "markov_chain": "Transition graph from feature value to target value with probability labels.",
    "sankey_plot": "Flow graph from feature value to target value with widths proportional to counts.",
}

PARAM_DESCRIPTIONS = {
    "x": "Column mapped to the x axis.",
    "y": "Column mapped to the y axis.",
    "hue": "Column mapped to color grouping.",
    "size": "Column mapped to marker/element size grouping.",
    "style": "Column mapped to marker or line style grouping.",
    "weights": "Column used as weights during estimation.",
    "row": "Facet row grouping column.",
    "col": "Facet column grouping column.",
    "vars": "Multiple columns used together for matrix-style plots.",
    "x_vars": "Columns for x variables in pair grids.",
    "y_vars": "Columns for y variables in pair grids.",
    "kind": "Subtype selector for figure-level wrappers.",
    "palette": "Color palette name or mapping.",
    "feature_columns": "Input feature columns for random forest regression.",
    "target_column": "Single output/target column for random forest regression.",
    "data_columns": "One or two numeric input columns used for clustering plots.",
    "class_column": "Optional class/group column used for coloring/grouping.",
    "feature_columns": "Input feature columns.",
    "z": "Value column used for grid cell intensity.",
    "feature_column": "Single feature column.",
    "target_column": "Single target/output column.",
}


# ------------------------------
# Data model classes
# ------------------------------
@dataclass
class PlotSpec:
    """Description of a plot type including required/optional input configuration."""
    name: str
    description: str
    required_slots: List[str]
    column_slots: List[str]
    multi_slots: Set[str]
    option_params: List[str]
    function_name: Optional[str] = None
    custom: bool = False
    group: str = "Seaborn Plots"


@dataclass
class DragItem:
    """Represents a currently dragged item in the UI."""
    kind: str  # "column"
    value: str
    table: Optional[str] = None
    values: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """Relationship (join edge) between two table columns."""
    left_table: str
    left_col: str
    right_table: str
    right_col: str
    join_mode: str = "inner"
    click_count: int = 0


@dataclass
class TableSource:
    """Metadata describing where table data is sourced from."""
    kind: str  # "memory" | "sqlite"
    name: str
    columns: List[str]
    file_path: Optional[str] = None
    sqlite_table: Optional[str] = None


class ScrollPanel:
    """Simple vertical scrolling state holder for a panel viewport."""
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.scroll_y = 0
        self.content_h = rect.h

    def clamp_scroll(self):
        max_scroll = max(0, self.content_h - self.rect.h)
        self.scroll_y = max(0, min(self.scroll_y, max_scroll))

    def wheel(self, dy: int):
        self.scroll_y -= dy * 28
        self.clamp_scroll()


# ------------------------------
# Plot specification construction
# ------------------------------
def _build_seaborn_spec(plot_name: str) -> PlotSpec:
    """Build a PlotSpec from a seaborn function signature."""
    fn = getattr(sns, plot_name)
    sig = inspect.signature(fn)

    required_slots: List[str] = []
    column_slots: List[str] = []
    option_params: List[str] = []

    for pname, param in sig.parameters.items():
        if pname in {"self", "data", "ax", "kwargs"}:
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue

        if pname in COLUMN_PARAM_CANDIDATES:
            column_slots.append(pname)
            if param.default is inspect._empty:
                required_slots.append(pname)
            continue

        option_params.append(pname)

    if plot_name == "countplot" and "x" not in required_slots and "y" not in required_slots:
        required_slots.append("x")
    if plot_name == "pairplot" and "vars" in column_slots and "vars" not in required_slots:
        required_slots.append("vars")

    multi_slots = {s for s in column_slots if s in {"vars", "x_vars", "y_vars"}}

    return PlotSpec(
        name=plot_name,
        description=PLOT_DESCRIPTIONS.get(plot_name, f"Seaborn {plot_name} plot."),
        required_slots=required_slots,
        column_slots=column_slots,
        multi_slots=multi_slots,
        option_params=option_params,
        function_name=plot_name,
        custom=False,
        group="Seaborn Plots",
    )


def build_plot_specs() -> Dict[str, PlotSpec]:
    """Construct all supported plot specifications for the application."""
    specs: Dict[str, PlotSpec] = {}

    for plot_name in SEABORN_PLOT_FUNCS:
        if hasattr(sns, plot_name):
            specs[plot_name] = _build_seaborn_spec(plot_name)

    for grid_plot in ("heatmap", "clustermap"):
        if grid_plot in specs:
            specs[grid_plot].required_slots = ["x", "y", "z"]
            specs[grid_plot].column_slots = ["x", "y", "z"]
            specs[grid_plot].multi_slots = set()
            specs[grid_plot].custom = True
            specs[grid_plot].function_name = None

    rf_sig = inspect.signature(RandomForestRegressor)
    rf_options: List[str] = []
    for pname, param in rf_sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        rf_options.append(pname)

    specs["rf_regression"] = PlotSpec(
        name="rf_regression",
        description=PLOT_DESCRIPTIONS["rf_regression"],
        required_slots=["feature_columns", "target_column"],
        column_slots=["feature_columns", "target_column"],
        multi_slots={"feature_columns"},
        option_params=rf_options,
        function_name=None,
        custom=True,
        group="AI Analysis",
    )

    specs["dendrogram"] = PlotSpec(
        name="dendrogram",
        description=PLOT_DESCRIPTIONS["dendrogram"],
        required_slots=["feature_columns"],
        column_slots=["feature_columns"],
        multi_slots={"feature_columns"},
        option_params=["method", "metric", "orientation", "truncate_mode", "p", "leaf_rotation", "leaf_font_size"],
        function_name=None,
        custom=True,
        group="AI Analysis",
    )

    km_sig = inspect.signature(KMeans)
    km_options: List[str] = []
    for pname, param in km_sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        km_options.append(pname)

    specs["kmeans_cluster"] = PlotSpec(
        name="kmeans_cluster",
        description=PLOT_DESCRIPTIONS["kmeans_cluster"],
        required_slots=["data_columns"],
        column_slots=["data_columns"],
        multi_slots={"data_columns"},
        option_params=km_options,
        function_name=None,
        custom=True,
        group="AI Analysis",
    )

    gm_sig = inspect.signature(GaussianMixture)
    gm_options: List[str] = []
    for pname, param in gm_sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        gm_options.append(pname)

    specs["gaussian_mixture"] = PlotSpec(
        name="gaussian_mixture",
        description=PLOT_DESCRIPTIONS["gaussian_mixture"],
        required_slots=["data_columns"],
        column_slots=["data_columns"],
        multi_slots={"data_columns"},
        option_params=gm_options,
        function_name=None,
        custom=True,
        group="AI Analysis",
    )

    pca_sig = inspect.signature(PCA)
    pca_options: List[str] = []
    for pname, param in pca_sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        pca_options.append(pname)
    specs["pca_plot"] = PlotSpec(
        name="pca_plot",
        description=PLOT_DESCRIPTIONS["pca_plot"],
        required_slots=["feature_columns"],
        column_slots=["feature_columns", "class_column"],
        multi_slots={"feature_columns"},
        option_params=pca_options,
        function_name=None,
        custom=True,
        group="AI Analysis",
    )

    tsne_sig = inspect.signature(TSNE)
    tsne_options: List[str] = []
    for pname, param in tsne_sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        tsne_options.append(pname)
    specs["tsne_plot"] = PlotSpec(
        name="tsne_plot",
        description=PLOT_DESCRIPTIONS["tsne_plot"],
        required_slots=["feature_columns"],
        column_slots=["feature_columns", "class_column"],
        multi_slots={"feature_columns"},
        option_params=tsne_options,
        function_name=None,
        custom=True,
        group="AI Analysis",
    )

    specs["quiver_plot"] = PlotSpec(
        name="quiver_plot",
        description=PLOT_DESCRIPTIONS["quiver_plot"],
        required_slots=["x", "y", "z"],
        column_slots=["x", "y", "z"],
        multi_slots=set(),
        option_params=[],
        function_name=None,
        custom=True,
        group="Seaborn Plots",
    )

    specs["parallel_lines"] = PlotSpec(
        name="parallel_lines",
        description=PLOT_DESCRIPTIONS["parallel_lines"],
        required_slots=["feature_columns"],
        column_slots=["feature_columns", "class_column"],
        multi_slots={"feature_columns"},
        option_params=[],
        function_name=None,
        custom=True,
        group="Seaborn Plots",
    )

    specs["markov_chain"] = PlotSpec(
        name="markov_chain",
        description=PLOT_DESCRIPTIONS["markov_chain"],
        required_slots=["feature_column", "target_column"],
        column_slots=["feature_column", "target_column"],
        multi_slots=set(),
        option_params=[],
        function_name=None,
        custom=True,
        group="Seaborn Plots",
    )

    specs["sankey_plot"] = PlotSpec(
        name="sankey_plot",
        description=PLOT_DESCRIPTIONS["sankey_plot"],
        required_slots=["feature_column", "target_column"],
        column_slots=["feature_column", "target_column"],
        multi_slots=set(),
        option_params=[],
        function_name=None,
        custom=True,
        group="Seaborn Plots",
    )

    return specs


# ------------------------------
# App implementation
# ------------------------------
class App:
    """Main Pygame application for interactive plot building and analysis."""
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Seaborn Visual Builder")
        self.fullscreen = False
        self.screen = pygame.display.set_mode((DEFAULT_WINDOW_W, DEFAULT_WINDOW_H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 18)
        self.font_sm = pygame.font.SysFont("arial", 15)
        self.font_lg = pygame.font.SysFont("arial", 22, bold=True)

        self.plot_specs = build_plot_specs()
        self.plot_groups: Dict[str, Dict[str, Any]] = {
            "Seaborn Plots": {"collapsed": False},
            "AI Analysis": {"collapsed": False},
        }

        self._current_menu_values: List[Any] = []

        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.table_sources: Dict[str, TableSource] = {}
        self.table_collapsed: Dict[str, bool] = {}
        self.data_layout: List[Tuple[pygame.Rect, str, str, str]] = []
        self.data_col_ui: Dict[Tuple[str, str], Dict[str, pygame.Rect]] = {}
        self.plot_layout: List[Tuple[pygame.Rect, str, str]] = []  # (rect, kind, value)
        self.slot_layout: Dict[str, pygame.Rect] = {}
        self.option_layout: Dict[str, pygame.Rect] = {}

        self.selected_plot: Optional[str] = None
        self.slot_values: Dict[str, Union[None, str, List[str]]] = {}
        self.option_values: Dict[str, Any] = {}
        self.active_slot: Optional[str] = None
        self.last_multi_anchor: Optional[Tuple[str, str]] = None
        self.input_header_layout: Dict[str, pygame.Rect] = {}
        self.input_sections = {"required": False, "optional": False}

        self.drag_item: Optional[DragItem] = None
        self.drag_pos = (0, 0)

        self.menu_target: Optional[Tuple[str, bool]] = None
        self.menu_rect: Optional[pygame.Rect] = None
        self.menu_items: List[Tuple[pygame.Rect, Any]] = []

        self.input_target: Optional[Tuple[str, bool]] = None
        self.input_rect: Optional[pygame.Rect] = None
        self.input_value: str = ""

        self.tooltip_text = ""
        self.chart_surface: Optional[pygame.Surface] = None
        self.status = "Load one or more CSV/Excel/SQLite files to begin."

        self.source_files: List[str] = []
        self.column_types: Dict[str, str] = {}  # "table::col" -> nominal|ordinal|interval|ratio
        self.ordinal_orders: Dict[str, List[str]] = {}  # "table::col" -> ordered category values

        self.col_type_menu_target: Optional[Tuple[str, str]] = None
        self.col_type_menu_rect: Optional[pygame.Rect] = None
        self.col_type_menu_items: List[Tuple[pygame.Rect, str]] = []

        self.last_cursor_key: Optional[str] = None

        # Relationship editor state.
        self.relationships: List[Relation] = []
        self.rel_editor_open = False
        self.rel_modal_rect = pygame.Rect(0, 0, 0, 0)
        self.rel_table_boxes: Dict[str, pygame.Rect] = {}
        self.rel_column_boxes: List[Tuple[pygame.Rect, str, str]] = []
        self.rel_line_layout: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
        self.rel_drag_start: Optional[Tuple[str, str]] = None
        self.rel_scroll_y = 0
        self.rel_scroll_max = 0
        self.rel_canvas_rect = pygame.Rect(0, 0, 0, 0)
        self.rel_clear_btn = pygame.Rect(0, 0, 0, 0)

        # Splitter-based resizing state.
        self.left_panel_w = 420
        self.right_panel_w = 320
        self.builder_h = 430
        self.resizing_panel: Optional[str] = None

        self.update_layout()

    # ------------------------------
    # Layout and panel setup
    # ------------------------------
    def update_layout(self):
        self.window_w, self.window_h = self.screen.get_size()

        # Top menu bar occupies fixed height and hosts all control actions.
        self.menu_bar_rect = pygame.Rect(0, 0, self.window_w, 50)

        self.left_panel_w = int(max(260, min(self.left_panel_w, self.window_w - 760)))
        self.right_panel_w = int(max(240, min(self.right_panel_w, self.window_w - self.left_panel_w - 360)))

        self.data_rect = pygame.Rect(12, 62, self.left_panel_w, self.window_h - 74)
        self.plot_type_rect = pygame.Rect(self.window_w - self.right_panel_w - 12, 62, self.right_panel_w, self.window_h - 74)

        center_x = self.data_rect.right + 12
        center_w = self.plot_type_rect.left - center_x - 12
        center_w = max(360, center_w)

        max_builder_h = self.window_h - 230
        self.builder_h = int(max(220, min(self.builder_h, max_builder_h)))
        self.builder_rect = pygame.Rect(center_x, 62, center_w, self.builder_h)
        self.chart_rect = pygame.Rect(center_x, self.builder_rect.bottom + 10, center_w, self.window_h - self.builder_rect.bottom - 22)
        self.options_rect = pygame.Rect(0, 0, 0, 0)

        self.left_splitter = pygame.Rect(self.data_rect.right + 4, self.data_rect.y, 8, self.data_rect.h)
        self.right_splitter = pygame.Rect(self.plot_type_rect.x - 8, self.plot_type_rect.y, 8, self.plot_type_rect.h)
        self.middle_splitter = pygame.Rect(self.builder_rect.x, self.builder_rect.bottom + 3, self.builder_rect.w, 8)

        self.data_panel = ScrollPanel(self.data_rect.inflate(-10, -10))
        self.plot_panel = ScrollPanel(self.plot_type_rect.inflate(-10, -10))
        self.builder_panel = ScrollPanel(pygame.Rect(self.builder_rect.x + 10, self.builder_rect.y + 92, self.builder_rect.w - 20, self.builder_rect.h - 102))
        self.options_panel = ScrollPanel(pygame.Rect(0, 0, 0, 0))

        self._build_menu_buttons()

    def _build_menu_buttons(self):
        # Menu buttons are arranged left-to-right and keep a consistent height.
        labels = [
            ("Load Files", ACCENT, "load_files"),
            ("Save Plot", GOOD, "save_plot"),
            ("Relationships", WARN, "relationships"),
            ("Save State", (92, 146, 206), "save_state"),
            ("Load State", (114, 174, 132), "load_state"),
            ("Fullscreen", (108, 112, 138), "fullscreen"),
            ("Calculate", GOOD, "calculate"),
            ("Clear", WARN, "clear"),
        ]
        x = 10
        y = 8
        h = 34
        self.menu_buttons: List[Tuple[pygame.Rect, str, Tuple[int, int, int], str]] = []
        for text, color, action in labels:
            w = max(96, self.font_sm.size(text)[0] + 20)
            r = pygame.Rect(x, y, w, h)
            self.menu_buttons.append((r, text, color, action))
            x += w + 8

    # ------------------------------
    # File dialogs and persistence
    # ------------------------------
    def _make_tk_root(self):
        if not tk:
            return None
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        root.lift()
        root.focus_force()
        return root

    def open_files_picker(self) -> List[str]:
        if not filedialog or not tk:
            self.status = "tkinter is unavailable; cannot open file dialog."
            return []
        pygame.event.pump()
        pygame.event.set_grab(False)
        root = self._make_tk_root()
        paths = filedialog.askopenfilenames(
            parent=root,
            title="Select data file(s)",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.db *.sqlite *.sqlite3"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("SQLite", "*.db *.sqlite *.sqlite3"),
                ("All files", "*.*"),
            ],
        )
        if root:
            root.attributes("-topmost", False)
            root.destroy()
        return list(paths)

    def pick_save_path(self, title: str, ext: str, filetypes: List[Tuple[str, str]]) -> Optional[str]:
        if not filedialog or not tk:
            self.status = "tkinter is unavailable; cannot open save dialog."
            return None
        pygame.event.pump()
        pygame.event.set_grab(False)
        root = self._make_tk_root()
        path = filedialog.asksaveasfilename(parent=root, title=title, defaultextension=ext, filetypes=filetypes)
        if root:
            root.attributes("-topmost", False)
            root.destroy()
        return path or None

    def pick_open_path(self, title: str, filetypes: List[Tuple[str, str]]) -> Optional[str]:
        if not filedialog or not tk:
            self.status = "tkinter is unavailable; cannot open file dialog."
            return None
        pygame.event.pump()
        pygame.event.set_grab(False)
        root = self._make_tk_root()
        path = filedialog.askopenfilename(parent=root, title=title, filetypes=filetypes)
        if root:
            root.attributes("-topmost", False)
            root.destroy()
        return path or None

    def _unique_table_name(self, base_name: str) -> str:
        if base_name not in self.dataframes:
            return base_name
        n = 2
        while f"{base_name}_{n}" in self.dataframes:
            n += 1
        return f"{base_name}_{n}"

    def fetch_table_dataframe(self, table: str, needed_cols: Optional[List[str]] = None) -> pd.DataFrame:
        src = self.table_sources.get(table)
        if src is None:
            return self.dataframes.get(table, pd.DataFrame()).copy()
        if src.kind == "memory":
            df = self.dataframes.get(table, pd.DataFrame()).copy()
            if needed_cols:
                keep = [c for c in needed_cols if c in df.columns]
                return df[keep].copy()
            return df
        if src.kind == "sqlite" and src.file_path and src.sqlite_table:
            cols = needed_cols if needed_cols else src.columns
            quoted = ", ".join([f'"{c}"' for c in cols])
            query = f'SELECT {quoted} FROM "{src.sqlite_table}"'
            conn = sqlite3.connect(src.file_path)
            try:
                return pd.read_sql_query(query, conn)
            finally:
                conn.close()
        return pd.DataFrame(columns=src.columns)

    def load_file(self, path: str, append: bool = True) -> bool:
        ext = os.path.splitext(path)[1].lower()
        new_data: Dict[str, pd.DataFrame] = {}

        try:
            if ext == ".csv":
                name = os.path.splitext(os.path.basename(path))[0]
                new_data[name] = pd.read_csv(path)
            elif ext in {".xlsx", ".xls"}:
                sheets = pd.read_excel(path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    new_data[str(sheet_name)] = df
            elif ext in {".db", ".sqlite", ".sqlite3"}:
                conn = sqlite3.connect(path)
                try:
                    table_names = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                        conn,
                    )["name"].tolist()
                    for table in table_names:
                        col_rows = pd.read_sql_query(f'PRAGMA table_info("{table}")', conn)
                        cols = [str(c) for c in col_rows["name"].tolist()]
                        new_data[str(table)] = pd.DataFrame(columns=cols)
                finally:
                    conn.close()
            else:
                self.status = f"Unsupported file extension: {ext}"
                return False
        except Exception as e:
            self.status = f"Failed to load file: {e}"
            return False

        if not new_data:
            self.status = "No tables/sheets found in selected file."
            return False

        if not append:
            self.dataframes = {}
            self.table_sources = {}
            self.table_collapsed = {}
            self.relationships = []

        count = 0
        for name, df in new_data.items():
            unique = self._unique_table_name(name)
            self.dataframes[unique] = df
            self.table_collapsed[unique] = False
            if ext in {".db", ".sqlite", ".sqlite3"}:
                self.table_sources[unique] = TableSource(
                    kind="sqlite",
                    name=unique,
                    columns=[str(c) for c in df.columns.tolist()],
                    file_path=path,
                    sqlite_table=name,
                )
            else:
                self.table_sources[unique] = TableSource(
                    kind="memory",
                    name=unique,
                    columns=[str(c) for c in df.columns.tolist()],
                )
            count += 1

        if path not in self.source_files:
            self.source_files.append(path)

        self.status = f"Loaded {count} table(s) from {os.path.basename(path)}"
        return True

    def load_files(self, paths: List[str], append: bool = True):
        if not paths:
            return
        loaded = 0
        for i, path in enumerate(paths):
            ok = self.load_file(path, append=(append if i == 0 else True))
            if ok:
                loaded += 1
        if loaded:
            self.status = f"Loaded {loaded} file(s). Total tables: {len(self.dataframes)}"

    def save_plot(self):
        if self.chart_surface is None:
            self.status = "No plot to save."
            return
        path = self.pick_save_path("Save plot", ".png", [("PNG", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            pygame.image.save(self.chart_surface, path)
            self.status = f"Saved plot to {os.path.basename(path)}"
        except Exception as e:
            self.status = f"Failed to save plot: {e}"

    def save_state(self):
        path = self.pick_save_path("Save app state", ".json", [("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        payload = {
            "source_files": self.source_files,
            "selected_plot": self.selected_plot,
            "slot_values": self.slot_values,
            "option_values": self.option_values,
            "relationships": [asdict(r) for r in self.relationships],
            "plot_groups": self.plot_groups,
            "column_types": self.column_types,
            "ordinal_orders": self.ordinal_orders,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.status = f"Saved state to {os.path.basename(path)}"
        except Exception as e:
            self.status = f"Failed to save state: {e}"

    def load_state(self):
        path = self.pick_open_path("Load app state", [("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            self.status = f"Failed to load state file: {e}"
            return

        files = payload.get("source_files", [])
        if not isinstance(files, list):
            self.status = "Invalid state file: source_files malformed"
            return

        self.clear_selection(clear_data=True)
        self.source_files = []
        self.load_files(files, append=True)

        self.selected_plot = payload.get("selected_plot")
        if self.selected_plot and self.selected_plot not in self.plot_specs:
            self.selected_plot = None

        self.slot_values = payload.get("slot_values", {})
        self.option_values = payload.get("option_values", {})
        self.column_types = payload.get("column_types", {})
        self.ordinal_orders = payload.get("ordinal_orders", {})

        rels = []
        for r in payload.get("relationships", []):
            try:
                rels.append(Relation(**r))
            except Exception:
                continue
        self.relationships = rels

        pg = payload.get("plot_groups", {})
        for name in self.plot_groups:
            if name in pg and isinstance(pg[name], dict):
                self.plot_groups[name]["collapsed"] = bool(pg[name].get("collapsed", False))

        self.status = f"Loaded state from {os.path.basename(path)}"

    # ------------------------------
    # Utility helpers
    # ------------------------------
    def current_spec(self) -> Optional[PlotSpec]:
        if not self.selected_plot:
            return None
        return self.plot_specs.get(self.selected_plot)

    def draw_text(self, text: str, pos: Tuple[int, int], color=TEXT, font=None):
        if font is None:
            font = self.font
        self.screen.blit(font.render(text, True, color), pos)

    def clip_text(self, text: str, max_w: int, font: pygame.font.Font) -> str:
        if font.size(text)[0] <= max_w:
            return text
        ellipsis = "..."
        for i in range(len(text), 0, -1):
            t = text[:i] + ellipsis
            if font.size(t)[0] <= max_w:
                return t
        return ellipsis

    def draw_button(self, rect: pygame.Rect, label: str, color: Tuple[int, int, int]):
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=1, border_radius=8)
        txt = self.font_sm.render(label, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def parse_input_value(self, raw: str) -> Any:
        raw = raw.strip()
        if raw == "":
            return None
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw

    def parameter_description(self, name: str) -> str:
        return PARAM_DESCRIPTIONS.get(name, f"Parameter '{name}' for selected plot.")

    def fixed_options_for(self, plot_name: str, param_name: str, default: Any = None) -> Optional[List[Any]]:
        if param_name in PLOT_PARAM_OVERRIDES.get(plot_name, {}):
            return PLOT_PARAM_OVERRIDES[plot_name][param_name]
        if param_name in GLOBAL_FIXED_OPTIONS:
            return GLOBAL_FIXED_OPTIONS[param_name]
        if isinstance(default, bool):
            return [True, False]
        return None

    def option_default(self, plot_name: str, opt: str) -> Any:
        if plot_name == "rf_regression":
            return inspect.signature(RandomForestRegressor).parameters[opt].default
        if plot_name == "kmeans_cluster":
            return inspect.signature(KMeans).parameters[opt].default
        if plot_name == "gaussian_mixture":
            return inspect.signature(GaussianMixture).parameters[opt].default
        if plot_name == "pca_plot":
            return inspect.signature(PCA).parameters[opt].default
        if plot_name == "tsne_plot":
            return inspect.signature(TSNE).parameters[opt].default
        if plot_name == "dendrogram":
            defaults = {
                "method": "ward",
                "metric": "euclidean",
                "orientation": "top",
                "truncate_mode": None,
                "p": 30,
                "leaf_rotation": 0,
                "leaf_font_size": 10,
            }
            return defaults.get(opt)
        return inspect.signature(getattr(sns, plot_name)).parameters[opt].default

    def close_menu(self):
        self.menu_target = None
        self.menu_rect = None
        self.menu_items = []

    def close_input_dialog(self):
        self.input_target = None
        self.input_rect = None
        self.input_value = ""

    def encode_col(self, table: str, col: str) -> str:
        return f"{table}::{col}"

    def decode_col(self, value: str) -> Tuple[str, str]:
        if "::" not in value:
            return "", value
        return value.split("::", 1)

    def col_key(self, table: str, col: str) -> str:
        return self.encode_col(table, col)

    def get_column_type(self, table: str, col: str) -> str:
        return self.column_types.get(self.col_key(table, col), "ratio")

    def set_column_type(self, table: str, col: str, col_type: str):
        key = self.col_key(table, col)
        if col_type not in COLUMN_TYPE_COLORS:
            return
        self.column_types[key] = col_type
        if col_type == "ordinal" and key not in self.ordinal_orders:
            self.ordinal_orders[key] = self.default_ordinal_order(table, col)
        if col_type != "ordinal":
            self.ordinal_orders.pop(key, None)

    def cycle_column_type(self, table: str, col: str):
        order = ["ratio", "nominal", "ordinal", "interval"]
        current = self.get_column_type(table, col)
        idx = order.index(current) if current in order else 0
        self.set_column_type(table, col, order[(idx + 1) % len(order)])

    def default_ordinal_order(self, table: str, col: str) -> List[str]:
        df = self.fetch_table_dataframe(table, [col])
        vals = [str(v) for v in df[col].dropna().unique().tolist()] if col in df.columns else []
        if not vals:
            return []
        numeric_pairs = []
        all_numeric = True
        for v in vals:
            try:
                numeric_pairs.append((float(v), v))
            except Exception:
                all_numeric = False
                break
        if all_numeric:
            numeric_pairs.sort(key=lambda x: x[0])
            return [v for _, v in numeric_pairs]
        return sorted(vals, key=lambda x: x.lower())

    def set_cursor_style(self, cursor_key: str):
        if self.last_cursor_key == cursor_key:
            return
        cursor_map = {
            "arrow": pygame.SYSTEM_CURSOR_ARROW,
            "hresize": pygame.SYSTEM_CURSOR_SIZEWE,
            "vresize": pygame.SYSTEM_CURSOR_SIZENS,
            "hand": pygame.SYSTEM_CURSOR_HAND,
        }
        pygame.mouse.set_cursor(pygame.cursors.Cursor(cursor_map.get(cursor_key, pygame.SYSTEM_CURSOR_ARROW)))
        self.last_cursor_key = cursor_key

    def is_column_selected_in_active_slot(self, table: str, col: str) -> bool:
        spec = self.current_spec()
        if not spec or not self.active_slot:
            return False
        encoded = self.encode_col(table, col)
        val = self.slot_values.get(self.active_slot)
        if isinstance(val, list):
            return encoded in val
        return isinstance(val, str) and val == encoded

    def get_related_column_set(self) -> Set[Tuple[str, str]]:
        related: Set[Tuple[str, str]] = set()
        for rel in self.relationships:
            related.add((rel.left_table, rel.left_col))
            related.add((rel.right_table, rel.right_col))
        return related

    def handle_multi_slot_column_click(self, slot: str, table: str, col: str, ctrl: bool, shift: bool):
        spec = self.current_spec()
        if not spec:
            return
        encoded = self.encode_col(table, col)
        existing = list(self.slot_values.get(slot)) if isinstance(self.slot_values.get(slot), list) else []

        if shift and self.last_multi_anchor and self.last_multi_anchor[0] == table:
            cols = [str(c) for c in self.dataframes.get(table, pd.DataFrame()).columns.tolist()]
            if col in cols and self.last_multi_anchor[1] in cols:
                i0 = cols.index(self.last_multi_anchor[1])
                i1 = cols.index(col)
                lo, hi = min(i0, i1), max(i0, i1)
                range_enc = [self.encode_col(table, c) for c in cols[lo : hi + 1]]
                if ctrl or encoded in existing:
                    existing = [e for e in existing if e not in range_enc]
                else:
                    for e in range_enc:
                        if e not in existing:
                            existing.append(e)
                self.slot_values[slot] = existing
                self.status = f"Updated range selection for {slot}"
                self.last_multi_anchor = (table, col)
                return

        if ctrl:
            if encoded in existing:
                existing.remove(encoded)
            else:
                existing.append(encoded)
            self.slot_values[slot] = existing
            self.status = f"Toggled {table}.{col} in {slot}"
        else:
            self.slot_values[slot] = [encoded]
            self.status = f"Selected {table}.{col} for {slot}"

        self.last_multi_anchor = (table, col)

    def clear_plot_state(self):
        spec = self.current_spec()
        self.slot_values = {}
        self.option_values = {}
        self.active_slot = None
        self.last_multi_anchor = None
        if not spec:
            return
        for s in spec.column_slots:
            self.slot_values[s] = [] if s in spec.multi_slots else None
        for op in spec.option_params:
            self.option_values[op] = None

    def select_plot(self, plot_name: str):
        self.selected_plot = plot_name
        self.clear_plot_state()
        self.close_menu()
        self.close_input_dialog()
        self.status = f"Selected plot: {plot_name}"

    # ------------------------------
    # Drawing helpers
    # ------------------------------
    def draw_scrollbar(self, panel: ScrollPanel):
        v = panel.rect
        track = pygame.Rect(v.right - 8, v.y + 2, 6, v.h - 4)
        pygame.draw.rect(self.screen, (60, 66, 78), track, border_radius=4)

        ratio = v.h / max(panel.content_h, 1)
        thumb_h = max(20, int(track.h * ratio))
        max_scroll = max(panel.content_h - v.h, 1)
        thumb_y = track.y + int((panel.scroll_y / max_scroll) * (track.h - thumb_h))
        thumb = pygame.Rect(track.x, thumb_y, track.w, thumb_h)
        pygame.draw.rect(self.screen, (120, 130, 150), thumb, border_radius=4)

    def draw_menu_bar(self):
        pygame.draw.rect(self.screen, MENU_BG, self.menu_bar_rect)
        pygame.draw.line(self.screen, (90, 96, 110), (0, self.menu_bar_rect.bottom), (self.window_w, self.menu_bar_rect.bottom), 1)
        for rect, text, color, _ in self.menu_buttons:
            self.draw_button(rect, text, color)

    def draw_data_panel(self):
        pygame.draw.rect(self.screen, PANEL, self.data_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.data_rect, width=1, border_radius=10)
        self.draw_text("Data", (self.data_rect.x + 12, self.data_rect.y + 10), font=self.font_lg)

        viewport = self.data_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        self.data_layout = []
        self.data_col_ui = {}
        y = viewport.y + 8 - self.data_panel.scroll_y
        row_h = 28
        col_h = 22

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(viewport)
        spec = self.current_spec()

        for table, df in self.dataframes.items():
            h_rect = pygame.Rect(viewport.x + 8, y, viewport.w - 16, row_h)
            marker = "+" if self.table_collapsed.get(table, False) else "-"
            if viewport.colliderect(h_rect):
                pygame.draw.rect(self.screen, (55, 62, 75), h_rect, border_radius=6)
                txt = self.clip_text(f"{marker} {table} ({len(df.columns)} cols)", h_rect.w - 12, self.font_sm)
                self.draw_text(txt, (h_rect.x + 8, h_rect.y + 6), font=self.font_sm)
            self.data_layout.append((h_rect, "header", table, ""))
            y += row_h + 4

            if not self.table_collapsed.get(table, False):
                for col in df.columns:
                    c_rect = pygame.Rect(viewport.x + 18, y, viewport.w - 28, col_h)
                    if viewport.colliderect(c_rect):
                        pygame.draw.rect(self.screen, (70, 78, 92), c_rect, border_radius=5)
                        if spec and self.active_slot and self.is_column_selected_in_active_slot(table, str(col)):
                            pygame.draw.rect(self.screen, ACCENT, c_rect, width=2, border_radius=5)
                        col_type = self.get_column_type(table, str(col))
                        type_color = COLUMN_TYPE_COLORS[col_type]
                        prefix = COLUMN_TYPE_PREFIX[col_type]
                        corner = pygame.Rect(c_rect.x + 2, c_rect.y + 2, 8, 8)
                        pygame.draw.rect(self.screen, type_color, corner, border_radius=2)
                        prefix_rect = pygame.Rect(c_rect.x + 14, c_rect.y + 1, 28, c_rect.h - 2)
                        prefix_surf = self.font_sm.render(prefix, True, type_color)
                        self.screen.blit(prefix_surf, (prefix_rect.x, prefix_rect.y + 1))
                        c_text = self.clip_text(str(col), c_rect.w - 42, self.font_sm)
                        self.draw_text(c_text, (c_rect.x + 40, c_rect.y + 3), font=self.font_sm)
                        self.data_col_ui[(table, str(col))] = {
                            "corner": corner,
                            "prefix": prefix_rect,
                            "box": c_rect,
                        }
                    self.data_layout.append((c_rect, "column", table, str(col)))
                    y += col_h + 3

            y += 4

        self.screen.set_clip(prev_clip)

        self.data_panel.content_h = max(viewport.h, y - viewport.y + self.data_panel.scroll_y + 8)
        self.data_panel.clamp_scroll()
        if self.data_panel.content_h > viewport.h:
            self.draw_scrollbar(self.data_panel)

    def draw_plot_type_panel(self):
        pygame.draw.rect(self.screen, PANEL, self.plot_type_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.plot_type_rect, width=1, border_radius=10)
        self.draw_text("Plot Types", (self.plot_type_rect.x + 12, self.plot_type_rect.y + 10), font=self.font_lg)

        viewport = self.plot_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        self.plot_layout = []
        y = viewport.y + 8 - self.plot_panel.scroll_y
        header_h = 30
        item_h = 32

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(viewport)

        for group_name in ["Seaborn Plots", "AI Analysis"]:
            g_rect = pygame.Rect(viewport.x + 6, y, viewport.w - 12, header_h)
            if viewport.colliderect(g_rect):
                pygame.draw.rect(self.screen, (48, 53, 64), g_rect, border_radius=6)
                marker = "+" if self.plot_groups[group_name]["collapsed"] else "-"
                self.draw_text(f"{marker} {group_name}", (g_rect.x + 8, g_rect.y + 7), font=self.font_sm)
            self.plot_layout.append((g_rect, "group", group_name))
            y += header_h + 4

            if self.plot_groups[group_name]["collapsed"]:
                continue

            for plot_name, spec in self.plot_specs.items():
                if spec.group != group_name:
                    continue
                p_rect = pygame.Rect(viewport.x + 16, y, viewport.w - 24, item_h)
                if viewport.colliderect(p_rect):
                    fill = (61, 91, 133) if plot_name == self.selected_plot else (58, 66, 82)
                    pygame.draw.rect(self.screen, fill, p_rect, border_radius=6)
                    if plot_name == self.selected_plot:
                        pygame.draw.rect(self.screen, ACCENT, p_rect, width=2, border_radius=6)
                    self.draw_text(plot_name, (p_rect.x + 8, p_rect.y + 7), font=self.font_sm)
                self.plot_layout.append((p_rect, "plot", plot_name))
                y += item_h + 5
            y += 4

        self.screen.set_clip(prev_clip)

        self.plot_panel.content_h = max(viewport.h, y - viewport.y + self.plot_panel.scroll_y + 8)
        self.plot_panel.clamp_scroll()
        if self.plot_panel.content_h > viewport.h:
            self.draw_scrollbar(self.plot_panel)

    def draw_builder_area(self):
        pygame.draw.rect(self.screen, PANEL, self.builder_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.builder_rect, width=1, border_radius=10)
        self.draw_text("Inputs", (self.builder_rect.x + 12, self.builder_rect.y + 8), font=self.font_lg)

        spec = self.current_spec()
        header_y = self.builder_rect.y + 34

        self.slot_layout = {}
        self.option_layout = {}
        self.input_header_layout = {}
        if not spec:
            self.draw_text("Select a plot from the right panel.", (self.builder_rect.x + 12, header_y), MUTED, self.font_sm)
            return

        self.draw_text(self.clip_text(spec.description, self.builder_rect.w - 24, self.font_sm), (self.builder_rect.x + 12, header_y), MUTED, self.font_sm)
        self.draw_text("Click an input row, then click/drag columns from Data panel.", (self.builder_rect.x + 12, header_y + 20), MUTED, self.font_sm)
        self.draw_text("Ctrl=toggle multi columns, Shift=range select/deselect.", (self.builder_rect.x + 12, header_y + 38), MUTED, self.font_sm)

        viewport = self.builder_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        y = viewport.y + 8 - self.builder_panel.scroll_y
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(viewport)

        req_header = pygame.Rect(viewport.x + 8, y, viewport.w - 16, 28)
        if viewport.colliderect(req_header):
            pygame.draw.rect(self.screen, (48, 53, 64), req_header, border_radius=6)
            marker = "+" if self.input_sections["required"] else "-"
            self.draw_text(f"{marker} Required Inputs", (req_header.x + 8, req_header.y + 6), font=self.font_sm)
        self.input_header_layout["required"] = req_header
        y += 32

        if not self.input_sections["required"]:
            for slot in spec.column_slots:
                s_rect = pygame.Rect(viewport.x + 8, y, viewport.w - 16, 38)
                if viewport.colliderect(s_rect):
                    pygame.draw.rect(self.screen, (66, 72, 88), s_rect, border_radius=8)
                    border = ACCENT if slot == self.active_slot else (120, 130, 150)
                    pygame.draw.rect(self.screen, border, s_rect, width=1, border_radius=8)

                    val = self.slot_values.get(slot)
                    if isinstance(val, list):
                        shown = ", ".join([f"{self.decode_col(v)[0]}.{self.decode_col(v)[1]}" for v in val]) if val else "[default]"
                    elif isinstance(val, str):
                        t, c = self.decode_col(val)
                        shown = f"{t}.{c}"
                    else:
                        shown = "[default]"

                    req = "*" if slot in spec.required_slots else ""
                    text = f"{slot}{req}: {shown}"
                    self.draw_text(self.clip_text(text, s_rect.w - 12, self.font_sm), (s_rect.x + 8, s_rect.y + 10), font=self.font_sm)
                self.slot_layout[slot] = s_rect
                y += 44

        opt_header = pygame.Rect(viewport.x + 8, y, viewport.w - 16, 28)
        if viewport.colliderect(opt_header):
            pygame.draw.rect(self.screen, (48, 53, 64), opt_header, border_radius=6)
            marker = "+" if self.input_sections["optional"] else "-"
            self.draw_text(f"{marker} Optional Inputs", (opt_header.x + 8, opt_header.y + 6), font=self.font_sm)
        self.input_header_layout["optional"] = opt_header
        y += 32

        if not self.input_sections["optional"]:
            for opt in spec.option_params:
                r = pygame.Rect(viewport.x + 8, y, viewport.w - 16, 32)
                if viewport.colliderect(r):
                    pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=6)
                    val = self.option_values.get(opt)
                    default_val = self.option_default(spec.name, opt)
                    shown_val = default_val if val is None else val
                    border_col = ACCENT if val is not None else (120, 130, 150)
                    pygame.draw.rect(self.screen, border_col, r, width=1, border_radius=6)
                    txt = f"{opt}: {shown_val}"
                    self.draw_text(self.clip_text(txt, r.w - 10, self.font_sm), (r.x + 8, r.y + 7), font=self.font_sm)
                self.option_layout[opt] = r
                y += 36

        self.screen.set_clip(prev_clip)

        self.builder_panel.content_h = max(viewport.h, y - viewport.y + self.builder_panel.scroll_y + 8)
        self.builder_panel.clamp_scroll()
        if self.builder_panel.content_h > viewport.h:
            self.draw_scrollbar(self.builder_panel)

    def draw_chart_area(self):
        pygame.draw.rect(self.screen, PANEL, self.chart_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.chart_rect, width=1, border_radius=10)
        self.draw_text("Preview", (self.chart_rect.x + 12, self.chart_rect.y + 8), font=self.font_lg)

        inner = self.chart_rect.inflate(-18, -42)
        inner.y += 14
        pygame.draw.rect(self.screen, (16, 18, 24), inner, border_radius=8)

        if self.chart_surface:
            scaled = pygame.transform.smoothscale(self.chart_surface, (inner.w, inner.h))
            self.screen.blit(scaled, inner.topleft)
        else:
            self.draw_text("Use Calculate to generate preview.", (inner.x + 10, inner.y + 8), MUTED, self.font_sm)

    def draw_options_area(self):
        pygame.draw.rect(self.screen, PANEL, self.options_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.options_rect, width=1, border_radius=10)
        self.draw_text("Optional Inputs", (self.options_rect.x + 12, self.options_rect.y + 8), font=self.font_lg)

        self.option_layout = {}
        spec = self.current_spec()
        if not spec:
            self.draw_text("Select a plot first.", (self.options_rect.x + 12, self.options_rect.y + 34), MUTED, self.font_sm)
            return

        viewport = self.options_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        y = viewport.y + 6 - self.options_panel.scroll_y
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(viewport)

        for opt in spec.option_params:
            r = pygame.Rect(viewport.x + 6, y, viewport.w - 12, 30)
            if viewport.colliderect(r):
                pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=6)
                val = self.option_values.get(opt)
                border_col = ACCENT if val is not None else (120, 130, 150)
                pygame.draw.rect(self.screen, border_col, r, width=1, border_radius=6)
                shown = "(unset)" if val is None else str(val)
                txt = f"{opt}: {shown}"
                self.draw_text(self.clip_text(txt, r.w - 10, self.font_sm), (r.x + 8, r.y + 7), font=self.font_sm)
            self.option_layout[opt] = r
            y += 35

        self.screen.set_clip(prev_clip)

        self.options_panel.content_h = max(viewport.h, y - viewport.y + self.options_panel.scroll_y + 8)
        self.options_panel.clamp_scroll()
        if self.options_panel.content_h > viewport.h:
            self.draw_scrollbar(self.options_panel)

    def draw_status(self):
        rect = pygame.Rect(max(10, self.window_w - 520), 8, min(510, self.window_w - 20), 34)
        pygame.draw.rect(self.screen, (40, 44, 54), rect, border_radius=8)
        txt = self.clip_text(self.status, rect.w - 12, self.font_sm)
        self.draw_text(txt, (rect.x + 6, rect.y + 8), MUTED, self.font_sm)

    def draw_splitters(self):
        pygame.draw.rect(self.screen, (80, 86, 98), self.left_splitter, border_radius=3)
        pygame.draw.rect(self.screen, (80, 86, 98), self.right_splitter, border_radius=3)
        pygame.draw.rect(self.screen, (80, 86, 98), self.middle_splitter, border_radius=3)

    def draw_dragging(self):
        if not self.drag_item:
            return
        cols = self.drag_item.values if self.drag_item.values else [self.drag_item.value]
        label = f"{self.drag_item.table}: {len(cols)} col(s)" if len(cols) > 1 else f"{self.drag_item.table}.{cols[0]}"
        txt = self.font_sm.render(label, True, (255, 255, 255))
        r = txt.get_rect(topleft=(self.drag_pos[0] + 10, self.drag_pos[1] + 8))
        bg = r.inflate(14, 8)
        pygame.draw.rect(self.screen, (90, 100, 120), bg, border_radius=6)
        self.screen.blit(txt, r)

    # ------------------------------
    # Popups and dialogs
    # ------------------------------
    def open_fixed_option_menu(self, name: str, is_option: bool, choices: List[Any]):
        self.close_input_dialog()
        h = min(self.window_h - 80, 50 + len(choices) * 34)
        w = min(420, self.builder_rect.w - 20)
        self.menu_rect = pygame.Rect(self.builder_rect.centerx - w // 2, self.builder_rect.centery - h // 2, w, h)
        self.menu_target = (name, is_option)
        self._current_menu_values = choices

    def open_input_dialog(self, name: str, is_option: bool, current_value: Any):
        self.close_menu()
        w = min(520, self.builder_rect.w - 20)
        self.input_rect = pygame.Rect(self.builder_rect.centerx - w // 2, self.builder_rect.centery - 70, w, 130)
        self.input_target = (name, is_option)
        self.input_value = "" if current_value is None else str(current_value)

    def draw_menu_popup(self):
        if not self.menu_target or not self.menu_rect:
            return
        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((5, 8, 14, 120))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, PANEL, self.menu_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.menu_rect, width=1, border_radius=10)

        label = self.menu_target[0]
        self.draw_text(f"Select value for {label}", (self.menu_rect.x + 12, self.menu_rect.y + 10), font=self.font_sm)

        self.menu_items = []
        y = self.menu_rect.y + 38
        clip = self.screen.get_clip()
        self.screen.set_clip(self.menu_rect)
        for v in self._current_menu_values:
            r = pygame.Rect(self.menu_rect.x + 12, y, self.menu_rect.w - 24, 30)
            pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=6)
            self.draw_text(self.clip_text(str(v), r.w - 10, self.font_sm), (r.x + 8, r.y + 7), font=self.font_sm)
            self.menu_items.append((r, v))
            y += 34
        self.screen.set_clip(clip)

    def draw_input_dialog(self):
        if not self.input_target or not self.input_rect:
            return
        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((5, 8, 14, 120))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, PANEL, self.input_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.input_rect, width=1, border_radius=10)

        label = self.input_target[0]
        self.draw_text(f"Type value for {label}", (self.input_rect.x + 10, self.input_rect.y + 10), font=self.font_sm)
        self.draw_text("Enter=Save Esc=Cancel", (self.input_rect.x + 10, self.input_rect.y + 32), MUTED, self.font_sm)

        field = pygame.Rect(self.input_rect.x + 10, self.input_rect.y + 58, self.input_rect.w - 20, 34)
        pygame.draw.rect(self.screen, (18, 21, 27), field, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT, field, width=1, border_radius=6)
        self.draw_text(self.clip_text(self.input_value, field.w - 10, self.font_sm), (field.x + 8, field.y + 8), font=self.font_sm)

    def open_col_type_menu(self, table: str, col: str, pos: Tuple[int, int]):
        self.col_type_menu_target = (table, col)
        self.col_type_menu_rect = pygame.Rect(pos[0], pos[1], 210, 170)
        if self.col_type_menu_rect.right > self.window_w - 8:
            self.col_type_menu_rect.x = self.window_w - self.col_type_menu_rect.w - 8
        if self.col_type_menu_rect.bottom > self.window_h - 8:
            self.col_type_menu_rect.y = self.window_h - self.col_type_menu_rect.h - 8
        self.col_type_menu_items = []

    def close_col_type_menu(self):
        self.col_type_menu_target = None
        self.col_type_menu_rect = None
        self.col_type_menu_items = []

    def draw_col_type_menu(self):
        if not self.col_type_menu_target or not self.col_type_menu_rect:
            return
        table, col = self.col_type_menu_target
        rect = self.col_type_menu_rect
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=1, border_radius=8)
        self.draw_text(self.clip_text(f"{table}.{col}", rect.w - 10, self.font_sm), (rect.x + 6, rect.y + 6), MUTED, self.font_sm)
        self.col_type_menu_items = []
        y = rect.y + 30
        for t in ["nominal", "ordinal", "interval", "ratio"]:
            r = pygame.Rect(rect.x + 8, y, rect.w - 16, 26)
            pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=5)
            color = COLUMN_TYPE_COLORS[t]
            self.draw_text(COLUMN_TYPE_PREFIX[t] + t.title(), (r.x + 6, r.y + 5), color, self.font_sm)
            self.col_type_menu_items.append((r, f"type::{t}"))
            y += 30
        if self.get_column_type(table, col) == "ordinal":
            r = pygame.Rect(rect.x + 8, y, rect.w - 16, 26)
            pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=5)
            self.draw_text("Set Ordinal Order...", (r.x + 6, r.y + 5), ACCENT, self.font_sm)
            self.col_type_menu_items.append((r, "set_order"))

    # ------------------------------
    # Relationship editor modal
    # ------------------------------
    def relation_exists(self, a_table: str, a_col: str, b_table: str, b_col: str) -> bool:
        for r in self.relationships:
            if (
                r.left_table == a_table
                and r.left_col == a_col
                and r.right_table == b_table
                and r.right_col == b_col
            ) or (
                r.left_table == b_table
                and r.left_col == b_col
                and r.right_table == a_table
                and r.right_col == a_col
            ):
                return True
        return False

    def cycle_join_mode(self, idx: int):
        if idx < 0 or idx >= len(self.relationships):
            return
        r = self.relationships[idx]
        r.click_count += 1
        if r.click_count > len(REL_JOIN_ORDER):
            removed = self.relationships.pop(idx)
            self.status = (
                f"Removed relationship: {removed.left_table}.{removed.left_col} <-> "
                f"{removed.right_table}.{removed.right_col}"
            )
            return
        current = REL_JOIN_ORDER.index(r.join_mode) if r.join_mode in REL_JOIN_ORDER else 0
        r.join_mode = REL_JOIN_ORDER[(current + 1) % len(REL_JOIN_ORDER)]
        self.status = f"Relationship join mode changed to {r.join_mode}"

    def draw_arrow_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], width: int = 3):
        pygame.draw.line(self.screen, color, start, end, width)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 2:
            return
        ux, uy = dx / length, dy / length
        arrow_len = 10
        arrow_w = 6
        tip = end
        left = (
            int(end[0] - arrow_len * ux + arrow_w * uy),
            int(end[1] - arrow_len * uy - arrow_w * ux),
        )
        right = (
            int(end[0] - arrow_len * ux - arrow_w * uy),
            int(end[1] - arrow_len * uy + arrow_w * ux),
        )
        pygame.draw.polygon(self.screen, color, [tip, left, right])

    def point_to_segment_distance(self, p: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx = ax + t * dx
        qy = ay + t * dy
        return math.hypot(px - qx, py - qy)

    def draw_relationship_editor(self):
        if not self.rel_editor_open:
            return

        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))

        w = min(1200, self.window_w - 60)
        h = min(760, self.window_h - 80)
        self.rel_modal_rect = pygame.Rect((self.window_w - w) // 2, (self.window_h - h) // 2, w, h)

        pygame.draw.rect(self.screen, REL_MODAL_BG, self.rel_modal_rect, border_radius=12)
        pygame.draw.rect(self.screen, (255, 255, 255), self.rel_modal_rect, width=1, border_radius=12)

        self.draw_text("Relationships Editor (Esc to close)", (self.rel_modal_rect.x + 14, self.rel_modal_rect.y + 12), font=self.font_lg)
        self.rel_clear_btn = pygame.Rect(self.rel_modal_rect.x + 390, self.rel_modal_rect.y + 12, 150, 28)
        self.draw_button(self.rel_clear_btn, "Clear Relationships", BAD)

        canvas = self.rel_modal_rect.inflate(-20, -70)
        canvas.y += 30
        self.rel_canvas_rect = canvas
        pygame.draw.rect(self.screen, PANEL_ALT, canvas, border_radius=8)

        # Compute table boxes on a grid and clip all content to canvas to avoid spill-over.
        self.rel_table_boxes = {}
        self.rel_column_boxes = []
        self.rel_line_layout = []

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(canvas)

        tables = list(self.dataframes.keys())
        cols = max(1, min(4, len(tables)))
        card_w = max(220, (canvas.w - (cols + 1) * 14) // cols)
        gap_y = 20
        table_box_h: Dict[str, int] = {}
        for t in tables:
            # Full column list is rendered; canvas scrolling handles overflow.
            table_box_h[t] = max(240, 42 + len(self.dataframes[t].columns) * 23 + 8)
        rows = max(1, math.ceil(max(1, len(tables)) / cols))
        row_heights: List[int] = []
        for r in range(rows):
            row_tables = tables[r * cols : (r + 1) * cols]
            row_heights.append(max([table_box_h[t] for t in row_tables], default=240))
        content_h = sum(row_heights) + max(0, rows - 1) * gap_y + 28
        self.rel_scroll_max = max(0, content_h - canvas.h)
        self.rel_scroll_y = max(0, min(self.rel_scroll_y, self.rel_scroll_max))
        x0 = canvas.x + 14
        y0 = canvas.y + 14 - self.rel_scroll_y
        related_cols = self.get_related_column_set()

        row_top_cache: Dict[int, int] = {}
        running_y = y0
        for r in range(rows):
            row_top_cache[r] = running_y
            running_y += row_heights[r] + gap_y

        for idx, table in enumerate(tables):
            c = idx % cols
            r = idx // cols
            x = x0 + c * (card_w + 14)
            y = row_top_cache.get(r, y0)
            box_h = table_box_h.get(table, 240)
            box = pygame.Rect(x, y, card_w, box_h)
            self.rel_table_boxes[table] = box

            pygame.draw.rect(self.screen, (50, 60, 76), box, border_radius=8)
            pygame.draw.rect(self.screen, (120, 130, 150), box, width=1, border_radius=8)
            self.draw_text(self.clip_text(table, box.w - 10, self.font_sm), (box.x + 6, box.y + 6), font=self.font_sm)

            col_y = box.y + 30
            for i, col in enumerate(self.dataframes[table].columns):
                cr = pygame.Rect(box.x + 6, col_y, box.w - 12, 20)
                pygame.draw.rect(self.screen, (72, 82, 100), cr, border_radius=4)
                if (table, str(col)) in related_cols:
                    pygame.draw.rect(self.screen, ACCENT, cr, width=2, border_radius=4)
                self.draw_text(self.clip_text(str(col), cr.w - 10, self.font_sm), (cr.x + 5, cr.y + 2), font=self.font_sm)
                self.rel_column_boxes.append((cr, table, str(col)))
                col_y += 23

        # Draw existing relationship lines.
        for i, rel in enumerate(self.relationships):
            a = self._column_box_center(rel.left_table, rel.left_col)
            b = self._column_box_center(rel.right_table, rel.right_col)
            if not a or not b:
                continue
            color = REL_JOIN_COLORS.get(rel.join_mode, (200, 200, 200))
            self.draw_arrow_line(a, b, color, 3)
            self.rel_line_layout.append((a, b, i))

        # Draw relationship drag preview line.
        if self.rel_drag_start:
            st = self._column_box_center(self.rel_drag_start[0], self.rel_drag_start[1])
            if st:
                pygame.draw.line(self.screen, ACCENT, st, pygame.mouse.get_pos(), 2)

        self.screen.set_clip(prev_clip)

        # Legend for line colors / join modes.
        legend = pygame.Rect(self.rel_modal_rect.right - 210, self.rel_modal_rect.y + 12, 190, 120)
        pygame.draw.rect(self.screen, (40, 46, 56), legend, border_radius=8)
        pygame.draw.rect(self.screen, (120, 130, 150), legend, width=1, border_radius=8)
        self.draw_text("Join Mode Colors", (legend.x + 8, legend.y + 8), font=self.font_sm)
        y = legend.y + 32
        for mode in REL_JOIN_ORDER:
            color = REL_JOIN_COLORS[mode]
            self.draw_arrow_line((legend.x + 12, y + 7), (legend.x + 42, y + 7), color, 3)
            self.draw_text(mode, (legend.x + 50, y), font=self.font_sm)
            y += 20

    def _column_box_center(self, table: str, col: str) -> Optional[Tuple[int, int]]:
        for rect, t, c in self.rel_column_boxes:
            if t == table and c == col:
                return rect.center
        return None

    # ------------------------------
    # Assignment and join logic
    # ------------------------------
    def assign_columns_to_slot(self, slot: str, table: str, columns: List[str]):
        spec = self.current_spec()
        if not spec or not columns:
            return

        encoded = [self.encode_col(table, c) for c in columns]

        if slot in spec.multi_slots:
            existing = self.slot_values.get(slot)
            merged = list(existing) if isinstance(existing, list) else []
            for e in encoded:
                if e not in merged:
                    merged.append(e)
            self.slot_values[slot] = merged
            self.status = f"Assigned {len(columns)} column(s) to {slot}"
        else:
            self.slot_values[slot] = encoded[0]
            self.status = f"Assigned {table}.{columns[0]} to {slot}"

        # Keep previous rendered chart until user explicitly recalculates.

    def gather_used_tables(self, spec: PlotSpec) -> Set[str]:
        used: Set[str] = set()
        for slot in spec.column_slots:
            v = self.slot_values.get(slot)
            if isinstance(v, list):
                for e in v:
                    t, _ = self.decode_col(e)
                    if t:
                        used.add(t)
            elif isinstance(v, str):
                t, _ = self.decode_col(v)
                if t:
                    used.add(t)
        return used

    def _reverse_join_mode(self, mode: str) -> str:
        if mode == "left":
            return "right"
        if mode == "right":
            return "left"
        return mode

    def build_joined_dataframe(self, used_tables: Set[str]) -> Tuple[Optional[pd.DataFrame], str]:
        if not used_tables:
            if not self.dataframes:
                return None, "No data loaded."
            # Fallback: allow plots that can run on whole dataframe with no explicit column slots.
            used_tables = {next(iter(self.dataframes.keys()))}

        root = next(iter(used_tables))
        if root not in self.table_sources:
            return None, f"Table '{root}' not found."

        # SQLite lazy path: every plotting request issues fresh SQL and joins through temp views.
        sqlite_paths = {self.table_sources[t].file_path for t in used_tables if self.table_sources.get(t) and self.table_sources[t].kind == "sqlite"}
        all_sqlite = len(sqlite_paths) == 1 and len(used_tables) == len([t for t in used_tables if self.table_sources[t].kind == "sqlite"])
        if all_sqlite:
            db_path = next(iter(sqlite_paths))
            if db_path:
                conn = sqlite3.connect(db_path)
                try:
                    table_views: Dict[str, str] = {}
                    for idx, t in enumerate(sorted(used_tables)):
                        src = self.table_sources[t]
                        vname = f"_src_{idx}"
                        select_parts = [f'"{c}" AS "{t}.{c}"' for c in src.columns]
                        conn.execute(f'DROP VIEW IF EXISTS "{vname}"')
                        conn.execute(
                            f'CREATE TEMP VIEW "{vname}" AS SELECT {", ".join(select_parts)} FROM "{src.sqlite_table}"'
                        )
                        table_views[t] = vname

                    current_view = table_views[root]
                    in_graph = {root}
                    pending = set(used_tables) - in_graph
                    join_idx = 0

                    while pending:
                        merged_any = False
                        for table in list(pending):
                            connector = None
                            reverse = False
                            for rel in self.relationships:
                                if rel.left_table in in_graph and rel.right_table == table:
                                    connector = rel
                                    reverse = False
                                    break
                                if rel.right_table in in_graph and rel.left_table == table:
                                    connector = rel
                                    reverse = True
                                    break
                            if connector is None:
                                continue

                            if not reverse:
                                left_key = f"{connector.left_table}.{connector.left_col}"
                                right_key = f"{connector.right_table}.{connector.right_col}"
                                how = connector.join_mode.upper()
                            else:
                                left_key = f"{connector.right_table}.{connector.right_col}"
                                right_key = f"{connector.left_table}.{connector.left_col}"
                                how = self._reverse_join_mode(connector.join_mode).upper()

                            next_view = f"_join_{join_idx}"
                            join_idx += 1
                            conn.execute(f'DROP VIEW IF EXISTS "{next_view}"')
                            conn.execute(
                                f'CREATE TEMP VIEW "{next_view}" AS '
                                f'SELECT j.*, t.* FROM "{current_view}" j '
                                f'{how} JOIN "{table_views[table]}" t ON j."{left_key}" = t."{right_key}"'
                            )
                            current_view = next_view
                            in_graph.add(table)
                            pending.remove(table)
                            merged_any = True

                        if not merged_any:
                            return None, "Selected tables are not fully connected by relationships."

                    return pd.read_sql_query(f'SELECT * FROM "{current_view}"', conn), ""
                finally:
                    conn.close()

        current = self.fetch_table_dataframe(root)
        current.columns = [f"{root}.{c}" for c in current.columns]
        in_graph = {root}

        pending = set(used_tables) - in_graph

        while pending:
            merged_any = False
            for table in list(pending):
                connector = None
                reverse = False
                for rel in self.relationships:
                    if rel.left_table in in_graph and rel.right_table == table:
                        connector = rel
                        reverse = False
                        break
                    if rel.right_table in in_graph and rel.left_table == table:
                        connector = rel
                        reverse = True
                        break
                if connector is None:
                    continue

                new_df = self.fetch_table_dataframe(table)
                new_df.columns = [f"{table}.{c}" for c in new_df.columns]

                if not reverse:
                    left_key = f"{connector.left_table}.{connector.left_col}"
                    right_key = f"{connector.right_table}.{connector.right_col}"
                    how = connector.join_mode
                else:
                    left_key = f"{connector.right_table}.{connector.right_col}"
                    right_key = f"{connector.left_table}.{connector.left_col}"
                    how = self._reverse_join_mode(connector.join_mode)

                if left_key not in current.columns:
                    continue
                if right_key not in new_df.columns:
                    continue

                current = pd.merge(current, new_df, how=how, left_on=left_key, right_on=right_key)
                in_graph.add(table)
                pending.remove(table)
                merged_any = True

            if not merged_any:
                return None, "Selected tables are not fully connected by relationships."

        return current, ""

    def _collect_seaborn_kwargs(self, spec: PlotSpec, merged_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {}

        for slot in spec.column_slots:
            v = self.slot_values.get(slot)
            if v is None or v == []:
                continue

            if isinstance(v, list):
                mapped = []
                for e in v:
                    t, c = self.decode_col(e)
                    cname = f"{t}.{c}" if t else c
                    if cname not in merged_df.columns:
                        self.status = f"Column not found after join: {cname}"
                        return None
                    mapped.append(cname)
                kwargs[slot] = mapped
            else:
                t, c = self.decode_col(v)
                cname = f"{t}.{c}" if t else c
                if cname not in merged_df.columns:
                    self.status = f"Column not found after join: {cname}"
                    return None
                kwargs[slot] = cname

        for op in spec.option_params:
            val = self.option_values.get(op)
            if val is None:
                continue
            if op == "estimator" and isinstance(val, str):
                estimators = {
                    "mean": np.mean,
                    "median": np.median,
                    "sum": np.sum,
                    "min": np.min,
                    "max": np.max,
                }
                kwargs[op] = estimators.get(val, val)
            else:
                kwargs[op] = val

        return kwargs

    def _is_figure_level(self, spec: PlotSpec) -> bool:
        return spec.name in {"jointplot", "pairplot", "relplot", "displot", "catplot", "lmplot", "clustermap"}

    def _render_figure_to_surface(self, fig) -> pygame.Surface:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        raw = fig.canvas.buffer_rgba()
        surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
        return surf.copy()

    def _calculate_rf_regression(self, merged_df: pd.DataFrame):
        feats = self.slot_values.get("feature_columns")
        target = self.slot_values.get("target_column")

        if not isinstance(feats, list) or not feats:
            self.status = "Random forest requires feature_columns."
            return
        if not isinstance(target, str):
            self.status = "Random forest requires target_column."
            return

        feature_cols = []
        for e in feats:
            t, c = self.decode_col(e)
            feature_cols.append(f"{t}.{c}" if t else c)

        t_t, t_c = self.decode_col(target)
        target_col = f"{t_t}.{t_c}" if t_t else t_c

        for c in feature_cols + [target_col]:
            if c not in merged_df.columns:
                self.status = f"RF column not found after join: {c}"
                return

        X = merged_df[feature_cols].copy()
        y = merged_df[target_col]

        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < 10:
            self.status = "Need at least 10 complete rows for RF regression."
            return

        model_kwargs = {k: v for k, v in self.option_values.items() if v is not None}

        nominal_cols: List[str] = []
        ordinal_cols: List[str] = []
        numeric_cols: List[str] = []
        ordinal_categories: List[List[str]] = []
        for full_col in feature_cols:
            table, col = full_col.split(".", 1) if "." in full_col else ("", full_col)
            ctype = self.get_column_type(table, col) if table else "ratio"
            if ctype == "nominal":
                nominal_cols.append(full_col)
            elif ctype == "ordinal":
                ordinal_cols.append(full_col)
                ord_vals = self.ordinal_orders.get(self.col_key(table, col), [])
                if not ord_vals:
                    ord_vals = self.default_ordinal_order(table, col)
                ordinal_categories.append([str(v) for v in ord_vals] if ord_vals else sorted([str(v) for v in X[full_col].dropna().unique().tolist()]))
            else:
                numeric_cols.append(full_col)

        transformers = []
        if nominal_cols:
            transformers.append(("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_cols))
        if ordinal_cols:
            transformers.append(
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=ordinal_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ordinal_cols,
                )
            )
        if numeric_cols:
            transformers.append(("numeric", "passthrough", numeric_cols))
        pre = ColumnTransformer(transformers=transformers, remainder="drop")

        try:
            bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
            if bins.nunique() >= 2:
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                folds = splitter.split(X, bins)
            else:
                splitter = KFold(n_splits=5, shuffle=True, random_state=42)
                folds = splitter.split(X)
        except Exception:
            splitter = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = splitter.split(X)

        scores: List[float] = []
        for tr, te in folds:
            pre_fold = pre
            X_train = X.iloc[tr].copy()
            X_test = X.iloc[te].copy()
            for c in numeric_cols:
                X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
                X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
            X_train_enc = pre_fold.fit_transform(X_train)
            X_test_enc = pre_fold.transform(X_test)
            m = RandomForestRegressor(**model_kwargs)
            m.fit(X_train_enc, y.iloc[tr])
            scores.append(float(m.score(X_test_enc, y.iloc[te])))

        final_model = RandomForestRegressor(**model_kwargs)
        X_fit = X.copy()
        for c in numeric_cols:
            X_fit[c] = pd.to_numeric(X_fit[c], errors="coerce")
        X_enc = pre.fit_transform(X_fit)
        final_model.fit(X_enc, y)

        try:
            transformed_names = list(pre.get_feature_names_out())
        except Exception:
            transformed_names = [f"f{i}" for i in range(len(final_model.feature_importances_))]
        agg_imp: Dict[str, float] = {c: 0.0 for c in feature_cols}
        for n, v in zip(transformed_names, final_model.feature_importances_):
            matched = None
            for c in feature_cols:
                if c in n:
                    matched = c
                    break
            if matched is None:
                matched = feature_cols[0]
            agg_imp[matched] += float(v)
        imp = pd.Series(agg_imp).sort_values(ascending=False)
        display_labels = []
        seen: Dict[str, int] = {}
        for full_name in imp.index.tolist():
            table, col = full_name.split(".", 1) if "." in full_name else ("", full_name)
            base = col
            if base in seen:
                seen[base] += 1
                base = f"{col} ({table})"
            else:
                seen[base] = 1
            display_labels.append(base)

        accuracy = float(np.mean(scores))
        accuracy = max(0.0, min(1.0, accuracy))

        plt.close("all")
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(8, 3.8), dpi=120, gridspec_kw={"height_ratios": [1.1, 2.2]}
        )

        # Top accuracy bar from 0% to 100%.
        ax_top.barh([0], [100], color="#2a2f3a", alpha=0.55, edgecolor="white")
        ax_top.barh([0], [accuracy * 100], color="#4da3ff", edgecolor="white")
        ax_top.set_xlim(0, 100)
        ax_top.set_yticks([])
        ax_top.set_xlabel("Model Accuracy (%)")
        ax_top.set_title(f"Random Forest Overall Accuracy: {accuracy * 100:.1f}%")

        # Bottom flow-like stacked contribution bar.
        contrib = imp / max(imp.sum(), 1e-12) * (accuracy * 100)
        left = 0.0
        colors = sns.color_palette("Blues", n_colors=max(3, len(contrib) + 2))[2:]
        for i, (name, val) in enumerate(contrib.items()):
            ax_bottom.barh([0], [val], left=left, color=colors[i % len(colors)], edgecolor="white")
            if val > 3:
                ax_bottom.text(left + val / 2, 0, display_labels[i], ha="center", va="center", color="white", fontsize=8)
            left += val
        ax_bottom.barh([0], [100 - accuracy * 100], left=accuracy * 100, color="#2a2f3a", alpha=0.35, edgecolor="white")
        ax_bottom.set_xlim(0, 100)
        ax_bottom.set_yticks([])
        ax_bottom.set_xlabel("Feature Contribution Flow To Accuracy (%)")
        ax_bottom.set_title("Feature Importance Flow")
        fig.tight_layout()

        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "Random forest plot generated successfully."

    def _resolve_slot_columns(self, merged_df: pd.DataFrame, slot_name: str) -> List[str]:
        raw = self.slot_values.get(slot_name)
        values = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) else [])
        cols: List[str] = []
        for v in values:
            t, c = self.decode_col(v)
            cname = f"{t}.{c}" if t else c
            if cname in merged_df.columns:
                cols.append(cname)
        return cols

    def _resolve_single_slot_column(self, merged_df: pd.DataFrame, slot_name: str) -> Optional[str]:
        cols = self._resolve_slot_columns(merged_df, slot_name)
        if cols:
            return cols[0]
        return None

    def _calculate_dendrogram(self, merged_df: pd.DataFrame):
        if dendrogram is None or linkage is None:
            self.status = "SciPy is required for dendrogram plotting but is unavailable."
            return
        cols = self._resolve_slot_columns(merged_df, "feature_columns")
        if len(cols) < 2:
            self.status = "Dendrogram requires at least two feature columns."
            return

        data = merged_df[cols].apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 2:
            self.status = "Dendrogram requires at least two complete numeric rows."
            return

        method = str(self.option_values.get("method") or "ward")
        metric = str(self.option_values.get("metric") or "euclidean")
        if method == "ward" and metric != "euclidean":
            metric = "euclidean"
        orientation = self.option_values.get("orientation") or "top"
        truncate_mode = self.option_values.get("truncate_mode")
        p = self.option_values.get("p")
        leaf_rotation = self.option_values.get("leaf_rotation")
        leaf_font_size = self.option_values.get("leaf_font_size")

        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        try:
            Z = linkage(data.values, method=method, metric=metric)
        except Exception as e:
            self.status = f"Dendrogram failed during linkage: {e}"
            plt.close(fig)
            return
        dendrogram(
            Z,
            ax=ax,
            orientation=orientation,
            truncate_mode=truncate_mode,
            p=30,
            leaf_rotation=leaf_rotation if leaf_rotation is not None else 0,
            leaf_font_size=leaf_font_size if leaf_font_size is not None else 9,
            no_labels=True,
        )
        ax.set_title("Dendrogram")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "Dendrogram generated successfully."

    def _calculate_kmeans_cluster(self, merged_df: pd.DataFrame):
        cols = self._resolve_slot_columns(merged_df, "data_columns")
        if len(cols) < 1 or len(cols) > 2:
            self.status = "KMeans requires one or two data_columns."
            return

        data = merged_df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 2:
            self.status = "KMeans requires complete numeric data."
            return

        model_kwargs = {k: v for k, v in self.option_values.items() if v is not None}
        model = KMeans(**model_kwargs)
        labels = model.fit_predict(data.values)

        plot_df = data.copy()
        plot_df["_group"] = labels.astype(str)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        if len(cols) == 1:
            sns.histplot(data=plot_df, x=cols[0], hue="_group", common_norm=False, element="step", ax=ax)
        else:
            sns.scatterplot(data=plot_df, x=cols[0], y=cols[1], hue="_group", ax=ax)
        ax.set_title("KMeans Clusters")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "KMeans plot generated successfully."

    def _gmm_covariance_matrix(self, model: GaussianMixture, component_idx: int) -> np.ndarray:
        cov_type = model.covariance_type
        covs = model.covariances_
        if cov_type == "full":
            return covs[component_idx]
        if cov_type == "tied":
            return covs
        if cov_type == "diag":
            return np.diag(covs[component_idx])
        if cov_type == "spherical":
            return np.eye(model.means_.shape[1]) * covs[component_idx]
        return np.eye(model.means_.shape[1])

    def _calculate_gaussian_mixture(self, merged_df: pd.DataFrame):
        cols = self._resolve_slot_columns(merged_df, "data_columns")
        if len(cols) < 1 or len(cols) > 2:
            self.status = "Gaussian mixture requires one or two data_columns."
            return

        data = merged_df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 2:
            self.status = "Gaussian mixture requires complete numeric data."
            return

        model_kwargs = {k: v for k, v in self.option_values.items() if v is not None}
        model = GaussianMixture(**model_kwargs)
        labels = model.fit_predict(data.values)

        plot_df = data.copy()
        plot_df["_group"] = labels.astype(str)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        if len(cols) == 1:
            sns.histplot(data=plot_df, x=cols[0], hue="_group", common_norm=False, element="step", ax=ax)
        else:
            sns.scatterplot(data=plot_df, x=cols[0], y=cols[1], hue="_group", ax=ax)
            for i in range(model.n_components):
                cov = self._gmm_covariance_matrix(model, i)
                if cov.shape != (2, 2):
                    continue
                vals, vecs = np.linalg.eigh(cov)
                vals = np.clip(vals, 1e-9, None)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                width, height = 2.0 * np.sqrt(vals[0]), 2.0 * np.sqrt(vals[1])
                e = Ellipse(
                    xy=(model.means_[i, 0], model.means_[i, 1]),
                    width=width,
                    height=height,
                    angle=angle,
                    fill=False,
                    linestyle=":",
                    linewidth=1.8,
                    edgecolor="black",
                    alpha=0.8,
                )
                ax.add_patch(e)
        ax.set_title("Gaussian Mixture Clusters")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "Gaussian mixture plot generated successfully."

    def _calculate_grid_plot(self, merged_df: pd.DataFrame, kind: str):
        x_col = self._resolve_single_slot_column(merged_df, "x")
        y_col = self._resolve_single_slot_column(merged_df, "y")
        z_col = self._resolve_single_slot_column(merged_df, "z")
        if not x_col or not y_col or not z_col:
            self.status = f"{kind} requires x, y, and z."
            return

        data = merged_df[[x_col, y_col, z_col]].copy()
        data[z_col] = pd.to_numeric(data[z_col], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            self.status = f"{kind} requires numeric z data."
            return

        grid = data.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
        if grid.empty:
            self.status = f"{kind} grid is empty after pivot."
            return

        plt.close("all")
        if kind == "heatmap":
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
            sns.heatmap(grid, ax=ax)
            ax.set_title("Heatmap")
            fig.tight_layout()
        elif kind == "clustermap":
            g = sns.clustermap(grid)
            fig = g.fig
            fig.set_size_inches(8, 3.2)
            fig.suptitle("Clustermap")
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
            arr = grid.values
            gy, gx = np.gradient(arr)
            xx, yy = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
            ax.quiver(xx, yy, gx, gy, arr, cmap="viridis")
            ax.invert_yaxis()
            ax.set_title("Quiver Plot (Grid Gradient)")
            fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = f"{kind} generated successfully."

    def _calculate_parallel_lines(self, merged_df: pd.DataFrame):
        feature_cols = self._resolve_slot_columns(merged_df, "feature_columns")
        if len(feature_cols) < 2:
            self.status = "parallel_lines requires at least two feature_columns."
            return
        class_col = self._resolve_single_slot_column(merged_df, "class_column")
        cols = feature_cols + ([class_col] if class_col else [])
        df = merged_df[cols].copy().dropna()
        if df.empty:
            self.status = "parallel_lines has no complete rows."
            return
        if class_col is None:
            class_col = "_group"
            df[class_col] = "all"
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        parallel_coordinates(df, class_col=class_col, cols=feature_cols, ax=ax, alpha=0.45)
        ax.set_title("Parallel Lines Plot")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "parallel_lines generated successfully."

    def _calculate_pca_plot(self, merged_df: pd.DataFrame):
        feature_cols = self._resolve_slot_columns(merged_df, "feature_columns")
        if len(feature_cols) < 2:
            self.status = "pca_plot requires at least two feature_columns."
            return
        class_col = self._resolve_single_slot_column(merged_df, "class_column")
        df = merged_df[feature_cols + ([class_col] if class_col else [])].copy().dropna()
        if df.empty:
            self.status = "pca_plot has no complete rows."
            return
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if X.empty:
            self.status = "pca_plot requires numeric feature columns."
            return
        kwargs = {k: v for k, v in self.option_values.items() if v is not None}
        kwargs["n_components"] = 2
        model = PCA(**kwargs)
        emb = model.fit_transform(X.values)
        out = pd.DataFrame({"PC1": emb[:, 0], "PC2": emb[:, 1]})
        if class_col and class_col in df.columns:
            out["class"] = df.loc[X.index, class_col].astype(str)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        if "class" in out:
            sns.scatterplot(data=out, x="PC1", y="PC2", hue="class", ax=ax)
        else:
            sns.scatterplot(data=out, x="PC1", y="PC2", ax=ax)
        ax.set_title("PCA Plot")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "pca_plot generated successfully."

    def _calculate_tsne_plot(self, merged_df: pd.DataFrame):
        feature_cols = self._resolve_slot_columns(merged_df, "feature_columns")
        if len(feature_cols) < 2:
            self.status = "tsne_plot requires at least two feature_columns."
            return
        class_col = self._resolve_single_slot_column(merged_df, "class_column")
        df = merged_df[feature_cols + ([class_col] if class_col else [])].copy().dropna()
        if df.empty:
            self.status = "tsne_plot has no complete rows."
            return
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if X.empty:
            self.status = "tsne_plot requires numeric feature columns."
            return
        kwargs = {k: v for k, v in self.option_values.items() if v is not None}
        kwargs["n_components"] = 2
        model = TSNE(**kwargs)
        emb = model.fit_transform(X.values)
        out = pd.DataFrame({"TSNE1": emb[:, 0], "TSNE2": emb[:, 1]})
        if class_col and class_col in df.columns:
            out["class"] = df.loc[X.index, class_col].astype(str)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        if "class" in out:
            sns.scatterplot(data=out, x="TSNE1", y="TSNE2", hue="class", ax=ax)
        else:
            sns.scatterplot(data=out, x="TSNE1", y="TSNE2", ax=ax)
        ax.set_title("t-SNE Plot")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "tsne_plot generated successfully."

    def _validate_categorical_column_for_flow(self, table_col: str) -> bool:
        table, col = table_col.split(".", 1) if "." in table_col else ("", table_col)
        ctype = self.get_column_type(table, col) if table else "ratio"
        return ctype in {"nominal", "ordinal"}

    def _calculate_markov_chain(self, merged_df: pd.DataFrame):
        f_col = self._resolve_single_slot_column(merged_df, "feature_column")
        t_col = self._resolve_single_slot_column(merged_df, "target_column")
        if not f_col or not t_col:
            self.status = "markov_chain requires feature_column and target_column."
            return
        if not self._validate_categorical_column_for_flow(f_col) or not self._validate_categorical_column_for_flow(t_col):
            self.status = "markov_chain requires nominal or ordinal columns."
            return

        data = merged_df[[f_col, t_col]].dropna()
        if data.empty:
            self.status = "markov_chain has no complete rows."
            return
        prob = pd.crosstab(data[f_col], data[t_col], normalize="index")
        features = [str(v) for v in prob.index.tolist()]
        targets = [str(v) for v in prob.columns.tolist()]

        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        ax.set_axis_off()

        fx = 0.15
        tx = 0.85
        fpos = {v: 0.1 + i * (0.8 / max(1, len(features) - 1)) for i, v in enumerate(features)}
        tpos = {v: 0.1 + i * (0.8 / max(1, len(targets) - 1)) for i, v in enumerate(targets)}

        for v, y in fpos.items():
            ax.text(fx, y, v, transform=ax.transAxes, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="#44506a", ec="white"))
        for v, y in tpos.items():
            ax.text(tx, y, v, transform=ax.transAxes, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="#4e5f80", ec="white"))

        for fv in features:
            for tv in targets:
                p = float(prob.loc[fv, tv])
                if p <= 0:
                    continue
                ax.annotate(
                    f"{p * 100:.1f}%",
                    xy=(tx - 0.04, tpos[tv]),
                    xytext=(fx + 0.04, fpos[fv]),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="#9ec2ff", lw=1.5),
                    fontsize=7,
                    color="white",
                )
        ax.set_title("Markov Chain Feature -> Target")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "markov_chain generated successfully."

    def _calculate_sankey_plot(self, merged_df: pd.DataFrame):
        f_col = self._resolve_single_slot_column(merged_df, "feature_column")
        t_col = self._resolve_single_slot_column(merged_df, "target_column")
        if not f_col or not t_col:
            self.status = "sankey_plot requires feature_column and target_column."
            return
        if not self._validate_categorical_column_for_flow(f_col) or not self._validate_categorical_column_for_flow(t_col):
            self.status = "sankey_plot requires nominal or ordinal columns."
            return

        data = merged_df[[f_col, t_col]].dropna()
        if data.empty:
            self.status = "sankey_plot has no complete rows."
            return
        cnt = pd.crosstab(data[f_col], data[t_col])
        features = [str(v) for v in cnt.index.tolist()]
        targets = [str(v) for v in cnt.columns.tolist()]
        max_flow = max(1, int(cnt.values.max()))

        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        ax.set_axis_off()
        fx = 0.15
        tx = 0.85
        fpos = {v: 0.1 + i * (0.8 / max(1, len(features) - 1)) for i, v in enumerate(features)}
        tpos = {v: 0.1 + i * (0.8 / max(1, len(targets) - 1)) for i, v in enumerate(targets)}

        for v, y in fpos.items():
            ax.text(fx, y, v, transform=ax.transAxes, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="#44506a", ec="white"))
        for v, y in tpos.items():
            ax.text(tx, y, v, transform=ax.transAxes, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="#4e5f80", ec="white"))

        for fv in features:
            for tv in targets:
                c = int(cnt.loc[fv, tv])
                if c <= 0:
                    continue
                lw = 1.0 + 10.0 * (c / max_flow)
                ax.annotate(
                    "",
                    xy=(tx - 0.05, tpos[tv]),
                    xytext=(fx + 0.05, fpos[fv]),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-", color="#8eb5ff", lw=lw, alpha=0.55),
                )
        ax.set_title("Sankey Feature -> Target Flow")
        fig.tight_layout()
        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "sankey_plot generated successfully."

    def calculate_plot(self):
        spec = self.current_spec()
        if not spec:
            self.status = "Select a plot type first."
            return

        for req in spec.required_slots:
            v = self.slot_values.get(req)
            if v is None or v == []:
                self.status = f"Missing required input: {req}"
                return

        used_tables = self.gather_used_tables(spec)
        merged_df, err = self.build_joined_dataframe(used_tables)
        if merged_df is None:
            self.status = err
            return

        try:
            if spec.custom:
                if spec.name == "rf_regression":
                    self._calculate_rf_regression(merged_df)
                    return
                if spec.name == "dendrogram":
                    self._calculate_dendrogram(merged_df)
                    return
                if spec.name == "kmeans_cluster":
                    self._calculate_kmeans_cluster(merged_df)
                    return
                if spec.name == "gaussian_mixture":
                    self._calculate_gaussian_mixture(merged_df)
                    return
                if spec.name in {"heatmap", "clustermap", "quiver_plot"}:
                    self._calculate_grid_plot(merged_df, spec.name)
                    return
                if spec.name == "parallel_lines":
                    self._calculate_parallel_lines(merged_df)
                    return
                if spec.name == "pca_plot":
                    self._calculate_pca_plot(merged_df)
                    return
                if spec.name == "tsne_plot":
                    self._calculate_tsne_plot(merged_df)
                    return
                if spec.name == "markov_chain":
                    self._calculate_markov_chain(merged_df)
                    return
                if spec.name == "sankey_plot":
                    self._calculate_sankey_plot(merged_df)
                    return

            kwargs = self._collect_seaborn_kwargs(spec, merged_df)
            if kwargs is None:
                return

            plt.close("all")
            sns.set_style("whitegrid")
            fn = getattr(sns, spec.function_name or spec.name)

            if self._is_figure_level(spec):
                grid = fn(data=merged_df, **kwargs)
                fig = grid.fig if hasattr(grid, "fig") else plt.gcf()
                fig.set_size_inches(8, 3.2)
                fig.suptitle(spec.name)
                fig.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
                fn(data=merged_df, ax=ax, **kwargs)
                ax.set_title(spec.name)
                fig.tight_layout()

            self.chart_surface = self._render_figure_to_surface(fig)
            plt.close(fig)
            self.status = "Plot generated successfully."
        except Exception as e:
            self.status = f"Plot failed: {e}"

    def clear_selection(self, clear_data: bool = False):
        self.selected_plot = None
        self.slot_values = {}
        self.option_values = {}
        self.active_slot = None
        self.close_menu()
        self.close_input_dialog()
        self.chart_surface = None
        if clear_data:
            self.dataframes = {}
            self.table_sources = {}
            self.table_collapsed = {}
            self.relationships = []
            self.source_files = []
            self.column_types = {}
            self.ordinal_orders = {}
        self.status = "Cleared current selection."

    # ------------------------------
    # Tooltip
    # ------------------------------
    def update_tooltip(self, pos: Tuple[int, int]):
        self.tooltip_text = ""

        for rect, text, _, action in self.menu_buttons:
            if rect.collidepoint(pos):
                tips = {
                    "load_files": "Load one or more data files.",
                    "save_plot": "Save current plot preview as PNG.",
                    "relationships": "Open relationship editor for table joins.",
                    "save_state": "Save app state to JSON.",
                    "load_state": "Load app state from JSON.",
                    "fullscreen": "Toggle fullscreen/windowed mode.",
                    "calculate": "Generate plot using current inputs.",
                    "clear": "Clear selected plot and inputs.",
                }
                self.tooltip_text = tips.get(action, text)
                return

        for rect, kind, table, col in self.data_layout:
            if rect.collidepoint(pos):
                self.tooltip_text = f"Table '{table}'" if kind == "header" else f"Column '{col}' in '{table}'"
                return

        for rect, kind, val in self.plot_layout:
            if rect.collidepoint(pos):
                if kind == "group":
                    self.tooltip_text = f"Toggle {val} section"
                else:
                    self.tooltip_text = PLOT_DESCRIPTIONS.get(val, val)
                return

        for k, rect in self.slot_layout.items():
            if rect.collidepoint(pos):
                self.tooltip_text = self.parameter_description(k)
                return

        for k, rect in self.option_layout.items():
            if rect.collidepoint(pos):
                self.tooltip_text = self.parameter_description(k)
                return

    def draw_tooltip(self, mouse_pos: Tuple[int, int]):
        if not self.tooltip_text:
            return
        pad = 7
        txt = self.font_sm.render(self.tooltip_text, True, TEXT)
        box = pygame.Rect(mouse_pos[0] + 14, mouse_pos[1] + 14, txt.get_width() + pad * 2, txt.get_height() + pad * 2)
        if box.right > self.window_w - 6:
            box.x = self.window_w - box.w - 6
        if box.bottom > self.window_h - 6:
            box.y = self.window_h - box.h - 6
        pygame.draw.rect(self.screen, TOOLTIP_BG, box, border_radius=6)
        pygame.draw.rect(self.screen, (120, 130, 150), box, width=1, border_radius=6)
        self.screen.blit(txt, (box.x + pad, box.y + pad))

    # ------------------------------
    # Event handling
    # ------------------------------
    def handle_menu_action(self, action: str):
        if action == "load_files":
            self.load_files(self.open_files_picker(), append=True)
        elif action == "save_plot":
            self.save_plot()
        elif action == "relationships":
            self.rel_editor_open = True
        elif action == "save_state":
            self.save_state()
        elif action == "load_state":
            self.load_state()
        elif action == "fullscreen":
            self.toggle_fullscreen()
        elif action == "calculate":
            self.calculate_plot()
        elif action == "clear":
            self.clear_selection(clear_data=False)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((DEFAULT_WINDOW_W, DEFAULT_WINDOW_H), pygame.RESIZABLE)
        self.update_layout()
        self.status = "Fullscreen enabled" if self.fullscreen else "Fullscreen disabled"

    def handle_relationship_mouse_down(self, pos: Tuple[int, int], button: int):
        if button == 1:
            if self.rel_clear_btn.collidepoint(pos):
                self.relationships = []
                self.status = "Cleared all relationships."
                return

            # Click line to cycle join mode.
            for a, b, idx in self.rel_line_layout:
                if self.point_to_segment_distance(pos, a, b) <= 6:
                    self.cycle_join_mode(idx)
                    return

            # Begin relation drag from a column box.
            for rect, table, col in self.rel_column_boxes:
                if rect.collidepoint(pos):
                    self.rel_drag_start = (table, col)
                    return

            # Click outside modal closes it.
            if not self.rel_modal_rect.collidepoint(pos):
                self.rel_editor_open = False
                self.rel_drag_start = None

        if button in (4, 5) and self.rel_canvas_rect.collidepoint(pos):
            dy = 1 if button == 4 else -1
            self.rel_scroll_y -= dy * 30
            self.rel_scroll_y = max(0, min(self.rel_scroll_y, self.rel_scroll_max))

    def handle_relationship_mouse_up(self, pos: Tuple[int, int], button: int):
        if button != 1 or not self.rel_drag_start:
            return

        src_t, src_c = self.rel_drag_start
        for rect, dst_t, dst_c in self.rel_column_boxes:
            if rect.collidepoint(pos) and (dst_t != src_t or dst_c != src_c):
                if not self.relation_exists(src_t, src_c, dst_t, dst_c):
                    self.relationships.append(Relation(src_t, src_c, dst_t, dst_c, "inner"))
                    self.status = f"Created relationship: {src_t}.{src_c} <-> {dst_t}.{dst_c}"
                else:
                    self.status = "Relationship already exists."
                break

        self.rel_drag_start = None

    def handle_mouse_down(self, pos: Tuple[int, int], button: int):
        if self.rel_editor_open:
            self.handle_relationship_mouse_down(pos, button)
            return

        if button == 3:
            # Right-click opens per-column type menu.
            for (table, col), ui in self.data_col_ui.items():
                if ui["box"].collidepoint(pos):
                    self.open_col_type_menu(table, col, pos)
                    return
            self.close_col_type_menu()
            return

        if button == 1:
            if self.col_type_menu_target and self.col_type_menu_rect:
                for r, action in self.col_type_menu_items:
                    if r.collidepoint(pos):
                        table, col = self.col_type_menu_target
                        if action.startswith("type::"):
                            self.set_column_type(table, col, action.split("::", 1)[1])
                            self.status = f"Set {table}.{col} type to {self.get_column_type(table, col)}"
                            self.close_col_type_menu()
                            return
                        if action == "set_order":
                            key = self.col_key(table, col)
                            current = self.ordinal_orders.get(key) or self.default_ordinal_order(table, col)
                            self.input_target = (f"ordinal::{table}::{col}", False)
                            self.input_rect = pygame.Rect(self.builder_rect.centerx - 260, self.builder_rect.centery - 70, 520, 130)
                            self.input_value = ",".join(current)
                            self.close_col_type_menu()
                            return
                if not self.col_type_menu_rect.collidepoint(pos):
                    self.close_col_type_menu()
                return

            if self.left_splitter.collidepoint(pos):
                self.resizing_panel = "left"
                return
            if self.right_splitter.collidepoint(pos):
                self.resizing_panel = "right"
                return
            if self.middle_splitter.collidepoint(pos):
                self.resizing_panel = "middle"
                return

            if self.menu_target:
                for r, val in self.menu_items:
                    if r.collidepoint(pos):
                        name, is_option = self.menu_target
                        if is_option:
                            self.option_values[name] = val
                        else:
                            self.slot_values[name] = val
                        self.status = f"Set {name} = {val}"
                        self.close_menu()
                        return
                if self.menu_rect and not self.menu_rect.collidepoint(pos):
                    self.close_menu()
                return

            if self.input_target:
                if self.input_rect and not self.input_rect.collidepoint(pos):
                    self.close_input_dialog()
                return

            spec = self.current_spec()

            for section, r in self.input_header_layout.items():
                if r.collidepoint(pos):
                    self.input_sections[section] = not self.input_sections[section]
                    return

            # Slot rows select active input target.
            if spec:
                for slot, rect in self.slot_layout.items():
                    if self.builder_panel.rect.collidepoint(pos) and rect.collidepoint(pos):
                        if self.active_slot != slot:
                            self.last_multi_anchor = None
                        self.active_slot = slot
                        self.status = f"Active input: {slot}"
                        return

                # Optional param rows open fixed-choice menu or typed dialog.
                for opt, rect in self.option_layout.items():
                    if self.builder_panel.rect.collidepoint(pos) and rect.collidepoint(pos):
                        default = self.option_default(spec.name, opt)
                        fixed = self.fixed_options_for(spec.name, opt, default)
                        if fixed:
                            self.open_fixed_option_menu(opt, True, fixed)
                        else:
                            self.open_input_dialog(opt, True, self.option_values.get(opt))
                        return

            # Data interactions.
            for rect, kind, table, col in self.data_layout:
                if self.data_panel.rect.collidepoint(pos) and rect.collidepoint(pos):
                    if kind == "header":
                        self.table_collapsed[table] = not self.table_collapsed.get(table, False)
                        return
                    if kind == "column":
                        ui = self.data_col_ui.get((table, col))
                        if ui and (ui["corner"].collidepoint(pos) or ui["prefix"].collidepoint(pos)):
                            self.cycle_column_type(table, col)
                            self.status = f"Set {table}.{col} type to {self.get_column_type(table, col)}"
                            return
                        if spec and self.active_slot:
                            mods = pygame.key.get_mods()
                            ctrl = bool(mods & pygame.KMOD_CTRL)
                            shift = bool(mods & pygame.KMOD_SHIFT)
                            if self.active_slot in spec.multi_slots:
                                self.handle_multi_slot_column_click(self.active_slot, table, col, ctrl=ctrl, shift=shift)
                            else:
                                self.assign_columns_to_slot(self.active_slot, table, [col])
                        else:
                            self.drag_item = DragItem(kind="column", value=col, table=table, values=[col])
                            self.drag_pos = pos
                        return

            # Plot panel interactions.
            for rect, kind, value in self.plot_layout:
                if self.plot_panel.rect.collidepoint(pos) and rect.collidepoint(pos):
                    if kind == "group":
                        self.plot_groups[value]["collapsed"] = not self.plot_groups[value]["collapsed"]
                    elif kind == "plot":
                        self.select_plot(value)
                    return

        if button in (4, 5):
            dy = 1 if button == 4 else -1
            if self.data_rect.collidepoint(pos):
                self.data_panel.wheel(dy)
            elif self.plot_type_rect.collidepoint(pos):
                self.plot_panel.wheel(dy)
            elif self.builder_rect.collidepoint(pos):
                self.builder_panel.wheel(dy)

    def handle_mouse_up(self, pos: Tuple[int, int], button: int):
        if self.rel_editor_open:
            self.handle_relationship_mouse_up(pos, button)
            return

        if button == 1 and self.resizing_panel:
            self.resizing_panel = None
            return

        if button == 1:
            for rect, _, _, action in self.menu_buttons:
                if rect.collidepoint(pos):
                    self.handle_menu_action(action)
                    return

        if button != 1 or not self.drag_item:
            return

        spec = self.current_spec()
        if spec:
            for slot, rect in self.slot_layout.items():
                if rect.collidepoint(pos):
                    self.active_slot = slot
                    self.assign_columns_to_slot(slot, self.drag_item.table or "", self.drag_item.values)
                    break

        self.drag_item = None

    def handle_mouse_motion(self, pos: Tuple[int, int], buttons: Tuple[int, int, int]):
        self.drag_pos = pos
        self.update_tooltip(pos)

        if self.rel_editor_open:
            return

        if self.resizing_panel and buttons[0]:
            if self.resizing_panel == "left":
                self.left_panel_w = max(260, min(pos[0] - 12, self.window_w - self.right_panel_w - 380))
            elif self.resizing_panel == "right":
                self.right_panel_w = max(240, min(self.window_w - pos[0] - 12, self.window_w - self.left_panel_w - 380))
            elif self.resizing_panel == "middle":
                self.builder_h = max(220, min(pos[1] - self.builder_rect.y - 4, self.window_h - 230))
            self.update_layout()
            return

        if self.left_splitter.collidepoint(pos) or self.right_splitter.collidepoint(pos):
            self.set_cursor_style("hresize")
        elif self.middle_splitter.collidepoint(pos):
            self.set_cursor_style("vresize")
        elif any(r.collidepoint(pos) for r, _, _, _ in self.menu_buttons):
            self.set_cursor_style("hand")
        else:
            self.set_cursor_style("arrow")

        if not self.drag_item or self.drag_item.kind != "column" or not buttons[0]:
            return

        table = self.drag_item.table
        if not table:
            return

        for rect, kind, t, col in self.data_layout:
            if kind == "column" and t == table and rect.collidepoint(pos):
                if col not in self.drag_item.values:
                    self.drag_item.values.append(col)
                break

    def handle_key_down(self, event: pygame.event.Event):
        if self.rel_editor_open and event.key == pygame.K_ESCAPE:
            self.rel_editor_open = False
            self.rel_drag_start = None
            return

        if event.key == pygame.K_F11:
            self.toggle_fullscreen()
            return

        if not self.input_target and event.key == pygame.K_RETURN:
            self.calculate_plot()
            return

        if not self.input_target:
            return

        if event.key == pygame.K_ESCAPE:
            self.close_input_dialog()
            return
        if event.key == pygame.K_RETURN:
            name, is_option = self.input_target
            if name.startswith("ordinal::"):
                _, table, col = name.split("::", 2)
                vals = [v.strip() for v in self.input_value.split(",") if v.strip()]
                self.set_column_type(table, col, "ordinal")
                self.ordinal_orders[self.col_key(table, col)] = vals
                self.status = f"Set ordinal order for {table}.{col}"
            else:
                parsed = self.parse_input_value(self.input_value)
                if is_option:
                    self.option_values[name] = parsed
                else:
                    self.slot_values[name] = parsed
                self.status = f"Set {name} = {parsed}"
            self.close_input_dialog()
            return
        if event.key == pygame.K_BACKSPACE:
            self.input_value = self.input_value[:-1]
            return
        if event.unicode and event.unicode.isprintable():
            self.input_value += event.unicode

    # ------------------------------
    # Main loop
    # ------------------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode((max(1200, event.w), max(760, event.h)), pygame.RESIZABLE)
                    self.update_layout()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos, event.buttons)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key_down(event)

            self.screen.fill(BG)
            self.draw_menu_bar()
            self.draw_data_panel()
            self.draw_plot_type_panel()
            self.draw_builder_area()
            self.draw_chart_area()
            self.draw_splitters()
            self.draw_status()
            self.draw_dragging()
            self.draw_menu_popup()
            self.draw_input_dialog()
            self.draw_col_type_menu()
            self.draw_relationship_editor()
            self.draw_tooltip(pygame.mouse.get_pos())

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    App().run()
