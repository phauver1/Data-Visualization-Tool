import ast
import inspect
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover - tkinter may be unavailable in some environments
    tk = None
    filedialog = None


# ------------------------------
# Global UI configuration
# ------------------------------
WINDOW_W = 1600
WINDOW_H = 920
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


# ------------------------------
# Seaborn plot registry
# ------------------------------
# This list targets the public plotting API (axes-level + figure-level) that ships with seaborn.
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


# Column-like arguments that should be filled from dataframe columns in the GUI.
COLUMN_PARAM_CANDIDATES = {
    "x", "y", "hue", "size", "style", "units", "weights", "row", "col",
    "x_vars", "y_vars", "vars",
}


# Parameters where selecting from a fixed option list is more appropriate than free text.
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
        "deep", "muted", "bright", "pastel", "dark", "colorblind",
        "viridis", "magma", "rocket", "mako", "crest", "flare",
    ],
    "estimator": ["mean", "median", "sum", "min", "max"],
    "errorbar": ["ci", "pi", "se", "sd", None],
    "kind": ["auto"],
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
    "relplot": "Figure-level relational plot (scatter/line) with faceting.",
    "displot": "Figure-level distribution plot with faceting.",
    "catplot": "Figure-level categorical plot with faceting.",
    "lmplot": "Figure-level regression plot with faceting.",
    "rf_regression": "Random Forest regression with stratified 5-fold CV and feature importances.",
}


PARAM_DESCRIPTIONS = {
    "x": "Column mapped to the x axis.",
    "y": "Column mapped to the y axis.",
    "hue": "Column mapped to color grouping.",
    "size": "Column mapped to marker/element size.",
    "style": "Column mapped to marker or line style grouping.",
    "weights": "Column used as weights during estimation.",
    "row": "Facet row grouping column.",
    "col": "Facet column grouping column.",
    "vars": "Multiple columns used together for matrix-style plots.",
    "x_vars": "Columns for x-axis variables in pair grids.",
    "y_vars": "Columns for y-axis variables in pair grids.",
    "kind": "Plot subtype for figure-level wrappers.",
    "palette": "Color palette name or mapping.",
    "orient": "Orientation of categorical plotting geometry.",
    "fill": "Whether filled geometry should be used.",
    "feature_columns": "Input feature columns for regression model.",
    "target_column": "Single target/output column for regression model.",
}


# ------------------------------
# Data model helpers
# ------------------------------
@dataclass
class PlotSpec:
    name: str
    description: str
    required_slots: List[str]
    column_slots: List[str]
    multi_slots: Set[str]
    option_params: List[str]
    function_name: Optional[str] = None
    custom: bool = False


@dataclass
class DragItem:
    kind: str  # currently "column"
    value: str
    table: Optional[str] = None
    values: List[str] = field(default_factory=list)


class ScrollPanel:
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
# Plot specification building
# ------------------------------
def _is_bool_default(param: inspect.Parameter) -> bool:
    return isinstance(param.default, bool)


def _build_seaborn_spec(plot_name: str) -> PlotSpec:
    fn = getattr(sns, plot_name)
    sig = inspect.signature(fn)

    required_slots: List[str] = []
    column_slots: List[str] = []
    option_params: List[str] = []

    for pname, param in sig.parameters.items():
        # Skip generic/internal parameters from the GUI.
        if pname in {"self", "data", "ax", "kwargs"}:
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue

        # Decide whether this parameter is a dataframe-column assignment slot.
        if pname in COLUMN_PARAM_CANDIDATES:
            column_slots.append(pname)
            if param.default is inspect._empty:
                required_slots.append(pname)
            continue

        # Everything else becomes an optional parameter field in the settings panel.
        option_params.append(pname)

    # Special handling where seaborn signatures do not mark practical requirements clearly.
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
    )


def build_plot_specs() -> Dict[str, PlotSpec]:
    specs: Dict[str, PlotSpec] = {}

    for plot_name in SEABORN_PLOT_FUNCS:
        if hasattr(sns, plot_name):
            specs[plot_name] = _build_seaborn_spec(plot_name)

    # Custom Random Forest Regression "plot" implemented through scikit-learn.
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
    )

    return specs


# ------------------------------
# Main app
# ------------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Seaborn Visual Builder")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 18)
        self.font_sm = pygame.font.SysFont("arial", 15)
        self.font_lg = pygame.font.SysFont("arial", 22, bold=True)

        # Precompute plot specs so GUI can be driven generically by signatures.
        self.plot_specs = build_plot_specs()

        # Screen layout.
        self.data_rect = pygame.Rect(15, 70, 430, WINDOW_H - 85)
        self.plot_type_rect = pygame.Rect(WINDOW_W - 345, 70, 330, WINDOW_H - 85)
        self.builder_rect = pygame.Rect(460, 70, WINDOW_W - 820, 500)
        self.options_rect = pygame.Rect(460, 585, WINDOW_W - 820, WINDOW_H - 600)
        self.chart_rect = pygame.Rect(460, 320, WINDOW_W - 820, 250)

        self.load_btn = pygame.Rect(15, 15, 130, 40)
        self.calc_btn = pygame.Rect(460, 15, 130, 40)
        self.clear_btn = pygame.Rect(600, 15, 130, 40)

        # Scroll panels for left and right columns.
        self.data_panel = ScrollPanel(self.data_rect.inflate(-12, -12))
        self.plot_panel = ScrollPanel(self.plot_type_rect.inflate(-12, -12))
        self.options_panel = ScrollPanel(self.options_rect.inflate(-12, -42))

        # Loaded data and layout caches.
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.table_collapsed: Dict[str, bool] = {}
        self.data_layout: List[Tuple[pygame.Rect, str, str, str]] = []
        self.plot_layout: List[Tuple[pygame.Rect, str]] = []

        # Selection state.
        self.selected_plot: Optional[str] = None
        self.selected_table: Optional[str] = None
        self.slot_values: Dict[str, Union[None, str, List[str]]] = {}
        self.option_values: Dict[str, Any] = {}

        # Interaction state.
        self.slot_layout: Dict[str, pygame.Rect] = {}
        self.option_layout: Dict[str, pygame.Rect] = {}
        self.active_slot: Optional[str] = None

        self.drag_item: Optional[DragItem] = None
        self.drag_pos = (0, 0)

        # Popup/menu states for fixed-choice options and typed-value options.
        self.menu_target: Optional[Tuple[str, bool]] = None  # (name, is_option)
        self.menu_rect: Optional[pygame.Rect] = None
        self.menu_items: List[Tuple[pygame.Rect, Any]] = []

        self.input_target: Optional[Tuple[str, bool]] = None  # (name, is_option)
        self.input_rect: Optional[pygame.Rect] = None
        self.input_value: str = ""

        # Tooltip state.
        self.tooltip_text: str = ""

        self.chart_surface: Optional[pygame.Surface] = None
        self.status = "Load a CSV, Excel, or SQLite file to begin."

    # ------------------------------
    # File loading
    # ------------------------------
    def open_file_picker(self) -> Optional[str]:
        if not filedialog or not tk:
            self.status = "tkinter is unavailable; cannot open file dialog in this environment."
            return None

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.db *.sqlite *.sqlite3"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("SQLite", "*.db *.sqlite *.sqlite3"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or None

    def load_file(self, path: str):
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
                        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
                        new_data[str(table)] = df
                finally:
                    conn.close()
            else:
                self.status = f"Unsupported file extension: {ext}"
                return
        except Exception as e:
            self.status = f"Failed to load file: {e}"
            return

        if not new_data:
            self.status = "No tables/sheets found in selected file."
            return

        self.dataframes = new_data
        self.table_collapsed = {k: False for k in new_data.keys()}
        self.selected_table = None
        self.selected_plot = None
        self.slot_values = {}
        self.option_values = {}
        self.active_slot = None
        self.close_menu()
        self.close_input_dialog()
        self.chart_surface = None

        self.status = f"Loaded {len(new_data)} table(s) from {os.path.basename(path)}"

    # ------------------------------
    # Utility helpers
    # ------------------------------
    def current_spec(self) -> Optional[PlotSpec]:
        if not self.selected_plot:
            return None
        return self.plot_specs.get(self.selected_plot)

    def clip_text(self, text: str, max_w: int, font: pygame.font.Font) -> str:
        if font.size(text)[0] <= max_w:
            return text
        ellipsis = "..."
        for i in range(len(text), 0, -1):
            t = text[:i] + ellipsis
            if font.size(t)[0] <= max_w:
                return t
        return ellipsis

    def draw_text(self, text: str, pos: Tuple[int, int], color=TEXT, font=None):
        if font is None:
            font = self.font
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)

    def draw_button(self, rect: pygame.Rect, label: str, color: Tuple[int, int, int]):
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=2, border_radius=8)
        txt = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def clear_plot_state(self):
        spec = self.current_spec()
        self.slot_values = {}
        self.option_values = {}
        self.active_slot = None
        if not spec:
            return
        for s in spec.column_slots:
            self.slot_values[s] = [] if s in spec.multi_slots else None
        for op in spec.option_params:
            self.option_values[op] = None

    def select_plot(self, plot_name: str):
        self.selected_plot = plot_name
        self.chart_surface = None
        self.close_menu()
        self.close_input_dialog()
        self.clear_plot_state()
        self.status = f"Selected plot: {plot_name}"

    def close_menu(self):
        self.menu_target = None
        self.menu_rect = None
        self.menu_items = []

    def close_input_dialog(self):
        self.input_target = None
        self.input_rect = None
        self.input_value = ""

    def fixed_options_for(self, plot_name: str, param_name: str, default: Any = None) -> Optional[List[Any]]:
        # Plot-specific options have priority over global options.
        if param_name in PLOT_PARAM_OVERRIDES.get(plot_name, {}):
            return PLOT_PARAM_OVERRIDES[plot_name][param_name]
        if param_name in GLOBAL_FIXED_OPTIONS:
            return GLOBAL_FIXED_OPTIONS[param_name]
        # If default is boolean, treat as fixed True/False.
        if isinstance(default, bool):
            return [True, False]
        return None

    def option_default(self, plot_name: str, opt: str) -> Any:
        if plot_name == "rf_regression":
            sig = inspect.signature(RandomForestRegressor)
            return sig.parameters[opt].default
        fn = getattr(sns, plot_name)
        return inspect.signature(fn).parameters[opt].default

    def parse_input_value(self, raw: str) -> Any:
        # Convert common literal inputs into Python types.
        # Example supported values: 10, 0.25, True, None, [1,2], ("a","b")
        raw = raw.strip()
        if raw == "":
            return None
        try:
            return ast.literal_eval(raw)
        except Exception:
            # Fall back to string when literal parsing fails.
            return raw

    def parameter_description(self, name: str) -> str:
        return PARAM_DESCRIPTIONS.get(name, f"Parameter '{name}' for the selected plot.")

    # ------------------------------
    # Drawing panels
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

    def draw_data_panel(self):
        pygame.draw.rect(self.screen, PANEL, self.data_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.data_rect, width=1, border_radius=10)
        self.draw_text("Data", (self.data_rect.x + 14, self.data_rect.y + 10), font=self.font_lg)

        viewport = self.data_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        self.data_layout = []
        y = viewport.y + 8 - self.data_panel.scroll_y
        row_h = 30
        col_h = 24

        for table, df in self.dataframes.items():
            header_rect = pygame.Rect(viewport.x + 8, y, viewport.w - 16, row_h)
            marker = "+" if self.table_collapsed.get(table, False) else "-"
            pygame.draw.rect(self.screen, (55, 62, 75), header_rect, border_radius=6)
            name = self.clip_text(f"{marker} {table} ({len(df.columns)} cols)", header_rect.w - 12, self.font)
            self.draw_text(name, (header_rect.x + 8, header_rect.y + 6))
            self.data_layout.append((header_rect, "header", table, ""))
            y += row_h + 4

            if not self.table_collapsed.get(table, False):
                for col in df.columns:
                    c_rect = pygame.Rect(viewport.x + 20, y, viewport.w - 30, col_h)
                    pygame.draw.rect(self.screen, (70, 78, 92), c_rect, border_radius=5)
                    c_text = self.clip_text(str(col), c_rect.w - 10, self.font_sm)
                    self.draw_text(c_text, (c_rect.x + 6, c_rect.y + 4), font=self.font_sm)
                    self.data_layout.append((c_rect, "column", table, str(col)))
                    y += col_h + 3

            y += 4

        self.data_panel.content_h = max(viewport.h, y - viewport.y + self.data_panel.scroll_y + 8)
        self.data_panel.clamp_scroll()
        if self.data_panel.content_h > viewport.h:
            self.draw_scrollbar(self.data_panel)

    def draw_plot_type_panel(self):
        pygame.draw.rect(self.screen, PANEL, self.plot_type_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.plot_type_rect, width=1, border_radius=10)
        self.draw_text("Plot Types", (self.plot_type_rect.x + 14, self.plot_type_rect.y + 10), font=self.font_lg)

        viewport = self.plot_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        self.plot_layout = []
        y = viewport.y + 8 - self.plot_panel.scroll_y
        item_h = 34

        for plot_name in self.plot_specs.keys():
            rect = pygame.Rect(viewport.x + 8, y, viewport.w - 16, item_h)
            fill = (61, 91, 133) if plot_name == self.selected_plot else (58, 66, 82)
            pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            self.draw_text(plot_name, (rect.x + 8, rect.y + 7), font=self.font_sm)
            self.plot_layout.append((rect, plot_name))
            y += item_h + 6

        self.plot_panel.content_h = max(viewport.h, y - viewport.y + self.plot_panel.scroll_y + 8)
        self.plot_panel.clamp_scroll()
        if self.plot_panel.content_h > viewport.h:
            self.draw_scrollbar(self.plot_panel)

    def draw_builder_area(self):
        pygame.draw.rect(self.screen, PANEL, self.builder_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.builder_rect, width=1, border_radius=10)
        self.draw_text("Builder", (self.builder_rect.x + 14, self.builder_rect.y + 10), font=self.font_lg)

        self.slot_layout = {}

        top = self.builder_rect.y + 44
        spec = self.current_spec()

        if not spec:
            self.draw_text("Click a plot type on the right to begin.", (self.builder_rect.x + 16, top), MUTED)
            return

        self.draw_text(self.clip_text(spec.description, self.builder_rect.w - 32, self.font_sm),
                       (self.builder_rect.x + 16, top), MUTED, self.font_sm)
        top += 30

        table_msg = f"Table: {self.selected_table}" if self.selected_table else "Table: (unset)"
        self.draw_text(table_msg, (self.builder_rect.x + 16, top), MUTED, self.font_sm)
        top += 28

        # Draw one row per column-slot parameter.
        for slot in spec.column_slots:
            s_rect = pygame.Rect(self.builder_rect.x + 16, top, self.builder_rect.w - 32, 40)
            pygame.draw.rect(self.screen, (66, 72, 88), s_rect, border_radius=8)
            border_col = ACCENT if slot == self.active_slot else (120, 130, 150)
            pygame.draw.rect(self.screen, border_col, s_rect, width=1, border_radius=8)

            val = self.slot_values.get(slot)
            if isinstance(val, list):
                body = ", ".join(val) if val else "[click slot, then click/drag columns]"
            else:
                body = str(val) if val else "[click slot, then click column]"

            req = "*" if slot in spec.required_slots else ""
            text = f"{slot}{req}: {body}"
            self.draw_text(self.clip_text(text, s_rect.w - 16, self.font_sm),
                           (s_rect.x + 8, s_rect.y + 11), font=self.font_sm)
            self.slot_layout[slot] = s_rect
            top += 47

    def draw_options_area(self):
        pygame.draw.rect(self.screen, PANEL, self.options_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.options_rect, width=1, border_radius=10)
        self.draw_text("Optional Inputs", (self.options_rect.x + 14, self.options_rect.y + 10), font=self.font_lg)

        self.option_layout = {}
        spec = self.current_spec()
        if not spec:
            self.draw_text("Select a plot first.", (self.options_rect.x + 16, self.options_rect.y + 44), MUTED)
            return

        viewport = self.options_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        y = viewport.y + 6 - self.options_panel.scroll_y
        x = viewport.x + 6
        w = viewport.w - 12
        h = 32

        for opt in spec.option_params:
            r = pygame.Rect(x, y, w, h)
            if viewport.colliderect(r):
                pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=6)
                pygame.draw.rect(self.screen, (120, 130, 150), r, width=1, border_radius=6)

                val = self.option_values.get(opt)
                shown = "(unset)" if val is None else str(val)
                text = f"{opt}: {shown}"
                self.draw_text(self.clip_text(text, r.w - 12, self.font_sm), (r.x + 8, r.y + 8), font=self.font_sm)
                self.option_layout[opt] = r
            y += h + 6

        self.options_panel.content_h = max(viewport.h, y - viewport.y + self.options_panel.scroll_y + 8)
        self.options_panel.clamp_scroll()
        if self.options_panel.content_h > viewport.h:
            self.draw_scrollbar(self.options_panel)

    def draw_chart_area(self):
        pygame.draw.rect(self.screen, PANEL, self.chart_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.chart_rect, width=1, border_radius=10)
        self.draw_text("Preview", (self.chart_rect.x + 14, self.chart_rect.y + 10), font=self.font_lg)

        inner = self.chart_rect.inflate(-24, -48)
        inner.y += 16
        pygame.draw.rect(self.screen, (16, 18, 24), inner, border_radius=8)

        if self.chart_surface:
            scaled = pygame.transform.smoothscale(self.chart_surface, (inner.w, inner.h))
            self.screen.blit(scaled, inner.topleft)
        else:
            self.draw_text("Press Calculate to generate plot", (inner.x + 12, inner.y + 10), MUTED)

    def draw_status(self):
        status_rect = pygame.Rect(740, 15, WINDOW_W - 755, 40)
        pygame.draw.rect(self.screen, (40, 44, 54), status_rect, border_radius=8)
        text = self.clip_text(self.status, status_rect.w - 14, self.font_sm)
        self.draw_text(text, (status_rect.x + 7, status_rect.y + 12), MUTED, self.font_sm)

    def draw_dragging(self):
        if not self.drag_item:
            return

        cols = self.drag_item.values if self.drag_item.values else [self.drag_item.value]
        label = f"{self.drag_item.table}: {len(cols)} col(s)" if len(cols) > 1 else f"{self.drag_item.table}.{cols[0]}"

        txt = self.font.render(label, True, (255, 255, 255))
        r = txt.get_rect()
        r.topleft = (self.drag_pos[0] + 10, self.drag_pos[1] + 8)
        bg = r.inflate(16, 10)
        pygame.draw.rect(self.screen, (90, 100, 120), bg, border_radius=6)
        self.screen.blit(txt, r)

    def draw_menu_popup(self):
        if not self.menu_target or not self.menu_rect:
            return

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((5, 8, 14, 120))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, PANEL, self.menu_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.menu_rect, width=1, border_radius=10)

        label = self.menu_target[0]
        self.draw_text(f"Select value for {label}", (self.menu_rect.x + 12, self.menu_rect.y + 10), font=self.font_sm)

        self.menu_items = []
        y = self.menu_rect.y + 38
        for val in self._current_menu_values:
            r = pygame.Rect(self.menu_rect.x + 12, y, self.menu_rect.w - 24, 30)
            pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=6)
            self.draw_text(str(val), (r.x + 8, r.y + 7), font=self.font_sm)
            self.menu_items.append((r, val))
            y += 35

    def draw_input_dialog(self):
        if not self.input_target or not self.input_rect:
            return

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((5, 8, 14, 120))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, PANEL, self.input_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.input_rect, width=1, border_radius=10)

        label = self.input_target[0]
        self.draw_text(f"Type value for {label}", (self.input_rect.x + 12, self.input_rect.y + 10), font=self.font_sm)
        self.draw_text("Enter=Save, Esc=Cancel", (self.input_rect.x + 12, self.input_rect.y + 34), MUTED, self.font_sm)

        field = pygame.Rect(self.input_rect.x + 12, self.input_rect.y + 60, self.input_rect.w - 24, 34)
        pygame.draw.rect(self.screen, (18, 21, 27), field, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT, field, width=1, border_radius=6)
        self.draw_text(self.clip_text(self.input_value, field.w - 12, self.font_sm), (field.x + 8, field.y + 8), font=self.font_sm)

    def draw_tooltip(self, mouse_pos: Tuple[int, int]):
        if not self.tooltip_text:
            return

        pad = 8
        txt = self.font_sm.render(self.tooltip_text, True, TEXT)
        r = txt.get_rect()
        box = pygame.Rect(mouse_pos[0] + 14, mouse_pos[1] + 14, r.w + pad * 2, r.h + pad * 2)

        # Keep tooltip on-screen.
        if box.right > WINDOW_W - 8:
            box.x = WINDOW_W - box.w - 8
        if box.bottom > WINDOW_H - 8:
            box.y = WINDOW_H - box.h - 8

        pygame.draw.rect(self.screen, TOOLTIP_BG, box, border_radius=6)
        pygame.draw.rect(self.screen, (120, 130, 150), box, width=1, border_radius=6)
        self.screen.blit(txt, (box.x + pad, box.y + pad))

    # ------------------------------
    # Value assignment helpers
    # ------------------------------
    def assign_columns_to_slot(self, slot: str, table: str, columns: List[str]):
        spec = self.current_spec()
        if not spec:
            return
        if not columns:
            return

        if self.selected_table is None:
            self.selected_table = table
        if table != self.selected_table:
            self.status = f"All selected columns must come from {self.selected_table}."
            return

        if slot in spec.multi_slots:
            existing = self.slot_values.get(slot)
            merged = list(existing) if isinstance(existing, list) else []
            for c in columns:
                if c not in merged:
                    merged.append(c)
            self.slot_values[slot] = merged
            self.status = f"Assigned {len(columns)} column(s) to {slot}."
        else:
            self.slot_values[slot] = columns[0]
            self.status = f"Assigned {columns[0]} to {slot}."

        self.chart_surface = None

    def open_fixed_option_menu(self, name: str, is_option: bool, choices: List[Any]):
        self.close_input_dialog()
        h = min(500, 55 + len(choices) * 35)
        self.menu_rect = pygame.Rect(self.builder_rect.centerx - 190, self.builder_rect.centery - h // 2, 380, h)
        self.menu_target = (name, is_option)
        self._current_menu_values = choices

    def open_input_dialog(self, name: str, is_option: bool, current_value: Any):
        self.close_menu()
        self.input_rect = pygame.Rect(self.builder_rect.centerx - 230, self.builder_rect.centery - 70, 460, 130)
        self.input_target = (name, is_option)
        self.input_value = "" if current_value is None else str(current_value)

    # ------------------------------
    # Plot calculation
    # ------------------------------
    def _collect_seaborn_kwargs(self, spec: PlotSpec, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {}

        for slot in spec.column_slots:
            val = self.slot_values.get(slot)
            if val is None or val == []:
                continue

            if isinstance(val, list):
                missing = [c for c in val if c not in df.columns]
                if missing:
                    self.status = f"Missing column(s): {', '.join(missing)}"
                    return None
                kwargs[slot] = val
            else:
                if val not in df.columns:
                    self.status = f"Column {val} not found in selected table"
                    return None
                kwargs[slot] = val

        # Convert user-entered options into kwargs.
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

    def _render_figure_to_surface(self, fig: Any) -> pygame.Surface:
        canvas = fig.canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        raw = canvas.buffer_rgba()
        surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
        return surf.copy()

    def _calculate_rf_regression(self, df: pd.DataFrame):
        features = self.slot_values.get("feature_columns")
        target = self.slot_values.get("target_column")

        if not isinstance(features, list) or not features:
            self.status = "Random forest requires at least one feature column."
            return
        if not isinstance(target, str) or not target:
            self.status = "Random forest requires one target column."
            return

        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            self.status = f"Missing column(s): {', '.join(missing)}"
            return

        model_kwargs = {k: v for k, v in self.option_values.items() if v is not None}

        model = RandomForestRegressor(**model_kwargs)

        X = df[features]
        y = df[target]

        # Remove rows with nulls in model inputs.
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]
        if len(X) < 10:
            self.status = "Need at least 10 complete rows for RF regression."
            return

        # "Stratified" CV for regression is approximated by quantile binning of y.
        # If bins collapse or become invalid, fallback to standard KFold.
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
        for train_idx, test_idx in folds:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fold_model = RandomForestRegressor(**model_kwargs)
            fold_model.fit(X_train, y_train)
            scores.append(float(fold_model.score(X_test, y_test)))

        # Fit final model on all valid data for feature importance plot.
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        sns.barplot(x=importances.index.tolist(), y=importances.values.tolist(), ax=ax, color="#4da3ff")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(f"RF Feature Importances | Mean 5-fold R²: {np.mean(scores):.4f}")
        fig.tight_layout()

        self.chart_surface = self._render_figure_to_surface(fig)
        plt.close(fig)
        self.status = "Random forest plot generated successfully."

    def calculate_plot(self):
        spec = self.current_spec()
        if not spec:
            self.status = "Select a plot type first."
            return

        if not self.selected_table:
            self.status = "Select at least one column to establish active table."
            return

        df = self.dataframes.get(self.selected_table)
        if df is None:
            self.status = "Selected table no longer exists."
            return

        # Validate required slot assignments.
        for req in spec.required_slots:
            val = self.slot_values.get(req)
            if val is None or val == []:
                self.status = f"Missing required input: {req}"
                return

        try:
            if spec.custom and spec.name == "rf_regression":
                self._calculate_rf_regression(df)
                return

            kwargs = self._collect_seaborn_kwargs(spec, df)
            if kwargs is None:
                return

            plt.close("all")
            sns.set_style("whitegrid")

            fn = getattr(sns, spec.function_name or spec.name)

            if self._is_figure_level(spec):
                grid = fn(data=df, **kwargs)
                fig = grid.fig if hasattr(grid, "fig") else plt.gcf()
                fig.set_size_inches(8, 3.2)
                fig.suptitle(f"{spec.name} ({self.selected_table})")
                fig.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
                fn(data=df, ax=ax, **kwargs)
                ax.set_title(f"{spec.name} ({self.selected_table})")
                fig.tight_layout()

            self.chart_surface = self._render_figure_to_surface(fig)
            plt.close(fig)
            self.status = "Plot generated successfully."
        except Exception as e:
            self.status = f"Plot failed: {e}"

    def clear_selection(self):
        self.selected_plot = None
        self.slot_values = {}
        self.option_values = {}
        self.selected_table = None
        self.active_slot = None
        self.close_menu()
        self.close_input_dialog()
        self.chart_surface = None
        self.status = "Cleared current builder state."

    # ------------------------------
    # Tooltip resolver
    # ------------------------------
    def update_tooltip(self, pos: Tuple[int, int]):
        self.tooltip_text = ""

        if self.load_btn.collidepoint(pos):
            self.tooltip_text = "Open CSV/Excel/SQLite file."
            return
        if self.calc_btn.collidepoint(pos):
            self.tooltip_text = "Generate plot preview from current settings."
            return
        if self.clear_btn.collidepoint(pos):
            self.tooltip_text = "Clear plot selection and assignments."
            return

        for rect, kind, table, col in self.data_layout:
            if rect.collidepoint(pos):
                if kind == "header":
                    self.tooltip_text = f"Toggle table '{table}'"
                else:
                    self.tooltip_text = f"Column '{col}' from table '{table}'"
                return

        for rect, plot_name in self.plot_layout:
            if rect.collidepoint(pos):
                self.tooltip_text = PLOT_DESCRIPTIONS.get(plot_name, plot_name)
                return

        spec = self.current_spec()
        if not spec:
            return

        for slot, rect in self.slot_layout.items():
            if rect.collidepoint(pos):
                self.tooltip_text = self.parameter_description(slot)
                return

        for opt, rect in self.option_layout.items():
            if rect.collidepoint(pos):
                self.tooltip_text = self.parameter_description(opt)
                return

    # ------------------------------
    # Event handling
    # ------------------------------
    def handle_mouse_down(self, pos: Tuple[int, int], button: int):
        if button == 1:
            # If a modal popup is active, consume click inside that popup first.
            if self.menu_target:
                for r, val in self.menu_items:
                    if r.collidepoint(pos):
                        name, is_option = self.menu_target
                        if is_option:
                            self.option_values[name] = val
                        else:
                            self.slot_values[name] = val
                        self.chart_surface = None
                        self.status = f"Set {name} = {val}"
                        self.close_menu()
                        return
                if self.menu_rect and not self.menu_rect.collidepoint(pos):
                    self.close_menu()
                return

            if self.input_target:
                # Click outside closes typed dialog.
                if self.input_rect and not self.input_rect.collidepoint(pos):
                    self.close_input_dialog()
                return

            if self.load_btn.collidepoint(pos):
                path = self.open_file_picker()
                if path:
                    self.load_file(path)
                return

            if self.calc_btn.collidepoint(pos):
                self.calculate_plot()
                return

            if self.clear_btn.collidepoint(pos):
                self.clear_selection()
                return

            spec = self.current_spec()

            # Click slot to make it active for direct column assignment.
            if spec:
                for slot, rect in self.slot_layout.items():
                    if rect.collidepoint(pos):
                        self.active_slot = slot
                        self.status = f"Active input: {slot}"
                        return

            # Clicking option rows opens fixed-choice menu or text dialog.
            if spec:
                for opt, rect in self.option_layout.items():
                    if rect.collidepoint(pos):
                        default = self.option_default(spec.name, opt)
                        fixed = self.fixed_options_for(spec.name, opt, default)
                        if fixed:
                            self.open_fixed_option_menu(opt, True, fixed)
                        else:
                            self.open_input_dialog(opt, True, self.option_values.get(opt))
                        return

            # Data panel click behavior (toggle headers / assign columns / begin drag).
            for rect, kind, table, col in self.data_layout:
                if rect.collidepoint(pos):
                    if kind == "header":
                        self.table_collapsed[table] = not self.table_collapsed.get(table, False)
                        return
                    if kind == "column":
                        if spec and self.active_slot:
                            self.assign_columns_to_slot(self.active_slot, table, [col])
                        else:
                            # Allow drag sweep to capture multiple columns quickly.
                            self.drag_item = DragItem(kind="column", value=col, table=table, values=[col])
                            self.drag_pos = pos
                        return

            # Plot type selection is now click-based for reliability.
            for rect, plot_name in self.plot_layout:
                if rect.collidepoint(pos):
                    self.select_plot(plot_name)
                    return

        if button in (4, 5):
            dy = 1 if button == 4 else -1
            if self.data_rect.collidepoint(pos):
                self.data_panel.wheel(dy)
            elif self.plot_type_rect.collidepoint(pos):
                self.plot_panel.wheel(dy)
            elif self.options_rect.collidepoint(pos):
                self.options_panel.wheel(dy)

    def handle_mouse_up(self, pos: Tuple[int, int], button: int):
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

        # When left-dragging a column, add any hovered column from same table to drag bundle.
        if not self.drag_item or self.drag_item.kind != "column" or not buttons[0]:
            return

        table = self.drag_item.table
        if not table:
            return

        for rect, kind, item_table, col in self.data_layout:
            if kind == "column" and item_table == table and rect.collidepoint(pos):
                if col not in self.drag_item.values:
                    self.drag_item.values.append(col)
                return

    def handle_key_down(self, event: pygame.event.Event):
        if not self.input_target:
            return

        if event.key == pygame.K_ESCAPE:
            self.close_input_dialog()
            return

        if event.key == pygame.K_RETURN:
            name, is_option = self.input_target
            parsed = self.parse_input_value(self.input_value)
            if is_option:
                self.option_values[name] = parsed
            else:
                self.slot_values[name] = parsed
            self.chart_surface = None
            self.status = f"Set {name} = {parsed}"
            self.close_input_dialog()
            return

        if event.key == pygame.K_BACKSPACE:
            self.input_value = self.input_value[:-1]
            return

        # Basic text entry for arbitrary values.
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
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos, event.buttons)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key_down(event)

            self.screen.fill(BG)
            self.draw_button(self.load_btn, "Load File", ACCENT)
            self.draw_button(self.calc_btn, "Calculate", GOOD)
            self.draw_button(self.clear_btn, "Clear", WARN)

            self.draw_data_panel()
            self.draw_plot_type_panel()
            self.draw_builder_area()
            self.draw_chart_area()
            self.draw_options_area()
            self.draw_status()
            self.draw_dragging()
            self.draw_menu_popup()
            self.draw_input_dialog()
            self.draw_tooltip(pygame.mouse.get_pos())

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    App().run()
