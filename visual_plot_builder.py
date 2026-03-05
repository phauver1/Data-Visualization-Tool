import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pygame
import seaborn as sns

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover - tkinter may be unavailable in some environments
    tk = None
    filedialog = None


# ------------------------------
# Config
# ------------------------------
WINDOW_W = 1400
WINDOW_H = 850
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


PLOT_SPECS = {
    "scatterplot": ["x", "y", "hue", "style", "size"],
    "lineplot": ["x", "y", "hue", "style", "size"],
    "barplot": ["x", "y", "hue"],
    "countplot": ["x", "hue"],
    "histplot": ["x", "hue"],
    "boxplot": ["x", "y", "hue"],
    "violinplot": ["x", "y", "hue"],
    "stripplot": ["x", "y", "hue"],
    "swarmplot": ["x", "y", "hue"],
    "pointplot": ["x", "y", "hue"],
    "kdeplot": ["x", "y", "hue"],
}


@dataclass
class DragItem:
    kind: str  # "plot" or "column"
    value: str
    table: Optional[str] = None


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


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Seaborn Visual Builder")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 18)
        self.font_sm = pygame.font.SysFont("arial", 15)
        self.font_lg = pygame.font.SysFont("arial", 22, bold=True)

        self.data_rect = pygame.Rect(15, 70, 420, WINDOW_H - 85)
        self.plot_type_rect = pygame.Rect(WINDOW_W - 300, 70, 285, WINDOW_H - 85)
        self.common_rect = pygame.Rect(450, 70, WINDOW_W - 765, 390)
        self.chart_rect = pygame.Rect(450, 475, WINDOW_W - 765, 290)

        self.load_btn = pygame.Rect(15, 15, 130, 40)
        self.calc_btn = pygame.Rect(450, 15, 130, 40)
        self.clear_btn = pygame.Rect(590, 15, 130, 40)

        self.data_panel = ScrollPanel(self.data_rect.inflate(-12, -12))
        self.plot_panel = ScrollPanel(self.plot_type_rect.inflate(-12, -12))

        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.table_collapsed: Dict[str, bool] = {}
        self.data_layout: List[Tuple[pygame.Rect, str, str, str]] = []
        self.plot_layout: List[Tuple[pygame.Rect, str]] = []

        self.selected_plot: Optional[str] = None
        self.slot_values: Dict[str, Optional[str]] = {}
        self.selected_table: Optional[str] = None
        self.slot_layout: Dict[str, pygame.Rect] = {}

        self.drag_item: Optional[DragItem] = None
        self.drag_pos = (0, 0)

        self.chart_surface: Optional[pygame.Surface] = None
        self.status = "Load a CSV, Excel, or SQLite file to begin."

    # ------------------------------
    # Data loading
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
        self.chart_surface = None

        table_count = len(new_data)
        self.status = f"Loaded {table_count} table(s) from {os.path.basename(path)}"

    # ------------------------------
    # Rendering helpers
    # ------------------------------
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

    def clip_text(self, text: str, max_w: int, font: pygame.font.Font) -> str:
        if font.size(text)[0] <= max_w:
            return text
        ellipsis = "..."
        for i in range(len(text), 0, -1):
            t = text[:i] + ellipsis
            if font.size(t)[0] <= max_w:
                return t
        return ellipsis

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
        self.draw_text("Seaborn Plots", (self.plot_type_rect.x + 14, self.plot_type_rect.y + 10), font=self.font_lg)

        viewport = self.plot_panel.rect
        pygame.draw.rect(self.screen, PANEL_ALT, viewport, border_radius=6)

        self.plot_layout = []
        y = viewport.y + 8 - self.plot_panel.scroll_y
        item_h = 34

        for plot_name in PLOT_SPECS.keys():
            rect = pygame.Rect(viewport.x + 8, y, viewport.w - 16, item_h)
            fill = (61, 91, 133) if plot_name == self.selected_plot else (58, 66, 82)
            pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            self.draw_text(plot_name, (rect.x + 8, rect.y + 7))
            self.plot_layout.append((rect, plot_name))
            y += item_h + 6

        self.plot_panel.content_h = max(viewport.h, y - viewport.y + self.plot_panel.scroll_y + 8)
        self.plot_panel.clamp_scroll()

        if self.plot_panel.content_h > viewport.h:
            self.draw_scrollbar(self.plot_panel)

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

    def draw_common_area(self):
        pygame.draw.rect(self.screen, PANEL, self.common_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.common_rect, width=1, border_radius=10)
        self.draw_text("Builder", (self.common_rect.x + 14, self.common_rect.y + 10), font=self.font_lg)

        self.slot_layout = {}

        drop_plot = pygame.Rect(self.common_rect.x + 16, self.common_rect.y + 46, self.common_rect.w - 32, 52)
        pygame.draw.rect(self.screen, (50, 80, 115), drop_plot, border_radius=8)
        label = self.selected_plot if self.selected_plot else "Drag plot type here"
        self.draw_text(label, (drop_plot.x + 12, drop_plot.y + 15))

        table_msg = f"Table: {self.selected_table}" if self.selected_table else "Table: (unset)"
        self.draw_text(table_msg, (self.common_rect.x + 18, self.common_rect.y + 108), MUTED, self.font_sm)

        if self.selected_plot:
            args = PLOT_SPECS[self.selected_plot]
            y = self.common_rect.y + 130
            for arg in args:
                s_rect = pygame.Rect(self.common_rect.x + 16, y, self.common_rect.w - 32, 42)
                pygame.draw.rect(self.screen, (66, 72, 88), s_rect, border_radius=8)
                pygame.draw.rect(self.screen, (120, 130, 150), s_rect, width=1, border_radius=8)
                value = self.slot_values.get(arg)
                text = f"{arg.upper()}: {value}" if value else f"{arg.upper()}: [drop column]"
                self.draw_text(self.clip_text(text, s_rect.w - 16, self.font_sm), (s_rect.x + 10, s_rect.y + 11), font=self.font_sm)
                self.slot_layout[arg] = s_rect
                y += 50

        self.plot_drop_rect = drop_plot

    def draw_chart_area(self):
        pygame.draw.rect(self.screen, PANEL, self.chart_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.chart_rect, width=1, border_radius=10)
        self.draw_text("Preview", (self.chart_rect.x + 14, self.chart_rect.y + 10), font=self.font_lg)

        inner = self.chart_rect.inflate(-24, -48)
        inner.y += 18
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
        label = self.drag_item.value if self.drag_item.kind == "plot" else f"{self.drag_item.table}.{self.drag_item.value}"
        txt = self.font.render(label, True, (255, 255, 255))
        r = txt.get_rect()
        r.topleft = (self.drag_pos[0] + 10, self.drag_pos[1] + 8)
        bg = r.inflate(16, 10)
        pygame.draw.rect(self.screen, (90, 100, 120), bg, border_radius=6)
        self.screen.blit(txt, r)

    # ------------------------------
    # Plot generation
    # ------------------------------
    def calculate_plot(self):
        if not self.selected_plot:
            self.status = "Select a plot type first."
            return
        if not self.selected_table:
            self.status = "Drop a column into a slot to set active table."
            return

        df = self.dataframes.get(self.selected_table)
        if df is None:
            self.status = "Selected table no longer exists."
            return

        args = PLOT_SPECS[self.selected_plot]
        required = ["x"] if self.selected_plot in {"countplot", "histplot"} else ["x", "y"]
        for r in required:
            if not self.slot_values.get(r):
                self.status = f"Missing required slot: {r}"
                return

        plot_kwargs = {}
        for a in args:
            col = self.slot_values.get(a)
            if col:
                if col not in df.columns:
                    self.status = f"Column {col} not found in table {self.selected_table}"
                    return
                plot_kwargs[a] = col

        try:
            plt.close("all")
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
            sns.set_style("whitegrid")

            plot_fn = getattr(sns, self.selected_plot)
            plot_fn(data=df, ax=ax, **plot_kwargs)

            ax.set_title(f"{self.selected_plot} ({self.selected_table})")
            fig.tight_layout()

            canvas = fig.canvas
            canvas.draw()
            w, h = canvas.get_width_height()
            raw = canvas.buffer_rgba()
            surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
            self.chart_surface = surf.copy()
            plt.close(fig)
            self.status = "Plot generated successfully."
        except Exception as e:
            self.status = f"Plot failed: {e}"

    def clear_selection(self):
        self.selected_plot = None
        self.slot_values = {}
        self.selected_table = None
        self.chart_surface = None
        self.status = "Cleared current builder state."

    # ------------------------------
    # Event handling
    # ------------------------------
    def handle_mouse_down(self, pos: Tuple[int, int], button: int):
        if button == 1:
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

            for rect, kind, table, col in self.data_layout:
                if rect.collidepoint(pos):
                    if kind == "header":
                        self.table_collapsed[table] = not self.table_collapsed.get(table, False)
                        return
                    if kind == "column":
                        self.drag_item = DragItem(kind="column", value=col, table=table)
                        self.drag_pos = pos
                        return

            for rect, plot_name in self.plot_layout:
                if rect.collidepoint(pos):
                    self.drag_item = DragItem(kind="plot", value=plot_name)
                    self.drag_pos = pos
                    return

        if button in (4, 5):
            dy = 1 if button == 4 else -1
            if self.data_rect.collidepoint(pos):
                self.data_panel.wheel(dy)
            elif self.plot_type_rect.collidepoint(pos):
                self.plot_panel.wheel(dy)

    def handle_mouse_up(self, pos: Tuple[int, int], button: int):
        if button != 1 or not self.drag_item:
            return

        item = self.drag_item

        if item.kind == "plot" and self.plot_drop_rect.collidepoint(pos):
            self.selected_plot = item.value
            self.slot_values = {k: None for k in PLOT_SPECS[self.selected_plot]}
            self.chart_surface = None
            self.status = f"Selected plot type: {self.selected_plot}"

        if item.kind == "column" and self.selected_plot:
            for arg, rect in self.slot_layout.items():
                if rect.collidepoint(pos):
                    if self.selected_table is None:
                        self.selected_table = item.table
                    if self.selected_table != item.table:
                        self.status = (
                            f"All slots must use the same table. Active: {self.selected_table}, dropped: {item.table}"
                        )
                        break
                    self.slot_values[arg] = item.value
                    self.chart_surface = None
                    self.status = f"Assigned {item.value} to {arg}"
                    break

        self.drag_item = None

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
                    self.drag_pos = event.pos

            self.screen.fill(BG)
            self.draw_button(self.load_btn, "Load File", ACCENT)
            self.draw_button(self.calc_btn, "Calculate", GOOD)
            self.draw_button(self.clear_btn, "Clear", WARN)

            self.draw_data_panel()
            self.draw_plot_type_panel()
            self.draw_common_area()
            self.draw_chart_area()
            self.draw_status()
            self.draw_dragging()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    App().run()
