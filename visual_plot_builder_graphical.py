"""Graphical (node-based) visual plot builder.

This LabView-style interface complements `visual_plot_builder.py` by allowing
users to place table and plot blocks in a panning/zooming canvas and connect
them with arrows.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pygame

from visual_plot_builder import build_plot_specs
from visual_plot_builder_common import (
    COLUMN_TYPE_COLORS,
    COLUMN_TYPE_PREFIX,
    GraphConnection,
    GraphicalState,
    PlotNodeState,
    PortRef,
    TableNodeState,
    clamp,
    load_graphical_state,
    save_graphical_state,
    screen_to_world,
    world_to_screen,
)

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None


@dataclass
class TableDef:
    """In-memory table descriptor with source and schema metadata."""

    name: str
    columns: List[str]
    source_kind: str
    file_path: Optional[str] = None
    sqlite_table: Optional[str] = None


@dataclass
class TableNode:
    """Placed table node in the graphical canvas."""

    node_id: str
    table_name: str
    x: float
    y: float
    width: float = 270.0
    height: float = 260.0
    scroll_y: float = 0.0


@dataclass
class PlotNode:
    """Placed graph/plot node in the graphical canvas."""

    node_id: str
    plot_name: str
    x: float
    y: float
    width: float = 380.0
    height: float = 340.0
    scroll_y: float = 0.0
    input_values: Dict[str, Any] = field(default_factory=dict)
    rendered_text: str = "No output yet"


class GraphicalApp:
    """Main application class for the graphical node-based builder."""

    def __init__(self):
        """Initialize Pygame window, palettes, canvas state, and data stores."""

        pygame.init()
        self.screen = pygame.display.set_mode((1700, 980), pygame.RESIZABLE)
        pygame.display.set_caption("Visual Plot Builder - Graphical")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 16)
        self.font_sm = pygame.font.SysFont("arial", 14)
        self.font_lg = pygame.font.SysFont("arial", 20, bold=True)

        self.plot_specs = build_plot_specs()

        self.tables: Dict[str, TableDef] = {}
        self.source_files: List[str] = []

        self.table_nodes: Dict[str, TableNode] = {}
        self.plot_nodes: Dict[str, PlotNode] = {}
        self.connections: List[GraphConnection] = []

        self.column_types: Dict[str, str] = {}

        self.pan_x = 360.0
        self.pan_y = 60.0
        self.zoom = 1.0

        self.drag_palette_item: Optional[Tuple[str, str]] = None
        self.drag_node_id: Optional[str] = None
        self.drag_offset = (0.0, 0.0)
        self.drag_connection_source: Optional[PortRef] = None

        self.active_plot_input_target: Optional[Tuple[str, str]] = None
        self.text_input_buffer: str = ""

        self.status = "Load files, then drag tables/plots into canvas."

    def _make_tk_root(self):
        """Create a hidden Tk root configured for file dialogs."""

        if not tk:
            return None
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        root.lift()
        return root

    def pick_files(self) -> List[str]:
        """Open multi-file picker for data sources."""

        if not filedialog or not tk:
            self.status = "tkinter unavailable"
            return []
        root = self._make_tk_root()
        paths = filedialog.askopenfilenames(
            parent=root,
            title="Select data file(s)",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.db *.sqlite *.sqlite3"),
                ("All files", "*.*"),
            ],
        )
        if root:
            root.destroy()
        return list(paths)

    def pick_save_path(self, title: str, ext: str) -> Optional[str]:
        """Open save dialog and return selected path."""

        if not filedialog or not tk:
            self.status = "tkinter unavailable"
            return None
        root = self._make_tk_root()
        path = filedialog.asksaveasfilename(parent=root, title=title, defaultextension=ext)
        if root:
            root.destroy()
        return path or None

    def load_file(self, path: str):
        """Load schema metadata from CSV/Excel/SQLite without heavy in-memory duplication."""

        ext = os.path.splitext(path)[1].lower()
        added = 0
        if ext == ".csv":
            df = pd.read_csv(path, nrows=5)
            name = os.path.splitext(os.path.basename(path))[0]
            self.tables[name] = TableDef(name=name, columns=[str(c) for c in df.columns], source_kind="memory", file_path=path)
            added += 1
        elif ext in {".xlsx", ".xls"}:
            sheets = pd.read_excel(path, sheet_name=None, nrows=5)
            for s, df in sheets.items():
                name = str(s)
                self.tables[name] = TableDef(name=name, columns=[str(c) for c in df.columns], source_kind="memory", file_path=path)
                added += 1
        elif ext in {".db", ".sqlite", ".sqlite3"}:
            conn = sqlite3.connect(path)
            try:
                names = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                    conn,
                )["name"].tolist()
                for t in names:
                    cols = pd.read_sql_query(f'PRAGMA table_info("{t}")', conn)["name"].astype(str).tolist()
                    self.tables[t] = TableDef(name=t, columns=cols, source_kind="sqlite", file_path=path, sqlite_table=t)
                    added += 1
            finally:
                conn.close()
        if path not in self.source_files:
            self.source_files.append(path)
        self.status = f"Loaded {added} table(s) from {os.path.basename(path)}"

    def load_files(self):
        """Pick and load one or more data files."""

        for p in self.pick_files():
            self.load_file(p)

    def palette_hit_test(self, pos: Tuple[int, int]) -> Optional[Tuple[str, str]]:
        """Identify drag source item from left palette at mouse position."""

        x, y = pos
        if x > 330:
            return None
        top = 70
        if 20 <= x <= 310 and 20 <= y <= 50:
            return ("action", "load")
        y0 = top
        for tname in self.tables.keys():
            r = pygame.Rect(12, y0, 306, 26)
            if r.collidepoint(pos):
                return ("table", tname)
            y0 += 30
        y0 += 18
        for pname in self.plot_specs.keys():
            r = pygame.Rect(12, y0, 306, 26)
            if r.collidepoint(pos):
                return ("plot", pname)
            y0 += 30
        return None

    def draw_palette(self):
        """Draw left-side palette containing tables and plot blocks."""

        pygame.draw.rect(self.screen, (34, 37, 45), pygame.Rect(0, 0, 330, self.screen.get_height()))
        pygame.draw.line(self.screen, (90, 95, 108), (330, 0), (330, self.screen.get_height()), 1)
        self.screen.blit(self.font_lg.render("Graphical Builder", True, (230, 232, 236)), (14, 12))

        load_btn = pygame.Rect(12, 20, 306, 30)
        pygame.draw.rect(self.screen, (75, 170, 255), load_btn, border_radius=7)
        self.screen.blit(self.font_sm.render("Load Files", True, (255, 255, 255)), (20, 28))

        y = 70
        self.screen.blit(self.font.render("Tables", True, (220, 224, 232)), (12, y))
        y += 24
        for tname in self.tables.keys():
            r = pygame.Rect(12, y, 306, 26)
            pygame.draw.rect(self.screen, (58, 66, 82), r, border_radius=5)
            self.screen.blit(self.font_sm.render(tname, True, (235, 237, 240)), (18, y + 5))
            y += 30

        y += 8
        self.screen.blit(self.font.render("Graphs", True, (220, 224, 232)), (12, y))
        y += 24
        for pname in self.plot_specs.keys():
            r = pygame.Rect(12, y, 306, 26)
            pygame.draw.rect(self.screen, (52, 60, 76), r, border_radius=5)
            self.screen.blit(self.font_sm.render(pname, True, (215, 220, 230)), (18, y + 5))
            y += 30

    def draw_canvas(self):
        """Render panning/zooming canvas with nodes and connections."""

        canvas = pygame.Rect(330, 0, self.screen.get_width() - 330, self.screen.get_height())
        pygame.draw.rect(self.screen, (20, 22, 28), canvas)

        # Draw connections first so node boxes sit above lines.
        for conn in self.connections:
            a = self.get_port_screen_pos(conn.source)
            b = self.get_port_screen_pos(conn.target)
            if not a or not b:
                continue
            pygame.draw.line(self.screen, (145, 180, 255), a, b, 2)
            self.draw_arrow_head(a, b, (145, 180, 255))

        if self.drag_connection_source:
            a = self.get_port_screen_pos(self.drag_connection_source)
            if a:
                b = pygame.mouse.get_pos()
                pygame.draw.line(self.screen, (95, 165, 255), a, b, 2)
                self.draw_arrow_head(a, b, (95, 165, 255))

        for node in self.table_nodes.values():
            self.draw_table_node(node)
        for node in self.plot_nodes.values():
            self.draw_plot_node(node)

    def draw_arrow_head(self, a: Tuple[int, int], b: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a simple triangular arrow head at the end of a connection."""

        dx, dy = b[0] - a[0], b[1] - a[1]
        n = (dx * dx + dy * dy) ** 0.5
        if n < 2:
            return
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        tip = b
        left = (int(b[0] - 10 * ux + 5 * px), int(b[1] - 10 * uy + 5 * py))
        right = (int(b[0] - 10 * ux - 5 * px), int(b[1] - 10 * uy - 5 * py))
        pygame.draw.polygon(self.screen, color, [tip, left, right])

    def draw_table_node(self, node: TableNode):
        """Draw a table node box with scrollable column list and type corner markers."""

        sx, sy = world_to_screen(node.x, node.y, self.pan_x, self.pan_y, self.zoom)
        w = node.width * self.zoom
        h = node.height * self.zoom
        rect = pygame.Rect(int(sx), int(sy), int(w), int(h))
        pygame.draw.rect(self.screen, (48, 56, 70), rect, border_radius=8)
        pygame.draw.rect(self.screen, (140, 150, 170), rect, width=1, border_radius=8)

        self.screen.blit(self.font_sm.render(node.table_name, True, (235, 238, 242)), (rect.x + 8, rect.y + 7))
        cols = self.tables.get(node.table_name, TableDef(node.table_name, [], "memory")).columns
        y = rect.y + 30 - int(node.scroll_y)
        for col in cols:
            row = pygame.Rect(rect.x + 8, y, rect.w - 16, 22)
            if rect.colliderect(row):
                pygame.draw.rect(self.screen, (72, 84, 102), row, border_radius=4)
                ctype = self.column_types.get(f"{node.table_name}::{col}", "ratio")
                colr = COLUMN_TYPE_COLORS[ctype]
                pygame.draw.rect(self.screen, colr, pygame.Rect(row.x + 2, row.y + 2, 7, 7), border_radius=2)
                prefix = COLUMN_TYPE_PREFIX[ctype]
                self.screen.blit(self.font_sm.render(prefix + str(col), True, colr), (row.x + 12, row.y + 4))
                pygame.draw.circle(self.screen, (168, 198, 255), (row.right - 8, row.centery), 4)
            y += 25

    def draw_plot_node(self, node: PlotNode):
        """Draw a plot node containing inputs, output region, and action buttons."""

        sx, sy = world_to_screen(node.x, node.y, self.pan_x, self.pan_y, self.zoom)
        w = node.width * self.zoom
        h = node.height * self.zoom
        rect = pygame.Rect(int(sx), int(sy), int(w), int(h))
        pygame.draw.rect(self.screen, (52, 60, 76), rect, border_radius=8)
        pygame.draw.rect(self.screen, (140, 150, 170), rect, width=1, border_radius=8)
        self.screen.blit(self.font_sm.render(node.plot_name, True, (240, 242, 245)), (rect.x + 8, rect.y + 7))

        # Draw per-node buttons.
        calc_btn = pygame.Rect(rect.x + 8, rect.y + 30, 70, 24)
        save_btn = pygame.Rect(rect.x + 84, rect.y + 30, 70, 24)
        pygame.draw.rect(self.screen, (74, 186, 108), calc_btn, border_radius=5)
        pygame.draw.rect(self.screen, (75, 170, 255), save_btn, border_radius=5)
        self.screen.blit(self.font_sm.render("Calculate", True, (255, 255, 255)), (calc_btn.x + 6, calc_btn.y + 4))
        self.screen.blit(self.font_sm.render("Save", True, (255, 255, 255)), (save_btn.x + 20, save_btn.y + 4))

        spec = self.plot_specs[node.plot_name]
        y = rect.y + 62 - int(node.scroll_y)
        for name in spec.column_slots + spec.option_params:
            row = pygame.Rect(rect.x + 8, y, rect.w - 16, 22)
            if rect.colliderect(row):
                pygame.draw.rect(self.screen, (72, 82, 100), row, border_radius=4)
                val = node.input_values.get(name, "")
                txt = f"{name}: {val}" if str(val) else f"{name}:"
                self.screen.blit(self.font_sm.render(txt, True, (225, 230, 236)), (row.x + 6, row.y + 4))
                pygame.draw.circle(self.screen, (168, 198, 255), (row.x + 7, row.centery), 4)
            y += 25

        out = pygame.Rect(rect.x + 8, rect.bottom - 74, rect.w - 16, 66)
        pygame.draw.rect(self.screen, (18, 22, 30), out, border_radius=5)
        pygame.draw.rect(self.screen, (100, 110, 130), out, width=1, border_radius=5)
        self.screen.blit(self.font_sm.render(node.rendered_text[:80], True, (180, 186, 196)), (out.x + 6, out.y + 8))

    def hit_plot_button(self, pos: Tuple[int, int]) -> Optional[Tuple[str, str]]:
        """Return (node_id, action) for a plot-node button hit, if any."""

        for node in self.plot_nodes.values():
            sx, sy = world_to_screen(node.x, node.y, self.pan_x, self.pan_y, self.zoom)
            rect = pygame.Rect(int(sx), int(sy), int(node.width * self.zoom), int(node.height * self.zoom))
            calc_btn = pygame.Rect(rect.x + 8, rect.y + 30, 70, 24)
            save_btn = pygame.Rect(rect.x + 84, rect.y + 30, 70, 24)
            if calc_btn.collidepoint(pos):
                return node.node_id, "calculate"
            if save_btn.collidepoint(pos):
                return node.node_id, "save"
        return None

    def hit_plot_input_row(self, pos: Tuple[int, int]) -> Optional[Tuple[str, str]]:
        """Return (node_id, input_name) if a plot input row was clicked."""

        for node in self.plot_nodes.values():
            sx, sy = world_to_screen(node.x, node.y, self.pan_x, self.pan_y, self.zoom)
            rect = pygame.Rect(int(sx), int(sy), int(node.width * self.zoom), int(node.height * self.zoom))
            spec = self.plot_specs[node.plot_name]
            y = rect.y + 62 - int(node.scroll_y)
            for name in spec.column_slots + spec.option_params:
                row = pygame.Rect(rect.x + 8, y, rect.w - 16, 22)
                if row.collidepoint(pos):
                    return node.node_id, name
                y += 25
        return None

    def get_port_screen_pos(self, port: PortRef) -> Optional[Tuple[int, int]]:
        """Resolve a source/target port reference to canvas screen coordinates."""

        if port.node_id in self.table_nodes:
            node = self.table_nodes[port.node_id]
            cols = self.tables.get(node.table_name, TableDef(node.table_name, [], "memory")).columns
            if port.port_name not in cols:
                return None
            idx = cols.index(port.port_name)
            wx = node.x + node.width - 8 / self.zoom
            wy = node.y + 40 + idx * 25
            sx, sy = world_to_screen(wx, wy, self.pan_x, self.pan_y, self.zoom)
            return int(sx), int(sy)
        if port.node_id in self.plot_nodes:
            node = self.plot_nodes[port.node_id]
            spec = self.plot_specs[node.plot_name]
            inputs = spec.column_slots + spec.option_params
            if port.port_name not in inputs:
                return None
            idx = inputs.index(port.port_name)
            wx = node.x + 7 / self.zoom
            wy = node.y + 70 + idx * 25
            sx, sy = world_to_screen(wx, wy, self.pan_x, self.pan_y, self.zoom)
            return int(sx), int(sy)
        return None

    def hit_test_table_port(self, pos: Tuple[int, int]) -> Optional[PortRef]:
        """Return table-column port under the pointer if one is hit."""

        wx, wy = screen_to_world(pos[0], pos[1], self.pan_x, self.pan_y, self.zoom)
        for node in self.table_nodes.values():
            cols = self.tables.get(node.table_name, TableDef(node.table_name, [], "memory")).columns
            for idx, col in enumerate(cols):
                px = node.x + node.width - 8 / self.zoom
                py = node.y + 40 + idx * 25
                if (wx - px) ** 2 + (wy - py) ** 2 <= (7 / self.zoom) ** 2:
                    return PortRef(node_id=node.node_id, port_name=col)
        return None

    def hit_test_plot_input_port(self, pos: Tuple[int, int]) -> Optional[PortRef]:
        """Return plot-input port under the pointer if one is hit."""

        wx, wy = screen_to_world(pos[0], pos[1], self.pan_x, self.pan_y, self.zoom)
        for node in self.plot_nodes.values():
            spec = self.plot_specs[node.plot_name]
            inputs = spec.column_slots + spec.option_params
            for idx, pname in enumerate(inputs):
                px = node.x + 7 / self.zoom
                py = node.y + 70 + idx * 25
                if (wx - px) ** 2 + (wy - py) ** 2 <= (7 / self.zoom) ** 2:
                    return PortRef(node_id=node.node_id, port_name=pname)
        return None

    def save_state(self):
        """Save graphical node/canvas state as JSON."""

        path = self.pick_save_path("Save graphical state", ".json")
        if not path:
            return
        state = GraphicalState(
            source_files=self.source_files,
            table_nodes=[TableNodeState(**node.__dict__) for node in self.table_nodes.values()],
            plot_nodes=[PlotNodeState(**node.__dict__) for node in self.plot_nodes.values()],
            connections=self.connections,
            pan_x=self.pan_x,
            pan_y=self.pan_y,
            zoom=self.zoom,
        )
        save_graphical_state(path, state)
        self.status = f"Saved graphical state to {os.path.basename(path)}"

    def load_state(self):
        """Load graphical node/canvas state from JSON and rebuild nodes."""

        if not filedialog or not tk:
            self.status = "tkinter unavailable"
            return
        root = self._make_tk_root()
        path = filedialog.askopenfilename(parent=root, title="Load graphical state", filetypes=[("JSON", "*.json")])
        if root:
            root.destroy()
        if not path:
            return

        state = load_graphical_state(path)
        self.tables = {}
        self.source_files = []
        for p in state.source_files:
            self.load_file(p)

        self.table_nodes = {n.node_id: TableNode(**n.__dict__) for n in state.table_nodes}
        self.plot_nodes = {n.node_id: PlotNode(**n.__dict__) for n in state.plot_nodes}
        self.connections = state.connections
        self.pan_x = state.pan_x
        self.pan_y = state.pan_y
        self.zoom = state.zoom
        self.status = f"Loaded graphical state from {os.path.basename(path)}"

    def handle_mouse_down(self, pos: Tuple[int, int], button: int):
        """Process mouse down events for palette dragging, node dragging, and panning."""

        if button == 1:
            item = self.palette_hit_test(pos)
            if item:
                kind, value = item
                if kind == "action" and value == "load":
                    self.load_files()
                    return
                self.drag_palette_item = item
                return

            src_port = self.hit_test_table_port(pos) or self.hit_test_plot_input_port(pos)
            if src_port:
                self.drag_connection_source = src_port
                return

            hit_btn = self.hit_plot_button(pos)
            if hit_btn:
                node_id, action = hit_btn
                node = self.plot_nodes[node_id]
                if action == "calculate":
                    node.rendered_text = f"Calculated {node.plot_name}"
                    self.status = f"Calculated {node.plot_name}"
                else:
                    path = self.pick_save_path(f"Save {node.plot_name} output", ".png")
                    if path:
                        surf = pygame.Surface((220, 120))
                        surf.fill((24, 26, 32))
                        surf.blit(self.font_sm.render(node.rendered_text[:40], True, (230, 232, 236)), (8, 10))
                        pygame.image.save(surf, path)
                        self.status = f"Saved node output to {os.path.basename(path)}"
                return

            hit_input = self.hit_plot_input_row(pos)
            if hit_input:
                node_id, input_name = hit_input
                self.active_plot_input_target = (node_id, input_name)
                self.text_input_buffer = str(self.plot_nodes[node_id].input_values.get(input_name, ""))
                self.status = f"Editing {self.plot_nodes[node_id].plot_name}.{input_name}"
                return

            # Begin dragging a node if clicking inside one.
            wx, wy = screen_to_world(pos[0], pos[1], self.pan_x, self.pan_y, self.zoom)
            for node in list(self.plot_nodes.values()) + list(self.table_nodes.values()):
                if node.x <= wx <= node.x + node.width and node.y <= wy <= node.y + node.height:
                    self.drag_node_id = node.node_id
                    self.drag_offset = (wx - node.x, wy - node.y)
                    return

        if button == 2:
            # Middle mouse panning.
            self.drag_node_id = "__pan__"
            self.drag_offset = pos

        if button == 4:
            self.zoom = clamp(self.zoom * 1.08, 0.35, 3.0)
        if button == 5:
            self.zoom = clamp(self.zoom / 1.08, 0.35, 3.0)

    def handle_mouse_up(self, pos: Tuple[int, int], button: int):
        """Process mouse release events and finalize drag/drop operations."""

        if button == 1 and self.drag_palette_item:
            kind, value = self.drag_palette_item
            wx, wy = screen_to_world(pos[0], pos[1], self.pan_x, self.pan_y, self.zoom)
            if kind == "table":
                node_id = str(uuid.uuid4())
                self.table_nodes[node_id] = TableNode(node_id=node_id, table_name=value, x=wx, y=wy)
            elif kind == "plot":
                node_id = str(uuid.uuid4())
                self.plot_nodes[node_id] = PlotNode(node_id=node_id, plot_name=value, x=wx, y=wy)
            self.drag_palette_item = None
            return

        if button == 1 and self.drag_connection_source:
            tgt = self.hit_test_plot_input_port(pos) or self.hit_test_table_port(pos)
            if tgt and (tgt.node_id != self.drag_connection_source.node_id or tgt.port_name != self.drag_connection_source.port_name):
                self.connections.append(GraphConnection(source=self.drag_connection_source, target=tgt))
                self.status = "Created connection"
            self.drag_connection_source = None
            return

        if button == 1 and self.drag_node_id:
            self.drag_node_id = None

        if button == 2 and self.drag_node_id == "__pan__":
            self.drag_node_id = None

    def handle_mouse_motion(self, pos: Tuple[int, int], rel: Tuple[int, int], buttons: Tuple[int, int, int]):
        """Handle dragging, panning, and interactive cursor updates."""

        if self.drag_node_id == "__pan__" and buttons[1]:
            self.pan_x += rel[0]
            self.pan_y += rel[1]
            return

        if self.drag_node_id and self.drag_node_id != "__pan__" and buttons[0]:
            wx, wy = screen_to_world(pos[0], pos[1], self.pan_x, self.pan_y, self.zoom)
            if self.drag_node_id in self.table_nodes:
                node = self.table_nodes[self.drag_node_id]
                node.x = wx - self.drag_offset[0]
                node.y = wy - self.drag_offset[1]
            elif self.drag_node_id in self.plot_nodes:
                node = self.plot_nodes[self.drag_node_id]
                node.x = wx - self.drag_offset[0]
                node.y = wy - self.drag_offset[1]

    def handle_key_down(self, event: pygame.event.Event):
        """Handle keyboard shortcuts for save/load and app-level operations."""

        if self.active_plot_input_target:
            node_id, input_name = self.active_plot_input_target
            if event.key == pygame.K_RETURN:
                self.plot_nodes[node_id].input_values[input_name] = self.text_input_buffer
                self.status = f"Set {self.plot_nodes[node_id].plot_name}.{input_name}"
                self.active_plot_input_target = None
                self.text_input_buffer = ""
                return
            if event.key == pygame.K_ESCAPE:
                self.active_plot_input_target = None
                self.text_input_buffer = ""
                return
            if event.key == pygame.K_BACKSPACE:
                self.text_input_buffer = self.text_input_buffer[:-1]
                return
            if event.unicode and event.unicode.isprintable():
                self.text_input_buffer += event.unicode
                return

        if event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            self.save_state()
        elif event.key == pygame.K_o and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            self.load_state()

    def run(self):
        """Run main event/render loop."""

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((max(1100, event.w), max(700, event.h)), pygame.RESIZABLE)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos, event.rel, event.buttons)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key_down(event)

            self.screen.fill((24, 26, 32))
            self.draw_palette()
            self.draw_canvas()
            self.screen.blit(self.font_sm.render(self.status, True, (172, 178, 188)), (340, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    GraphicalApp().run()
