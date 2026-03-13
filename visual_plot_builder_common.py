"""Common utilities shared by visual plot builder applications.

This module centralizes data structures and small helper utilities used by
both the standard UI and the graphical/LabView-style UI variants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Shared column measurement/type display settings.
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


@dataclass
class PortRef:
    """Reference to a specific input/output port on a graph node."""

    node_id: str
    port_name: str


@dataclass
class GraphConnection:
    """Connection edge between two ports in the graphical builder canvas."""

    source: PortRef
    target: PortRef


@dataclass
class TableNodeState:
    """Serializable state for a table node in the graphical builder."""

    node_id: str
    table_name: str
    x: float
    y: float
    width: float = 270.0
    height: float = 260.0


@dataclass
class PlotNodeState:
    """Serializable state for a plot node in the graphical builder."""

    node_id: str
    plot_name: str
    x: float
    y: float
    width: float = 380.0
    height: float = 340.0
    input_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphicalState:
    """Top-level serialization model for the graphical builder state."""

    source_files: List[str] = field(default_factory=list)
    table_nodes: List[TableNodeState] = field(default_factory=list)
    plot_nodes: List[PlotNodeState] = field(default_factory=list)
    connections: List[GraphConnection] = field(default_factory=list)
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 1.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value to an inclusive [minimum, maximum] range."""

    return max(minimum, min(value, maximum))


def world_to_screen(x: float, y: float, pan_x: float, pan_y: float, zoom: float) -> Tuple[float, float]:
    """Convert world coordinates into screen coordinates for pan/zoom canvas."""

    return x * zoom + pan_x, y * zoom + pan_y


def screen_to_world(x: float, y: float, pan_x: float, pan_y: float, zoom: float) -> Tuple[float, float]:
    """Convert screen coordinates into world coordinates for pan/zoom canvas."""

    return (x - pan_x) / zoom, (y - pan_y) / zoom


def save_graphical_state(path: str, state: GraphicalState) -> None:
    """Write graphical builder state to disk as JSON."""

    payload = asdict(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_graphical_state(path: str) -> GraphicalState:
    """Load graphical builder state from JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state = GraphicalState()
    state.source_files = payload.get("source_files", [])
    state.table_nodes = [TableNodeState(**item) for item in payload.get("table_nodes", [])]
    state.plot_nodes = [PlotNodeState(**item) for item in payload.get("plot_nodes", [])]

    # Rehydrate nested dataclasses for connections.
    conn_items = []
    for c in payload.get("connections", []):
        src = PortRef(**c["source"])
        tgt = PortRef(**c["target"])
        conn_items.append(GraphConnection(source=src, target=tgt))
    state.connections = conn_items

    state.pan_x = float(payload.get("pan_x", 0.0))
    state.pan_y = float(payload.get("pan_y", 0.0))
    state.zoom = float(payload.get("zoom", 1.0))
    return state
