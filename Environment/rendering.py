"""
Environment/rendering.py
========================
Matplotlib-based 2D top-down renderer for the 20x20 residential grid-world.

This module is decoupled from the Gym environment: it takes a snapshot of
the current state (agent position, goal position) and produces either a
publication-quality PNG (300 DPI) or an RGB numpy array suitable for video
recording / wrappers like gymnasium.wrappers.RecordVideo.

Typical usage from custom_env.py:
    from Environment.rendering import render_environment, render_rgb_array
    render_environment(agent_pos=self._agent_pos,
                       goal_pos=self._goal_pos,
                       save_path="custom_env_render.png")

Nothing in this file imports gymnasium — it is a pure visualization layer.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. Map constants — the single source of truth for the layout
# ---------------------------------------------------------------------------

GRID_SIZE = 20  # 20 x 20 grid (reproducible)

# Cell-type identifiers used in the room-assignment grid
LIVING_ROOM = 1
KITCHEN     = 2
BEDROOM     = 3
BATHROOM    = 4
HALLWAY     = 5

# Floor colors (one per room)
ROOM_COLORS = {
    LIVING_ROOM: "#6FA8DC",   # Blue
    KITCHEN:     "#FFD966",   # Yellow
    BEDROOM:     "#93C47D",   # Green
    BATHROOM:    "#B084CC",   # Purple
    HALLWAY:     "#BFBFBF",   # Gray
}

# Obstacle colors (by category)
OBSTACLE_COLORS = {
    "furniture":  "#7B4B25",   # Brown
    "appliance":  "#E06CA7",   # Violet-pink
    "decoration": "#D62828",   # Red
    "floor_item": "#111111",   # Black
}

DOORWAY_COLOR   = "#F39C12"   # Orange
AGENT_COLOR     = "#1FAA3B"   # Green
GOAL_COLOR      = "#FF1744"   # Bright red (goal flag)
GRID_LINE_COLOR = "#FFFFFF"   # White grid lines over colored cells


# ---------------------------------------------------------------------------
# 2. Static map construction
# ---------------------------------------------------------------------------

def build_room_grid() -> np.ndarray:
    """
    Build the 20x20 room-assignment grid.

    Convention: grid[row, col] with row 0 at the TOP of the rendered figure.
    The hallway forms a central '+' connecting four quadrant rooms.
    """
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    g[0:9,   0:9 ] = LIVING_ROOM   # Top-left
    g[0:9,  11:20] = KITCHEN       # Top-right
    g[11:20, 0:9 ] = BEDROOM       # Bottom-left
    g[11:20,11:20] = BATHROOM      # Bottom-right
    # Central '+' hallway
    g[9:11, 0:20] = HALLWAY
    g[0:20, 9:11] = HALLWAY
    return g


def get_doorways() -> list[tuple[int, int]]:
    """Cells marked as room <-> hallway transitions (orange in the render)."""
    return [
        (4, 8),   # Living room -> hallway (right edge)
        (8, 4),   # Living room -> hallway (bottom edge)
        (4, 11),  # Kitchen -> hallway (left edge)
        (8, 15),  # Kitchen -> hallway (bottom edge)
        (11, 4),  # Bedroom -> hallway (top edge)
        (15, 8),  # Bedroom -> hallway (right edge)
        (11, 15), # Bathroom -> hallway (top edge)
        (15, 11), # Bathroom -> hallway (left edge)
    ]


def get_obstacles() -> list[dict]:
    """
    Hand-placed static obstacles, one dict per obstacle:
        {'pos': (row, col), 'type': <category>, 'label': <human-readable>}.
    Obstacles never overlap doorways and leave the hallway fully navigable.
    """
    return [
        # Living room
        {"pos": (1, 1), "type": "furniture",  "label": "Sofa"},
        {"pos": (1, 2), "type": "furniture",  "label": "Sofa"},
        {"pos": (2, 6), "type": "furniture",  "label": "Armchair"},
        {"pos": (5, 2), "type": "furniture",  "label": "Coffee table"},
        {"pos": (6, 6), "type": "decoration", "label": "Plant"},
        {"pos": (1, 6), "type": "decoration", "label": "TV stand"},

        # Kitchen
        {"pos": (1, 12), "type": "appliance",  "label": "Fridge"},
        {"pos": (1, 13), "type": "appliance",  "label": "Stove"},
        {"pos": (1, 14), "type": "appliance",  "label": "Oven"},
        {"pos": (5, 13), "type": "furniture",  "label": "Dining table"},
        {"pos": (5, 14), "type": "furniture",  "label": "Dining table"},
        {"pos": (6, 18), "type": "appliance",  "label": "Microwave"},
        {"pos": (2, 18), "type": "decoration", "label": "Fruit bowl"},

        # Bedroom
        {"pos": (13, 1), "type": "furniture",  "label": "Bed"},
        {"pos": (13, 2), "type": "furniture",  "label": "Bed"},
        {"pos": (14, 1), "type": "furniture",  "label": "Bed"},
        {"pos": (14, 2), "type": "furniture",  "label": "Bed"},
        {"pos": (13, 6), "type": "furniture",  "label": "Nightstand"},
        {"pos": (18, 2), "type": "furniture",  "label": "Wardrobe"},
        {"pos": (18, 6), "type": "floor_item", "label": "Rug/shoes"},

        # Bathroom
        {"pos": (12, 12), "type": "appliance",  "label": "Toilet"},
        {"pos": (12, 18), "type": "appliance",  "label": "Sink"},
        {"pos": (17, 18), "type": "appliance",  "label": "Bathtub"},
        {"pos": (17, 13), "type": "floor_item", "label": "Laundry basket"},
        {"pos": (14, 17), "type": "decoration", "label": "Towel rack"},
    ]


# The agent's deterministic start cell (hallway centre).
AGENT_START: tuple[int, int] = (10, 10)


# ---------------------------------------------------------------------------
# 3. Publication renderer (matplotlib -> PNG)
# ---------------------------------------------------------------------------

def render_environment(
    agent_pos: Optional[tuple[int, int]] = None,
    goal_pos:  Optional[tuple[int, int]] = None,
    save_path: str = "custom_env_render.png",
    dpi: int = 300,
    show_legend: bool = True,
) -> str:
    """
    Render the full environment to a PNG file.

    Parameters
    ----------
    agent_pos : (row, col), optional
        Agent location. Defaults to AGENT_START.
    goal_pos  : (row, col), optional
        Goal location; drawn as a red star. If None, no goal is drawn.
    save_path : str
        Output PNG path.
    dpi : int
        Figure DPI (300 = print-quality).
    show_legend : bool
        Whether to attach the categorical legend (turn off for thumbnail frames).

    Returns
    -------
    str
        The resolved `save_path` (convenient for chaining / logging).
    """
    if agent_pos is None:
        agent_pos = AGENT_START

    room_grid = build_room_grid()
    doorways  = get_doorways()
    obstacles = get_obstacles()

    # Paint the room colours as an RGB image (one pixel per cell).
    rgb = np.ones((GRID_SIZE, GRID_SIZE, 3))
    for room_id, hex_color in ROOM_COLORS.items():
        rgb[room_grid == room_id] = _hex_to_rgb(hex_color)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(
        rgb,
        extent=(0, GRID_SIZE, 0, GRID_SIZE),
        origin="upper",
        interpolation="nearest",
    )

    # Doorways (orange cells over the floor)
    for (r, c) in doorways:
        ax.add_patch(mpatches.Rectangle(
            (c, GRID_SIZE - 1 - r), 1, 1,
            facecolor=DOORWAY_COLOR, edgecolor="black",
            linewidth=0.8, zorder=2,
        ))

    # Obstacles (category-coloured squares with a black outline)
    for obs in obstacles:
        r, c = obs["pos"]
        ax.add_patch(mpatches.Rectangle(
            (c + 0.1, GRID_SIZE - 1 - r + 0.1), 0.8, 0.8,
            facecolor=OBSTACLE_COLORS[obs["type"]],
            edgecolor="black", linewidth=0.8, zorder=3,
        ))

    # Goal (drawn beneath the agent so the agent stays visible when on-goal)
    if goal_pos is not None:
        gr, gc = goal_pos
        ax.add_patch(mpatches.RegularPolygon(
            (gc + 0.5, GRID_SIZE - 1 - gr + 0.5),
            numVertices=5, radius=0.42, orientation=0,
            facecolor=GOAL_COLOR, edgecolor="black",
            linewidth=1.0, zorder=3.5,
        ))

    # Agent (green circle)
    ar, ac = agent_pos
    ax.add_patch(mpatches.Circle(
        (ac + 0.5, GRID_SIZE - 1 - ar + 0.5), 0.38,
        facecolor=AGENT_COLOR, edgecolor="black",
        linewidth=1.2, zorder=4,
    ))

    # White grid lines across the full map
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color=GRID_LINE_COLOR, linewidth=0.7, zorder=1.5)
        ax.axvline(i, color=GRID_LINE_COLOR, linewidth=0.7, zorder=1.5)

    # Outer frame
    ax.add_patch(mpatches.Rectangle(
        (0, 0), GRID_SIZE, GRID_SIZE,
        fill=False, edgecolor="black", linewidth=1.8, zorder=5,
    ))

    # Axis cosmetics
    ax.set_xlim(-0.2, GRID_SIZE + 0.2)
    ax.set_ylim(-0.2, GRID_SIZE + 0.2)
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 2))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 2))
    ax.set_xticklabels(np.arange(0, GRID_SIZE + 1, 2), fontsize=8)
    ax.set_yticklabels(np.arange(0, GRID_SIZE + 1, 2)[::-1], fontsize=8)
    ax.set_xlabel("Column", fontsize=10)
    ax.set_ylabel("Row",    fontsize=10)
    ax.set_title("Custom 20×20 Residential Grid-World Environment",
                 fontsize=12, pad=12)
    ax.set_aspect("equal")

    # Room labels
    for name, (x, y) in {
        "Living Room": (4.5,  GRID_SIZE - 4.5),
        "Kitchen":     (15.5, GRID_SIZE - 4.5),
        "Bedroom":     (4.5,  4.5),
        "Bathroom":    (15.5, 4.5),
        "Hallway":     (10.0, 10.0),
    }.items():
        ax.text(x, y, name,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white",
                          edgecolor="black",
                          alpha=0.75),
                zorder=4)

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=ROOM_COLORS[LIVING_ROOM], edgecolor="black", label="Living room (floor)"),
            mpatches.Patch(facecolor=ROOM_COLORS[KITCHEN],     edgecolor="black", label="Kitchen (floor)"),
            mpatches.Patch(facecolor=ROOM_COLORS[BEDROOM],     edgecolor="black", label="Bedroom (floor)"),
            mpatches.Patch(facecolor=ROOM_COLORS[BATHROOM],    edgecolor="black", label="Bathroom (floor)"),
            mpatches.Patch(facecolor=ROOM_COLORS[HALLWAY],     edgecolor="black", label="Hallway (floor)"),
            mpatches.Patch(facecolor=DOORWAY_COLOR,            edgecolor="black", label="Doorway"),
            mpatches.Patch(facecolor=OBSTACLE_COLORS["furniture"],  edgecolor="black", label="Furniture"),
            mpatches.Patch(facecolor=OBSTACLE_COLORS["appliance"],  edgecolor="black", label="Appliance"),
            mpatches.Patch(facecolor=OBSTACLE_COLORS["decoration"], edgecolor="black", label="Decoration"),
            mpatches.Patch(facecolor=OBSTACLE_COLORS["floor_item"], edgecolor="black", label="Floor item"),
            mpatches.Patch(facecolor=GOAL_COLOR,               edgecolor="black", label="Goal"),
            mpatches.Circle((0, 0), radius=0.3, facecolor=AGENT_COLOR, edgecolor="black", label="Agent"),
        ]
        ax.legend(
            handles=handles,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=True, fontsize=8.5,
            title="Legend", title_fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 4. Lightweight rgb_array renderer (no matplotlib)
# ---------------------------------------------------------------------------

def render_rgb_array(
    agent_pos: tuple[int, int],
    goal_pos: Optional[tuple[int, int]] = None,
    cell_px: int = 16,
) -> np.ndarray:
    """
    Compact RGB rendering suitable for gymnasium's 'rgb_array' render mode
    and for video-recording wrappers. Returns (H, W, 3) uint8 where
    H = W = GRID_SIZE * cell_px.
    """
    room_palette = {
        LIVING_ROOM: (111, 168, 220),
        KITCHEN:     (255, 217, 102),
        BEDROOM:     (147, 196, 125),
        BATHROOM:    (176, 132, 204),
        HALLWAY:     (191, 191, 191),
    }
    room_grid = build_room_grid()
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for rid, rgb in room_palette.items():
        img[room_grid == rid] = rgb

    # Obstacles
    for obs in get_obstacles():
        r, c = obs["pos"]
        img[r, c] = _hex_to_rgb_u8(OBSTACLE_COLORS[obs["type"]])
    # Doorways
    for (r, c) in get_doorways():
        img[r, c] = _hex_to_rgb_u8(DOORWAY_COLOR)
    # Goal (drawn before agent so agent stays visible when overlapping)
    if goal_pos is not None:
        img[goal_pos[0], goal_pos[1]] = _hex_to_rgb_u8(GOAL_COLOR)
    # Agent
    img[agent_pos[0], agent_pos[1]] = _hex_to_rgb_u8(AGENT_COLOR)

    # Upsample each cell to a visible block
    return np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)


# ---------------------------------------------------------------------------
# 5. Utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _hex_to_rgb_u8(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# 6. Direct-run helper: save the default environment figure
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple self-test: renders the empty environment (no goal) to PNG.
    path = render_environment(save_path="custom_env_render.png", dpi=300)
    print(f"[✓] Saved environment render to: {path}")
