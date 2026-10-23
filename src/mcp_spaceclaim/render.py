"""Shaded PNG views of the design, so the model can look at what it built.

Tessellation comes from the backend (``Body.tessellate`` -> pyvista meshes); drawing is
plain matplotlib on the Agg backend. No OpenGL context is needed, so this also works on
a headless Windows VM, where VTK off-screen rendering often does not.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

# (elevation, azimuth) in matplotlib's convention. "front" looks along +Y, "right"
# along -X, "top" down -Z: the usual third-angle CAD views for a Z-up model.
VIEWS: dict[str, tuple[float, float]] = {
    "iso": (30.0, -60.0),
    "front": (0.0, -90.0),
    "top": (90.0, -90.0),
    "right": (0.0, 0.0),
    "back": (0.0, 90.0),
    "left": (0.0, 180.0),
    "bottom": (-90.0, -90.0),
}

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#72B7B2", "#E45756"]
HIGHLIGHT = "#E8112D"
MAX_TRIANGLES = 120_000


@dataclass
class Mesh:
    """One body's triangles, already in working units."""

    name: str
    points: np.ndarray  # (n, 3)
    triangles: np.ndarray  # (m, 3) indices into points
    color: str | None = None
    highlight: bool = False


@dataclass
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    lines: list[np.ndarray] = field(default_factory=list)  # polylines to draw in HIGHLIGHT
    unit: str = "mm"


def mesh_from_polydata(name: str, poly, scale: float, color: str | None = None) -> Mesh:
    """Convert a pyvista PolyData (SI units) into a Mesh in working units."""
    tri = poly.triangulate()
    if tri.n_cells > MAX_TRIANGLES:
        tri = tri.decimate(1.0 - MAX_TRIANGLES / tri.n_cells)
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:] if tri.n_cells else np.empty((0, 3), int)
    return Mesh(name=name, points=np.asarray(tri.points) * scale, triangles=faces, color=color)


def render(scene: Scene, views: list[str], size: int = 900) -> bytes:
    if not scene.meshes:
        raise ValueError("nothing to render: the design has no bodies")
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; use {sorted(VIEWS)}")

    cols = 1 if len(views) == 1 else 2
    rows = (len(views) + cols - 1) // cols
    inches = size / 100
    fig = plt.figure(figsize=(inches, inches * rows / cols), dpi=100)
    lo, hi = _bounds(scene)
    _refine(scene, max_edge=float(np.max(hi - lo)) / 12)
    for i, view in enumerate(views):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _draw(ax, scene, view, lo, hi)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=_color(m, k), label=m.name)
        for k, m in enumerate(scene.meshes[:12])
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8, frameon=False
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _draw(ax, scene: Scene, view: str, lo: np.ndarray, hi: np.ndarray) -> None:
    elev, azim = VIEWS[view]
    ax.view_init(elev=elev, azim=azim)
    # Headlight plus a light from above, so tops read lighter than walls in every view.
    light = _view_vector(elev, azim) + np.array([0.0, 0.0, 0.9])
    light /= np.linalg.norm(light)
    # With a selection to show, fade everything else so the selection is never hidden
    # behind the faces it belongs to (matplotlib sorts polygons, it has no depth buffer).
    focus = any(m.highlight for m in scene.meshes) or bool(scene.lines)
    # All triangles go into ONE collection: matplotlib depth-sorts polygons within a
    # collection but orders whole collections by their mean depth, which would draw a
    # boss standing on a plate behind the plate.
    all_tris, all_colors = [], []
    for k, mesh in enumerate(scene.meshes):
        if not len(mesh.triangles):
            continue
        tris = mesh.points[mesh.triangles]
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        shade = 0.4 + 0.6 * np.abs(normals @ light)
        base = np.array(matplotlib.colors.to_rgb(_color(mesh, k)))
        rgb = np.clip(shade[:, None] * base[None, :], 0, 1)
        alpha = 1.0 if mesh.highlight or not focus else 0.25
        all_tris.append(tris)
        all_colors.append(np.column_stack([rgb, np.full(len(rgb), alpha)]))
    if all_tris:
        colors = np.concatenate(all_colors)
        # Edges drawn in the face colour hide the anti-aliasing seams between triangles.
        ax.add_collection3d(
            Poly3DCollection(
                np.concatenate(all_tris), facecolors=colors, edgecolors=colors, linewidths=0.3
            )
        )
    if scene.lines:
        ax.add_collection3d(Line3DCollection(scene.lines, colors=HIGHLIGHT, linewidths=2.5))

    center = (lo + hi) / 2
    half = max(float(np.max(hi - lo)) / 2, 1e-9) * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(f"X [{scene.unit}]", fontsize=8)
    ax.set_ylabel(f"Y [{scene.unit}]", fontsize=8)
    ax.set_zlabel(f"Z [{scene.unit}]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(view, fontsize=10)
    if view in ("top", "bottom"):
        ax.set_zticks([])
    elif view in ("front", "back"):
        ax.set_yticks([])
    elif view in ("right", "left"):
        ax.set_xticks([])


def _refine(scene: Scene, max_edge: float, budget: int = 60_000) -> None:
    """Split long triangles so the painter's sort works on small pieces.

    CAD tessellation turns a plate face into two huge triangles; sorted by centroid
    they draw through whatever touches them. Splitting is linear (shape unchanged) and
    stops at a triangle budget so big assemblies stay quick.
    """
    import pyvista as pv

    if max_edge <= 0 or sum(len(m.triangles) for m in scene.meshes) >= budget:
        return
    for mesh in scene.meshes:
        if not len(mesh.triangles):
            continue
        faces = np.column_stack([np.full(len(mesh.triangles), 3), mesh.triangles]).ravel()
        poly = pv.PolyData(mesh.points, faces)
        fine = poly.subdivide_adaptive(max_edge_len=max_edge, max_n_passes=4)
        if fine.n_cells > budget:
            continue
        mesh.points = np.asarray(fine.points)
        mesh.triangles = np.asarray(fine.faces).reshape(-1, 4)[:, 1:]


def _bounds(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    pts = [m.points for m in scene.meshes if len(m.points)]
    pts += [line for line in scene.lines if len(line)]
    if not pts:
        raise ValueError("nothing to render: the bodies returned no tessellation")
    allp = np.vstack(pts)
    return allp.min(axis=0), allp.max(axis=0)


def _view_vector(elev: float, azim: float) -> np.ndarray:
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _color(mesh: Mesh, index: int) -> str:
    if mesh.highlight:
        return HIGHLIGHT
    if mesh.color:
        try:
            matplotlib.colors.to_rgb(mesh.color)
            return mesh.color
        except ValueError:
            pass
    return PALETTE[index % len(PALETTE)]
