"""One connection to a SpaceClaim/Discovery/Geometry-service backend, shared by all tools.

The session owns the ``Modeler``, the active ``Design`` and a lock: MCP runs sync tools
in worker threads, and a PyAnsys Geometry client must not be driven from two threads
at once.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .units import Units

log = logging.getLogger(__name__)

Product = Literal["spaceclaim", "discovery", "geometry-service"]

# A backend reports these as (major, minor, patch); the product calls 25.2 "2025 R2".
MIN_BACKEND = (25, 2, 0)


class SessionError(RuntimeError):
    """A failure the model can act on (wrong state, unknown id, unsupported backend)."""


@dataclass
class BodyState:
    """What a body looked like at one instant; enough to tell whether an edit did anything."""

    id: str
    name: str
    detailed: bool = False
    volume: float | None = None
    faces: int | None = None
    edges: int | None = None
    bbox: tuple[float, ...] | None = None

    def same_as(self, other: BodyState) -> bool:
        if not (self.detailed and other.detailed):
            return True  # only presence was recorded
        if (self.faces, self.edges) != (other.faces, other.edges):
            return False
        if not _close(self.volume, other.volume):
            return False
        if self.bbox is None or other.bbox is None:
            return self.bbox == other.bbox
        return all(_close(a, b) for a, b in zip(self.bbox, other.bbox, strict=True))


@dataclass
class Session:
    units: Units = field(default_factory=Units)
    modeler: Any = None
    design: Any = None
    product: str | None = None
    launched: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    # --- connection -------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.modeler is not None

    def launch(self, product: Product, version: int | None, hidden: bool, timeout: int) -> None:
        """Start a new product instance with the ApiServer add-in and connect to it."""
        from ansys.geometry.core import (
            launch_modeler_with_core_service,
            launch_modeler_with_discovery,
            launch_modeler_with_spaceclaim,
        )

        self.disconnect()
        kwargs: dict[str, Any] = {"version": version, "timeout": timeout}
        if product in ("spaceclaim", "discovery"):
            kwargs["hidden"] = hidden
            # The launcher defaults the add-in API to the newest one PyAnsys knows (27.2),
            # which an older product rejects. Pin it to the product's own release.
            api = _api_version(version)
            if api is not None:
                kwargs["api_version"] = api
        launcher = {
            "spaceclaim": launch_modeler_with_spaceclaim,
            "discovery": launch_modeler_with_discovery,
            "geometry-service": launch_modeler_with_core_service,
        }[product]
        self.modeler = launcher(**kwargs)
        self.product = product
        self.launched = True
        self._check_backend()

    def attach(self, host: str, port: int, transport_mode: str | None) -> None:
        """Connect to a product that is already running with the ApiServer add-in."""
        from ansys.geometry.core import Modeler

        self.disconnect()
        kwargs: dict[str, Any] = {"host": host, "port": port}
        if transport_mode:
            kwargs["transport_mode"] = transport_mode
        self.modeler = Modeler(**kwargs)
        self.product = "attached"
        self.launched = False
        self._check_backend()
        # Pick up the document the user already has open, if any.
        self.design = _safe(self.modeler.read_existing_design)

    def disconnect(self) -> None:
        """Drop the connection.

        PyAnsys Geometry shuts a product down when the client that launched it closes,
        so a product started by ``launch`` exits here; an attached one keeps running.
        """
        if self.modeler is None:
            return
        try:
            self.modeler.close(close_design=False)
        except Exception as exc:  # the backend may already be gone
            log.warning("error while disconnecting: %s", exc)
        finally:
            self.modeler = None
            self.design = None
            self.product = None
            self.launched = False

    def _check_backend(self) -> None:
        version = self.backend_version
        if version is not None and tuple(version[:2]) < MIN_BACKEND[:2]:
            found = ".".join(str(v) for v in version[:2])
            self.disconnect()
            raise SessionError(
                f"backend version {found} is too old: this server needs 2025 R2 (25.2) or newer"
            )

    @property
    def backend_version(self) -> tuple[int, ...] | None:
        if self.modeler is None:
            return None
        v = _safe(lambda: self.modeler.client.backend_version)
        if v is None:
            return None
        return (int(v.major), int(v.minor), int(v.patch))

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {"connected": self.connected, "units": self.units.name}
        if not self.connected:
            return info
        version = self.backend_version
        info.update(
            product=self.product,
            launched_by_server=self.launched,
            backend_version=".".join(str(v) for v in version) if version else None,
            backend_type=_safe(lambda: self.modeler.client.backend_type.name),
            healthy=_safe(lambda: bool(self.modeler.client.healthy)),
        )
        if self.design is not None:
            info["design"] = _safe(lambda: self.design.name)
            info["bodies"] = _safe(lambda: len(self.design.get_all_bodies()))
        return info

    # --- design -------------------------------------------------------------------

    def require_modeler(self):
        if self.modeler is None:
            raise SessionError("not connected: call spaceclaim_connect first")
        return self.modeler

    def require_design(self):
        self.require_modeler()
        if self.design is None or not _safe(lambda: self.design.is_active, default=False):
            # The user may have opened a document in the UI; pick it up if so.
            self.design = _safe(self.modeler.read_existing_design)
        if self.design is None:
            raise SessionError("no design is open: call new_design or open_file first")
        return self.design

    def refresh_design(self):
        """Re-read the active design from the backend (after UI edits or scripts)."""
        modeler = self.require_modeler()
        design = _safe(lambda: modeler.get_active_design(sync_with_backend=True))
        self.design = design if design is not None else modeler.read_existing_design()
        return self.design

    def bodies(self) -> list:
        return list(self.require_design().get_all_bodies())

    def body(self, ref: str):
        """Resolve a body by id, or by name when the name is unique."""
        bodies = self.bodies()
        for b in bodies:
            if b.id == ref:
                return b
        named = [b for b in bodies if b.name == ref]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            ids = ", ".join(b.id for b in named)
            raise SessionError(f"{len(named)} bodies are named {ref!r} ({ids}); use the body id")
        known = ", ".join(f"{b.name} ({b.id})" for b in bodies[:20]) or "none"
        raise SessionError(f"no body with id or name {ref!r}; bodies: {known}")

    def faces(self, body_ref: str, face_ids: Iterable[str]) -> list:
        body = self.body(body_ref)
        by_id = {f.id: f for f in body.faces}
        return [_pick(by_id, fid, "face", body) for fid in face_ids]

    def edges(self, body_ref: str, edge_ids: Iterable[str]) -> list:
        body = self.body(body_ref)
        by_id = {e.id: e for e in body.edges}
        return [_pick(by_id, eid, "edge", body) for eid in edge_ids]

    # --- change tracking ------------------------------------------------------------

    def snapshot(self, focus: Iterable[str] | None = None) -> dict[str, BodyState]:
        """Record every body's presence, and full metrics for the ``focus`` ids.

        ``focus=None`` measures every body. Measuring costs a few backend round trips per
        body, so edits pass just the bodies they touch; any other body that appears or
        disappears is still reported.
        """
        wanted = None if focus is None else set(focus)
        states = {}
        for b in self.bodies():
            if wanted is not None and b.id not in wanted:
                states[b.id] = BodyState(id=b.id, name=b.name)
                continue
            box = _safe(lambda b=b: b.bounding_box)
            states[b.id] = BodyState(
                id=b.id,
                name=b.name,
                detailed=True,
                volume=_safe(lambda b=b: self.units.volume_out(b.volume)),
                faces=_safe(lambda b=b: len(b.faces)),
                edges=_safe(lambda b=b: len(b.edges)),
                bbox=None
                if box is None
                else tuple(
                    self.units.point_out(box.min_corner) + self.units.point_out(box.max_corner)
                ),
            )
        return states

    def diff(self, before: dict[str, BodyState], after: dict[str, BodyState]) -> dict[str, Any]:
        created = [_state_dict(after[i]) for i in after if i not in before]
        deleted = [_state_dict(before[i]) for i in before if i not in after]
        modified = []
        for i in before.keys() & after.keys():
            if not before[i].same_as(after[i]):
                entry = _state_dict(after[i])
                if before[i].volume is not None and after[i].volume is not None:
                    entry["volume_change"] = round(after[i].volume - before[i].volume, 6)
                modified.append(entry)
        return {"created": created, "deleted": deleted, "modified": modified}


def _api_version(version: int | None):
    if version is None:
        return None
    from ansys.geometry.core.connection.backend import ApiVersions

    known = sorted(v.value for v in ApiVersions)
    eligible = [v for v in known if v <= int(version)]
    return ApiVersions(eligible[-1]) if eligible else None


def _pick(by_id: dict, ref: str, kind: str, body) -> Any:
    if ref in by_id:
        return by_id[ref]
    raise SessionError(
        f"{kind} {ref!r} is not on body {body.name} ({body.id}); "
        f"call describe_body or select_{kind}s to get current ids"
    )


def _state_dict(state: BodyState) -> dict[str, Any]:
    out: dict[str, Any] = {"id": state.id, "name": state.name}
    if state.detailed:
        out.update(volume=state.volume, faces=state.faces, edges=state.edges)
        if state.bbox is not None:
            out["bbox"] = {"min": list(state.bbox[:3]), "max": list(state.bbox[3:])}
    return out


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a == b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default
