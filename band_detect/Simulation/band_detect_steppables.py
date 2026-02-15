"""Steppables for the Basu bullseye band-detect simulation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from cc3d.core.PySteppables import SteppableBasePy

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    MATPLOTLIB_AVAILABLE = True
except (ImportError, RuntimeError):
    MATPLOTLIB_AVAILABLE = False


# ─────────────────────────── helper utilities ────────────────────────────
def _bounded_range(low: int, high: int, dim: int):
    return range(max(low, 0), min(high, dim))


def _hill(value: float, k_half: float, hill_n: float) -> float:
    if value <= 0.0:
        return 0.0
    denom = k_half**hill_n + value**hill_n
    return (value**hill_n) / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class CircuitParameters:

    R_max: float = 1.0
    K_R: float = 6.4
    n_R: float = 2.7

    CI_max: float = 3.2
    CI_basal: float = 0.05
    K_CI: float = 0.32
    n_CI: float = 2.7

    LacIM1_max: float = 3.8
    LacIM1_basal: float = 0.05
    K_LacIM1: float = 0.55
    n_LacIM1: float = 3.1

    LacI_high: float = 2.1
    LacI_low: float = 0.08
    K_CI_repress: float = 0.38
    n_CI_repress: float = 4.0

    GFP_max: float = 1.0
    K_GFP_LacIM1: float = 0.7
    K_GFP_LacI: float = 0.28
    n_GFP_LacIM1: float = 2.8
    n_GFP_LacI: float = 2.2

    FFL_max: float = 1.0
    K_FFL: float = 0.35 # 0.33
    n_FFL: float = 1.9
    K_FFL_off: float = 0.85 # 0.70
    n_FFL_off: float = 3.8
    tau_FFL_up: float = 3600 # 3600.0
    tau_FFL_down: float = 500 # 500.0
    K_GFP_FFL: float = 0.40
    n_GFP_FFL: float = 4.2


class BandDetectInitializerSteppable(SteppableBasePy):
    """
    Seeds a compact sender colony at the center and a receiver monolayer elsewhere.

    The initialization follows the CC3D manual's recommendation to fill the lattice
    using direct pixel writes via self.cell_field together with newCell factory calls.
    """

    def __init__(self, frequency=1, sender_radius_px: float = 16.0, tile_size: int = 5):
        super().__init__(frequency)
        self.sender_radius = sender_radius_px
        self.tile_size = tile_size

    def start(self):
        dim_x, dim_y = int(self.dim.x), int(self.dim.y)
        half_tile = self.tile_size // 2
        center_x, center_y = dim_x / 2.0, dim_y / 2.0

        for cx in range(half_tile, dim_x - half_tile, self.tile_size):
            for cy in range(half_tile, dim_y - half_tile, self.tile_size):
                dx = cx - center_x
                dy = cy - center_y
                dist_sq = dx * dx + dy * dy
                cell_type = self.SENDER if dist_sq <= self.sender_radius**2 else self.RECEIVER
                self._seed_cell(cell_type, cx, cy, half_tile, dim_x, dim_y)

    def _seed_cell(self, cell_type_id: int, cx: int, cy: int, half_tile: int, dim_x: int, dim_y: int):
        cell = self.newCell(cell_type_id)
        for x in _bounded_range(cx - half_tile, cx + half_tile + 1, dim_x):
            for y in _bounded_range(cy - half_tile, cy + half_tile + 1, dim_y):
                if self.cell_field[x, y, 0] is None:
                    self.cell_field[x, y, 0] = cell
        cell.dict.setdefault("GFP", 0.0)


class BandDetectCircuitSteppable(SteppableBasePy):
    """
    Implements the Basu band-pass gene circuit inside the receiver lattice.

    Receiver cells sample the local AHL concentration, form an effective LuxR–AHL
    complex, and update downstream repressors (CI, LacIM1, LacI). GFP is repressed
    jointly by LacIM1 and LacI so that GFP is high only at intermediate AHL.
    """

    def __init__(self, frequency=1, params: CircuitParameters | None = None):
        super().__init__(frequency)
        self.params = params or CircuitParameters()
        self.ahl_field = None

    def start(self):
        self.ahl_field = self.field.AHL
        for cell in self.cell_list:
            self._initialize_dict(cell)

    def step(self, mcs):
        for cell in self.cell_list_by_type(self.RECEIVER):
            self._update_cell_state(cell)

    # ───────────────────────── helper methods ──────────────────────────
    def _initialize_dict(self, cell):
        cell.dict.setdefault("AHL", 0.0)
        cell.dict.setdefault("R", 0.0)
        cell.dict.setdefault("CI", self.params.CI_basal)
        cell.dict.setdefault("LacIM1", self.params.LacIM1_basal)
        cell.dict.setdefault("LacI", self.params.LacI_high)
        cell.dict.setdefault("FFL", 0.0)
        cell.dict.setdefault("GFP", 0.0)

    def _avg_ahl(self, cell) -> float:
        pixels = list(self.get_cell_pixel_list(cell))
        if not pixels:
            return 0.0
        accum = 0.0
        for pixel_data in pixels:
            x = int(pixel_data.pixel.x)
            y = int(pixel_data.pixel.y)
            accum += self.ahl_field[x, y, 0]
        return accum / len(pixels)

    def _update_cell_state(self, cell):
        p = self.params
        local_ahl = self._avg_ahl(cell)
        R_act = p.R_max * _hill(local_ahl, p.K_R, p.n_R)
        CI = p.CI_basal + p.CI_max * _hill(R_act, p.K_CI, p.n_CI)
        LacIM1 = p.LacIM1_basal + p.LacIM1_max * _hill(R_act, p.K_LacIM1, p.n_LacIM1)

        repression_from_CI = _hill(CI, p.K_CI_repress, p.n_CI_repress)
        LacI = p.LacI_low + (1.0 - repression_from_CI) * (p.LacI_high - p.LacI_low)

        low_gate = _hill(R_act, p.K_FFL, p.n_FFL)
        high_gate = _hill(R_act, p.K_FFL_off, p.n_FFL_off)
        ffl_drive = p.FFL_max * low_gate * max(0.0, 1.0 - high_gate)
        prev_ffl = cell.dict.get("FFL", 0.0)
        tau = p.tau_FFL_up if ffl_drive >= prev_ffl else p.tau_FFL_down
        tau = max(tau, 1.0)
        ffl_state = prev_ffl + (ffl_drive - prev_ffl) / tau
        ffl_state = max(0.0, min(p.FFL_max, ffl_state))
        ffl_gate = _hill(ffl_state, p.K_GFP_FFL, p.n_GFP_FFL)

        lacim1_gate = 1.0 / (1.0 + (LacIM1 / p.K_GFP_LacIM1) ** p.n_GFP_LacIM1)
        laci_gate = 1.0 / (1.0 + (LacI / p.K_GFP_LacI) ** p.n_GFP_LacI)
        GFP = p.GFP_max * lacim1_gate * laci_gate * ffl_gate

        cell.dict["AHL"] = local_ahl
        cell.dict["R"] = R_act
        cell.dict["CI"] = CI
        cell.dict["LacIM1"] = LacIM1
        cell.dict["LacI"] = LacI
        cell.dict["FFL"] = ffl_state
        cell.dict["GFP"] = GFP


class BandDetectRadialHeatmapSteppable(SteppableBasePy):
    """
    Builds a time–radius heat map of GFP for receiver cells.

    Each recorded frame averages GFP over concentric radial bins measured from
    the lattice center. A numpy array and a rendered heat map are written to a
    dedicated heatmap subdirectory, enabling quantitative inspection of the band
    detector dynamics.
    """

    def __init__(
        self,
        frequency: int = 10,
        radial_bin_width: float = 2.5,
        max_radius: float | None = None,
        run_tag: str | None = None,
        receiver_only: bool = True,
    ):
        super().__init__(frequency)
        self.radial_bin_width = float(radial_bin_width)
        self.max_radius = max_radius
        self.run_tag = run_tag or datetime.now().strftime("band_detect_%Y%m%d_%H%M%S")
        self.receiver_only = receiver_only
        self.results_dir = None
        self.heatmap_dir = None
        self.bin_edges = None
        self.bin_count = 0
        self.distance_bin_index = None
        self.time_points: List[int] = []
        self.heatmap_rows: List[np.ndarray] = []
        self._last_recorded_mcs: int | None = None

    def start(self):
        nx, ny = int(self.dim.x), int(self.dim.y)
        base_dir = Path(__file__).parent.parent / "Results"
        self.results_dir = base_dir / self.run_tag
        self.heatmap_dir = self.results_dir / "heatmap"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        bin_width = max(self.radial_bin_width, 1e-6)
        max_r = float(self.max_radius) if self.max_radius is not None else 0.5 * np.hypot(nx, ny)
        self.bin_edges = np.arange(0.0, max_r + bin_width, bin_width)
        if self.bin_edges.shape[0] < 2:
            self.bin_edges = np.array([0.0, max_r], dtype=float)
        self.bin_count = len(self.bin_edges) - 1

        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.indices((ny, nx))
        radii = np.hypot(xx - cx, yy - cy)
        bin_index = np.digitize(radii, self.bin_edges) - 1
        bin_index[(bin_index < 0) | (bin_index >= self.bin_count)] = -1
        self.distance_bin_index = bin_index.astype(np.int16)

        self._record_heatmap_slice(0)

    def step(self, mcs):
        self._record_heatmap_slice(mcs)

    def finish(self):
        final_mcs = int(self.simulator.getStep())
        if self._last_recorded_mcs != final_mcs:
            self._record_heatmap_slice(final_mcs)
        if not self.heatmap_rows:
            return

        heatmap = np.vstack(self.heatmap_rows)
        np.save(self.heatmap_dir / "gfp_radial_heatmap.npy", heatmap)

        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        csv_path = self.heatmap_dir / "gfp_radial_heatmap.csv"
        header = ["mcs", *[f"r_{c:.2f}" for c in bin_centers]]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for mcs_val, row in zip(self.time_points, heatmap):
                writer.writerow([mcs_val, *[f"{val:.6f}" for val in row]])

        metadata = {
            "run_tag": self.run_tag,
            "bin_edges": [float(x) for x in self.bin_edges],
            "time_points": [int(t) for t in self.time_points],
            "radial_bin_width": float(self.radial_bin_width),
            "max_radius": float(self.max_radius) if self.max_radius is not None else None,
            "receiver_only": bool(self.receiver_only),
        }
        with (self.heatmap_dir / "heatmap_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        if MATPLOTLIB_AVAILABLE:
            self._render_heatmap(heatmap)

    def _current_gfp_grid(self) -> np.ndarray:
        nx, ny = int(self.dim.x), int(self.dim.y)
        gfp_grid = np.zeros((ny, nx), dtype=float)
        for x in range(nx):
            for y in range(ny):
                cell = self.cell_field[x, y, 0]
                if cell is None:
                    continue
                if self.receiver_only and cell.type != self.RECEIVER:
                    continue
                gfp_grid[y, x] = cell.dict.get("GFP", 0.0)
        return gfp_grid

    def _record_heatmap_slice(self, mcs: int):
        if self.distance_bin_index is None:
            return

        gfp_grid = self._current_gfp_grid()
        flat_gfp = gfp_grid.ravel()
        flat_bins = self.distance_bin_index.ravel()
        mask = flat_bins >= 0
        if not np.any(mask):
            return

        sums = np.bincount(flat_bins[mask], weights=flat_gfp[mask], minlength=self.bin_count)
        counts = np.bincount(flat_bins[mask], minlength=self.bin_count)
        means = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0,
        )

        self.time_points.append(int(mcs))
        self.heatmap_rows.append(means)
        self._last_recorded_mcs = int(mcs)

    def _render_heatmap(self, heatmap: np.ndarray):
        if not self.time_points:
            return

        time_vals = np.array(self.time_points, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        extent = [self.bin_edges[0], self.bin_edges[-1], time_vals[0], time_vals[-1]]
        im = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
        )
        ax.set_xlabel("Distance from center (pixels)")
        ax.set_ylabel("Monte Carlo step")
        ax.set_title("Radial GFP heat map over time")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean GFP per radial bin (a.u.)")
        fig.tight_layout()
        fig.savefig(self.heatmap_dir / "gfp_radial_heatmap.png", dpi=180)
        plt.close(fig)


class BullseyeResultsSteppable(SteppableBasePy):
    """
    Periodically saves lattice snapshots and the final cell field / ID grids.

    Images render sender cells in red and receiver cells with a green intensity that
    scales with their GFP level.
    """

    def __init__(self, frequency=10, snapshot_interval=100, run_tag: str | None = None):
        super().__init__(frequency)
        self.snapshot_interval = snapshot_interval
        self.results_dir = None
        self.images_dir = None
        self.data_dir = None
        self.fields_dir = None
        self.field_arrays_dir = None
        self.gfp_display_max = 0.4
        self.gfp_quiescent_level = 0.02
        self.receiver_quiescent_color = (0.55, 0.55, 0.55)
        self.ahl_display_max = 0.05
        self.summary_rows = []
        self.radial_rows = []
        self.angular_rows = []
        self.radial_edges = None
        self.angular_segments = 24
        self.gfp_high_threshold = 0.12
        self.run_tag = run_tag or datetime.now().strftime("band_detect_%Y%m%d_%H%M%S")

    def start(self):
        base_dir = Path(__file__).parent.parent / "Results"
        self.results_dir = base_dir / self.run_tag
        self.images_dir = self.results_dir / "images"
        self.data_dir = self.results_dir / "data"
        self.fields_dir = self.results_dir / "fields"
        self.field_arrays_dir = self.results_dir / "field_arrays"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fields_dir.mkdir(parents=True, exist_ok=True)
        self.field_arrays_dir.mkdir(parents=True, exist_ok=True)
        nx, ny = int(self.dim.x), int(self.dim.y)
        max_radius = 0.5 * (nx + ny) / 2.0
        self.radial_edges = np.arange(0.0, max_radius + 5.0, 5.0)
        self._save_snapshot(0)
        self._save_ahl_field(0)
        self._record_circuit_summary(0)
        self._record_spatial_profiles(0)

    def step(self, mcs):
        if mcs % self.snapshot_interval == 0:
            self._save_snapshot(mcs)
            self._save_ahl_field(mcs)
            self._record_circuit_summary(mcs)
            self._record_spatial_profiles(mcs)

    def finish(self):
        final_mcs = int(self.simulator.getStep())
        self._save_snapshot(final_mcs)
        self._save_ahl_field(final_mcs)
        self._save_lattice_arrays(final_mcs)
        self._record_circuit_summary(final_mcs)
        self._record_spatial_profiles(final_mcs)
        self._write_metadata(final_mcs)
        self._write_summary_table()
        self._write_radial_profiles()
        self._write_angular_profiles()

    # ───────────────────────────── snapshots ────────────────────────────
    def _save_snapshot(self, mcs: int):
        if not MATPLOTLIB_AVAILABLE:
            return

        nx, ny = int(self.dim.x), int(self.dim.y)
        rgb, borders = self._rgb_image(nx, ny)
        if borders is not None:
            rgb[borders] = (1.0, 1.0, 0.0)

        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Basu band-detect at MCS {mcs}", color="white", fontsize=11)
        legend_items = [
            Patch(color=(0.85, 0.2, 0.2), label="Sender"),
            Patch(color=self.receiver_quiescent_color, label="Receiver (OFF)"),
            Patch(color=(0.12, 0.9, 0.18), label="Receiver (GFP ON)"),
            Patch(color=(0.02, 0.02, 0.02), label="Medium"),
        ]
        ax.legend(
            handles=legend_items,
            loc="upper right",
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
            fontsize=8,
        )
        fig.tight_layout(pad=0.05)
        fname = self.images_dir / f"snapshot_mcs_{mcs:04d}.png"
        fig.savefig(fname, dpi=160, facecolor="black")
        plt.close(fig)

    def _rgb_image(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
        img = np.zeros((nx, ny, 3), dtype=float)
        ids = -np.ones((nx, ny), dtype=int)
        for x in range(nx):
            for y in range(ny):
                cell = self.cell_field[x, y, 0]
                if cell is None:
                    img[x, y] = (0.02, 0.02, 0.02)
                    continue
                ids[x, y] = cell.id
                if cell.type == self.SENDER:
                    img[x, y] = (0.85, 0.2, 0.2)
                elif cell.type == self.RECEIVER:
                    gfp = cell.dict.get("GFP", 0.0)
                    if gfp <= self.gfp_quiescent_level:
                        img[x, y] = self.receiver_quiescent_color
                    else:
                        denom = max(self.gfp_display_max - self.gfp_quiescent_level, 1e-6)
                        g_norm = min(1.0, (gfp - self.gfp_quiescent_level) / denom)
                        img[x, y] = (0.10, 0.35 + 0.60 * g_norm, 0.10 + 0.05 * g_norm)
                else:
                    img[x, y] = (0.4, 0.4, 0.4)

        borders = np.zeros((nx, ny), dtype=bool)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.roll(ids, (dx, dy), axis=(0, 1))
            mask = (ids != shifted) & (ids >= 0) & (shifted >= 0) & (ids < shifted)
            borders |= mask
        return img.transpose(1, 0, 2), borders.transpose(1, 0)

    # ───────────────────────────── data dumps ───────────────────────────
    def _save_lattice_arrays(self, mcs: int):
        nx, ny = int(self.dim.x), int(self.dim.y)
        cell_types = np.zeros((nx, ny), dtype=np.int16)
        cell_ids = -np.ones((nx, ny), dtype=np.int32)
        gfp_grid = np.zeros((nx, ny), dtype=float)

        for x in range(nx):
            for y in range(ny):
                cell = self.cell_field[x, y, 0]
                if cell:
                    cell_types[x, y] = cell.type
                    cell_ids[x, y] = cell.id
                    if cell.type == self.RECEIVER:
                        gfp_grid[x, y] = cell.dict.get("GFP", 0.0)

        np.save(self.data_dir / f"cell_types_mcs_{mcs:04d}.npy", cell_types)
        np.save(self.data_dir / f"cell_ids_mcs_{mcs:04d}.npy", cell_ids)
        np.save(self.data_dir / f"gfp_grid_mcs_{mcs:04d}.npy", gfp_grid)
        np.save(self.data_dir / f"ahl_grid_mcs_{mcs:04d}.npy", self._copy_ahl_field())

    def _write_metadata(self, mcs: int):
        metadata = {
            "tag": self.run_tag,
            "final_mcs": int(mcs),
            "snapshot_interval": self.snapshot_interval,
            "results_dir": str(self.results_dir),
        }
        with (self.results_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _copy_ahl_field(self):
        nx, ny = int(self.dim.x), int(self.dim.y)
        ahl_field = np.zeros((nx, ny), dtype=float)
        field_handle = self.field.AHL
        for x in range(nx):
            for y in range(ny):
                ahl_field[x, y] = field_handle[x, y, 0]
        return ahl_field

    def _save_ahl_field(self, mcs: int):
        ahl_data = self._copy_ahl_field()
        if self.field_arrays_dir is not None:
            np.save(self.field_arrays_dir / f"ahl_mcs_{mcs:04d}.npy", ahl_data)
        if not MATPLOTLIB_AVAILABLE:
            return
        ahl = ahl_data.transpose(1, 0)
        vmax = max(self.ahl_display_max, np.percentile(ahl, 99))
        self.ahl_display_max = vmax if vmax > 1e-6 else self.ahl_display_max
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        im = ax.imshow(
            ahl,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=self.ahl_display_max,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"AHL field at MCS {mcs}", color="white", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white")
        cbar.set_label("AHL (a.u.)", color="white")
        fig.tight_layout(pad=0.05)
        fig.savefig(self.fields_dir / f"ahl_mcs_{mcs:04d}.png", dpi=160, facecolor="black")
        plt.close(fig)

    def _record_circuit_summary(self, mcs: int):
        rows = []
        for cell in self.cell_list_by_type(self.RECEIVER):
            rows.append(
                [
                    cell.dict.get("AHL", 0.0),
                    cell.dict.get("R", 0.0),
                    cell.dict.get("CI", 0.0),
                    cell.dict.get("LacIM1", 0.0),
                    cell.dict.get("LacI", 0.0),
                    cell.dict.get("FFL", 0.0),
                    cell.dict.get("GFP", 0.0),
                ]
            )
        if not rows:
            return
        arr = np.array(rows)
        means = arr.mean(axis=0)
        self.summary_rows.append([mcs, *means])

    def _write_summary_table(self):
        if not self.summary_rows:
            return
        header = ["mcs", "AHL", "R", "CI", "LacIM1", "LacI", "FFL", "GFP"]
        summary_path = self.results_dir / "receiver_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(self.summary_rows)

    def _current_gfp_grid(self):
        nx, ny = int(self.dim.x), int(self.dim.y)
        gfp_grid = np.zeros((ny, nx), dtype=float)
        for x in range(nx):
            for y in range(ny):
                cell = self.cell_field[x, y, 0]
                if cell and cell.type == self.RECEIVER:
                    gfp_grid[y, x] = cell.dict.get("GFP", 0.0)
        return gfp_grid

    def _record_spatial_profiles(self, mcs: int):
        if self.radial_edges is None:
            return
        gfp = self._current_gfp_grid()
        ahl = self._copy_ahl_field().transpose(1, 0)
        ny, nx = gfp.shape
        yy, xx = np.indices((ny, nx))
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        radii = np.hypot(yy - cy, xx - cx)
        angles = (np.arctan2(yy - cy, xx - cx) + 2.0 * np.pi) % (2.0 * np.pi)

        for start, end in zip(self.radial_edges[:-1], self.radial_edges[1:]):
            mask = (radii >= start) & (radii < end)
            if not mask.any():
                continue
            ahl_vals = ahl[mask]
            gfp_vals = gfp[mask]
            frac_high = np.mean(gfp_vals >= self.gfp_high_threshold)
            self.radial_rows.append(
                [
                    mcs,
                    float(start),
                    float(end),
                    float(ahl_vals.mean()),
                    float(ahl_vals.std()),
                    float(gfp_vals.mean()),
                    float(gfp_vals.std()),
                    float(frac_high),
                ]
            )

        seg_width = 2.0 * np.pi / self.angular_segments
        radial_mask = (radii >= 25.0) & (radii <= 40.0)
        for seg in range(self.angular_segments):
            ang_mask = (angles >= seg * seg_width) & (angles < (seg + 1) * seg_width)
            mask = radial_mask & ang_mask
            if not mask.any():
                continue
            gfp_vals = gfp[mask]
            self.angular_rows.append(
                [
                    mcs,
                    float(seg),
                    float(seg_width),
                    float(gfp_vals.mean()),
                    float(np.mean(gfp_vals >= self.gfp_high_threshold)),
                ]
            )

    def _write_radial_profiles(self):
        if not self.radial_rows:
            return
        header = [
            "mcs",
            "r_start",
            "r_end",
            "ahl_mean",
            "ahl_std",
            "gfp_mean",
            "gfp_std",
            "gfp_frac_high",
        ]
        path = self.results_dir / "radial_profile.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(self.radial_rows)

    def _write_angular_profiles(self):
        if not self.angular_rows:
            return
        header = ["mcs", "segment", "segment_width", "gfp_mean", "gfp_high_fraction"]
        path = self.results_dir / "angular_profile.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(self.angular_rows)

