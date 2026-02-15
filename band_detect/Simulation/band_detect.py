"""Main entry point for the Basu bullseye band-detect simulation."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cc3d.CompuCellSetup as CompuCellSetup

try:
    from .band_detect_steppables import (
        CircuitParameters,
        BandDetectCircuitSteppable,
        BandDetectInitializerSteppable,
        BandDetectRadialHeatmapSteppable,
        BullseyeResultsSteppable,
    )
except ImportError:
    # Fallback for execution contexts where relative imports are unavailable.
    steppables_path = Path(__file__).resolve().parent / "band_detect_steppables.py"
    spec = importlib.util.spec_from_file_location("band_detect_steppables", steppables_path)
    if spec is None or spec.loader is None:
        raise
    steppables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(steppables)
    CircuitParameters = steppables.CircuitParameters
    BandDetectCircuitSteppable = steppables.BandDetectCircuitSteppable
    BandDetectInitializerSteppable = steppables.BandDetectInitializerSteppable
    BandDetectRadialHeatmapSteppable = steppables.BandDetectRadialHeatmapSteppable
    BullseyeResultsSteppable = steppables.BullseyeResultsSteppable

globals().update(
    {
        "CircuitParameters": CircuitParameters,
        "BandDetectCircuitSteppable": BandDetectCircuitSteppable,
        "BandDetectInitializerSteppable": BandDetectInitializerSteppable,
        "BandDetectRadialHeatmapSteppable": BandDetectRadialHeatmapSteppable,
        "BullseyeResultsSteppable": BullseyeResultsSteppable,
        "os": os,
        "json": json,
        "datetime": datetime,
        "Path": Path,
        "fields": fields,
        "replace": replace,
    }
)


@dataclass(frozen=True)
class RunConfig:
    run_tag: str
    sender_radius_px: float
    tile_size: int
    circuit_params: CircuitParameters
    gfp_threshold: float
    snapshot_interval: int
    p_on: float
    k_on: int
    t_min_frac: float
    t_mid_frac: float
    t_max_frac: float
    window_snapshots: int
    sigma_r_star_px: float
    p_min_end: float
    p_early: float
    alpha: float
    gate_c_enabled: bool
    gate_c_t_frac: float
    gate_c_p_max: float
    enable_heatmap: bool
    heatmap_bin_width: float
    heatmap_frequency: int
    save_midrun_arrays: bool
    save_midrun_images: bool
    params_path: Path | None


globals()["RunConfig"] = RunConfig

def _load_run_config() -> RunConfig:
    """
    Load run-time parameters from a params.json file written by the orchestrator.

    The orchestrator places params.json in the run output directory and sets
    BAND_PARAMS_PATH so that the steppable layer can consume it. Environment
    variables remain as a backward-compatible fallback.
    """
    import os
    import json
    from pathlib import Path
    from dataclasses import fields, replace
    from datetime import datetime

    params_path_env = os.environ.get("BAND_PARAMS_PATH")
    params_data: Dict[str, Any] = {}
    params_path: Path | None = None
    if params_path_env:
        candidate = Path(params_path_env)
        if candidate.exists():
            try:
                params_data = json.loads(candidate.read_text())
                params_path = candidate
            except Exception:
                params_data = {}

    # Fallback to environment variables if no params.json is available.
    run_tag_default = datetime.now().strftime("band_detect_%Y%m%d_%H%M%S")
    run_tag = str(params_data.get("run_tag", os.environ.get("BAND_RUN_TAG", run_tag_default)))
    sender_radius = float(params_data.get("sender_radius_px", os.environ.get("BAND_SENDER_RADIUS_PX", 16.0)))
    tile_size = int(params_data.get("tile_size", os.environ.get("BAND_TILE_SIZE", 5)))
    gfp_threshold = float(params_data.get("gfp_threshold", os.environ.get("BAND_GFP_THRESHOLD", 0.12)))
    snapshot_interval = int(params_data.get("snapshot_interval", os.environ.get("BAND_SNAPSHOT_INTERVAL", 100)))

    temporal_payload = params_data.get("temporal_params", {})
    if isinstance(temporal_payload, str):
        try:
            temporal_payload = json.loads(temporal_payload)
        except json.JSONDecodeError:
            temporal_payload = {}
    if not isinstance(temporal_payload, dict):
        temporal_payload = {}

    def _temporal_float(key: str, default: float) -> float:
        return float(temporal_payload.get(key, default))

    def _temporal_int(key: str, default: int) -> int:
        return int(temporal_payload.get(key, default))

    def _temporal_bool(key: str, default: bool) -> bool:
        val = temporal_payload.get(key, default)
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y")
        return bool(val)

    p_on = _temporal_float("p_on", 0.30)
    k_on = _temporal_int("k_on", 3)
    t_min_frac = _temporal_float("t_min_frac", 0.10)
    t_mid_frac = _temporal_float("t_mid_frac", 0.35)
    t_max_frac = _temporal_float("t_max_frac", 0.60)
    window_snapshots = _temporal_int("window_snapshots", 8)
    sigma_r_star_px = _temporal_float("sigma_r_star_px", 5.0)
    p_min_end = _temporal_float("p_min_end", 0.25)
    p_early = _temporal_float("p_early", 0.35)
    alpha = _temporal_float("alpha", 0.70)
    gate_c_enabled = _temporal_bool("gate_c_enabled", False)
    gate_c_t_frac = _temporal_float("gate_c_t_frac", 0.40)
    gate_c_p_max = _temporal_float("gate_c_p_max", 0.15)

    output_payload = params_data.get("output_controls", {})
    if isinstance(output_payload, str):
        try:
            output_payload = json.loads(output_payload)
        except json.JSONDecodeError:
            output_payload = {}
    if not isinstance(output_payload, dict):
        output_payload = {}

    def _output_float(key: str, default: float) -> float:
        return float(output_payload.get(key, default))

    def _output_int(key: str, default: int) -> int:
        return int(output_payload.get(key, default))

    def _output_bool(key: str, default: bool) -> bool:
        val = output_payload.get(key, default)
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y")
        return bool(val)

    enable_heatmap = _output_bool("enable_heatmap", False)
    heatmap_bin_width = _output_float("heatmap_bin_width", 2.5)
    heatmap_frequency = _output_int("heatmap_frequency", snapshot_interval)
    save_midrun_arrays = _output_bool("save_midrun_arrays", False)
    save_midrun_images = _output_bool("save_midrun_images", False)

    # Circuit parameters can be provided either inside params.json or via BAND_CIRCUIT_PARAMS.
    base_params = CircuitParameters()
    circuit_payload = params_data.get("circuit_params", {})
    if isinstance(circuit_payload, str):
        try:
            circuit_payload = json.loads(circuit_payload)
        except json.JSONDecodeError:
            circuit_payload = {}
    env_payload = os.environ.get("BAND_CIRCUIT_PARAMS")
    if env_payload and not circuit_payload:
        try:
            circuit_payload = json.loads(env_payload)
        except json.JSONDecodeError:
            circuit_payload = {}
    allowed = {f.name for f in fields(CircuitParameters)}
    updates = {k: float(v) for k, v in circuit_payload.items() if k in allowed}
    circuit_params = replace(base_params, **updates) if updates else base_params

    return RunConfig(
        run_tag=run_tag,
        sender_radius_px=sender_radius,
        tile_size=tile_size,
        circuit_params=circuit_params,
        gfp_threshold=gfp_threshold,
        snapshot_interval=snapshot_interval,
        p_on=p_on,
        k_on=k_on,
        t_min_frac=t_min_frac,
        t_mid_frac=t_mid_frac,
        t_max_frac=t_max_frac,
        window_snapshots=window_snapshots,
        sigma_r_star_px=sigma_r_star_px,
        p_min_end=p_min_end,
        p_early=p_early,
        alpha=alpha,
        gate_c_enabled=gate_c_enabled,
        gate_c_t_frac=gate_c_t_frac,
        gate_c_p_max=gate_c_p_max,
        enable_heatmap=enable_heatmap,
        heatmap_bin_width=heatmap_bin_width,
        heatmap_frequency=heatmap_frequency,
        save_midrun_arrays=save_midrun_arrays,
        save_midrun_images=save_midrun_images,
        params_path=params_path,
    )


RUN_CONFIG = _load_run_config()

CompuCellSetup.register_steppable(
    steppable=BandDetectInitializerSteppable(
        frequency=1,
        sender_radius_px=RUN_CONFIG.sender_radius_px,
        tile_size=RUN_CONFIG.tile_size,
    )
)
CompuCellSetup.register_steppable(
    steppable=BandDetectCircuitSteppable(frequency=1, params=RUN_CONFIG.circuit_params)
)
if RUN_CONFIG.enable_heatmap:
    CompuCellSetup.register_steppable(
        steppable=BandDetectRadialHeatmapSteppable(
            frequency=RUN_CONFIG.heatmap_frequency,
            radial_bin_width=RUN_CONFIG.heatmap_bin_width,
            run_tag=RUN_CONFIG.run_tag,
            receiver_only=True,
            enabled=True,
        )
    )
CompuCellSetup.register_steppable(
    steppable=BullseyeResultsSteppable(
        frequency=10,
        snapshot_interval=RUN_CONFIG.snapshot_interval,
        run_tag=RUN_CONFIG.run_tag,
        gfp_threshold=RUN_CONFIG.gfp_threshold,
        params_json_path=RUN_CONFIG.params_path,
        p_on=RUN_CONFIG.p_on,
        k_on=RUN_CONFIG.k_on,
        t_min_frac=RUN_CONFIG.t_min_frac,
        t_mid_frac=RUN_CONFIG.t_mid_frac,
        t_max_frac=RUN_CONFIG.t_max_frac,
        window_snapshots=RUN_CONFIG.window_snapshots,
        sigma_r_star_px=RUN_CONFIG.sigma_r_star_px,
        p_min_end=RUN_CONFIG.p_min_end,
        p_early=RUN_CONFIG.p_early,
        alpha=RUN_CONFIG.alpha,
        gate_c_enabled=RUN_CONFIG.gate_c_enabled,
        gate_c_t_frac=RUN_CONFIG.gate_c_t_frac,
        gate_c_p_max=RUN_CONFIG.gate_c_p_max,
        save_midrun_arrays=RUN_CONFIG.save_midrun_arrays,
        save_midrun_images=RUN_CONFIG.save_midrun_images,
    )
)

CompuCellSetup.run()

