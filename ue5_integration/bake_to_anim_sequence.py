"""
bake_to_anim_sequence.py
Empathetic Digital Concierge — UE 5.7.4 AnimSequence Baker

Reads an inference CSV (260-dim per frame, 30 FPS), maps column indices
to MetaHuman Control Rig curve names via blendshape_map.py, and writes
a native AnimSequence .uasset into the Content Browser.

USAGE
  Tools → Execute Python Script → select this file, OR
  paste into the UE Output Log Python console.

CONFIGURE the five variables in the CONFIG block before running.
"""

import csv
import sys
import os

# ── UE import ────────────────────────────────────────────────────────────────
import unreal

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these before running
# ════════════════════════════════════════════════════════════════════════════

# Absolute path to your inference CSV on Windows
CSV_PATH = r"D:\Major\Empathetic-Concierge-Project_v3\joy_anim.csv"

# Folder containing blendshape_map.py
MAP_MODULE_DIR = r"D:\Major\Empathetic-Concierge-Project_v3"

# UE Content Browser path to the MetaHuman FACE skeleton
# To find it: open your MetaHuman Blueprint → select the Face mesh component
# → Details → Skeletal Mesh → double-click → Right-click the Skeleton in the
# asset header → "Copy Reference" — paste the /Game/... part here.
# Updated to the specific character skeleton found in your logs
SKELETON_PATH = "/Script/Engine.Skeleton'/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton'"
# Where to create the output asset (no file extension)
OUTPUT_ASSET = "/Game/EmpathConcierge/Animations/joy_anim"

# Must match the FPS used during inference (inference_v3.py uses 30)
FPS = 30

# ════════════════════════════════════════════════════════════════════════════
#  END CONFIG
# ════════════════════════════════════════════════════════════════════════════


# ── Step 1 — Mock torch so blendshape_map.py can be imported inside UE ───────

def _install_torch_mock():
    """
    UE's embedded Python has no torch. We mock the entire package tree so that
    any 'import torch' inside blendshape_map (or files it imports) succeeds.

    Key requirement: blendshape_map.build_loss_weights() does
        w = torch.ones(260)
        w[43:51] = 0.2          # scalar broadcast into a slice
    Plain Python lists reject this (requires iterable). _FakeTensor fixes it.
    """
    from unittest.mock import MagicMock

    class _FakeTensor(list):
        """Minimal torch.Tensor stand-in that supports scalar slice-assignment."""

        def __setitem__(self, key, value):
            if isinstance(key, slice):
                for i in range(*key.indices(len(self))):
                    list.__setitem__(self, i, float(value) if not hasattr(value, "__iter__") else value)
            else:
                list.__setitem__(self, key, value)

        def __getitem__(self, key):
            if isinstance(key, slice):
                return _FakeTensor(list.__getitem__(self, key))
            return list.__getitem__(self, key)

        def __getattr__(self, name):
            # Silently absorb .float(), .cuda(), .to(), etc.
            return lambda *a, **kw: self

    def _ones(size, *args, **kwargs):
        n = size if isinstance(size, int) else size[0]
        return _FakeTensor([1.0] * n)

    def _zeros(size, *args, **kwargs):
        n = size if isinstance(size, int) else size[0]
        return _FakeTensor([0.0] * n)

    def _tensor(data, *args, **kwargs):
        return _FakeTensor(data) if hasattr(data, "__iter__") else data

    mock_torch = MagicMock()
    mock_torch.ones   = _ones
    mock_torch.zeros  = _zeros
    mock_torch.tensor = _tensor

    for mod_name in (
        "torch", "torch.nn", "torch.nn.functional",
        "torch.optim", "torch.utils", "torch.utils.data",
        "numpy", "scipy", "scipy.signal",
    ):
        sys.modules[mod_name] = MagicMock()

    sys.modules["torch"] = mock_torch


_install_torch_mock()

if MAP_MODULE_DIR not in sys.path:
    sys.path.insert(0, MAP_MODULE_DIR)

# Force-evict any stale cached import from a previous run in this UE session
import importlib
if "blendshape_map" in sys.modules:
    del sys.modules["blendshape_map"]

try:
    import blendshape_map
    unreal.log("[EmpathConcierge] blendshape_map imported successfully.")
    unreal.log(f"  LOSS_WEIGHTS present: {hasattr(blendshape_map, 'LOSS_WEIGHTS')}")
    if hasattr(blendshape_map, "LOSS_WEIGHTS"):
        unreal.log(f"  LOSS_WEIGHTS length : {len(blendshape_map.LOSS_WEIGHTS)}")
except Exception as e:
    unreal.log_error(f"[EmpathConcierge] Could not import blendshape_map: {e}")
    raise


# ── Step 2 — Build index → MetaHuman curve-name lookup ──────────────────────
#
# blendshape_map.py exposes a REGIONS dict structured as:
#   REGIONS = { "mouthSmileL": 77, "mouthSmileR": 78, ... }
#   or        { "mouthSmile": [77, 78], ... }
#
# The convention for MetaHuman Control Rig float curves is:
#   CTRL_expressions_<stem>   (single index)
#   CTRL_expressions_<stem>L  / CTRL_expressions_<stem>R  (paired)
#
# Dims 251-256 → head pose (HP_), dims 257-259 → flags (kept generic).

def _build_index_map() -> dict:
    index_map = {}

    # 1. Preferred: Use a flat index→name mapping if defined in your module
    if hasattr(blendshape_map, "INDEX_TO_NAME"):
        for k, v in blendshape_map.INDEX_TO_NAME.items():
            index_map[int(k)] = str(v)
        unreal.log(f"[EmpathConcierge] Loaded INDEX_TO_NAME directly ({len(index_map)} entries).")
        return index_map

    # 2. Safety Map: Ensure standard MetaHuman smiles work (Indices 0-51 are usually ARKit)
    # If your model uses standard ARKit ordering, this is your 'insurance policy'
    standard_arkit_names = {
        77: "mouthSmileL", 78: "mouthSmileR", # Common indices for Joy
        1:  "eyeBlinkL",   2:  "eyeBlinkR",    # Common indices for Blinks
        25: "jawOpen"                          # Common index for Speech
    }

    # 3. Standard: Process the REGIONS dict from blendshape_map.py
    if hasattr(blendshape_map, "REGIONS"):
        prefix = "CTRL_expressions_"
        for stem, payload in blendshape_map.REGIONS.items():
            if isinstance(payload, (list, tuple)):
                # If you have "mouthSmile": [77, 78], this creates:
                # CTRL_expressions_mouthSmileL and CTRL_expressions_mouthSmileR
                suffixes = ("L", "R") if len(payload) == 2 else [str(i) for i in range(len(payload))]
                for idx, sfx in zip(payload, suffixes):
                    index_map[int(idx)] = f"{prefix}{stem}{sfx}"
            else:
                # If you have "jawOpen": 25, this creates: CTRL_expressions_jawOpen
                index_map[int(payload)] = f"{prefix}{stem}"

    # 4. Final Fallback: Fill gaps with the correct MetaHuman prefix
    for i in range(260):
        if i not in index_map:
            # Check the safety map first
            if i in standard_arkit_names:
                index_map[i] = f"CTRL_expressions_{standard_arkit_names[i]}"
            elif i < 251:
                index_map[i] = f"CTRL_expressions_custom_{i:03d}"
            elif i < 257:
                index_map[i] = f"CTRL_headPose_{i-251}"
            else:
                index_map[i] = f"curve_flag_{i-257}"

    unreal.log(f"[EmpathConcierge] Built MetaHuman-ready index map: {len(index_map)} entries.")
    return index_map

# ── Step 3 — Read the inference CSV ─────────────────────────────────────────
#
# Handles two formats:
#   A) Raw inference output  : 260 columns, no header row
#   B) Live Link Face export : 262 columns, row 0 = header, cols 0-1 = Frame/Time_sec

def _read_csv(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[EmpathConcierge] CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))

    if not raw_rows:
        raise ValueError(f"[EmpathConcierge] CSV is empty: {path}")

    # Detect header row — first cell non-numeric means it's a header
    start_row = 0
    try:
        float(raw_rows[0][0])
    except (ValueError, IndexError):
        start_row = 1

    # Detect leading metadata columns (Frame, Time_sec)
    sample_row  = raw_rows[start_row] if len(raw_rows) > start_row else raw_rows[0]
    col_offset  = max(0, len(sample_row) - 260)

    frames = []
    for row in raw_rows[start_row:]:
        if not row:
            continue
        try:
            vals = [float(v) for v in row[col_offset : col_offset + 260]]
            if vals:
                frames.append(vals)
        except ValueError:
            pass  # skip malformed rows

    n_dims = len(frames[0]) if frames else 0
    unreal.log(
        f"[EmpathConcierge] CSV loaded: {len(frames)} frames × {n_dims} dims "
        f"(header_rows={start_row}, col_offset={col_offset})"
    )
    return frames, n_dims


FRAMES, N_DIMS   = _read_csv(CSV_PATH)
N_FRAMES         = len(FRAMES)
SEQ_LENGTH_SEC   = N_FRAMES / FPS

if N_FRAMES == 0:
    raise RuntimeError("[EmpathConcierge] No frames parsed from CSV. Check CSV_PATH and file format.")


# ── Step 4 — Resolve the MetaHuman face skeleton ─────────────────────────────

def _load_skeleton():
    skel = unreal.load_asset(SKELETON_PATH)
    if skel is None:
        raise RuntimeError(
            f"\n[EmpathConcierge] Skeleton not found at: {SKELETON_PATH}\n"
            "  How to find the correct path:\n"
            "  1. Open your MetaHuman Blueprint in UE.\n"
            "  2. Select the 'Face' Skeletal Mesh Component.\n"
            "  3. In Details, click the Skeletal Mesh asset → it opens in Content Browser.\n"
            "  4. Right-click the .uasset → Asset Actions → Copy Reference.\n"
            "  5. Paste the path (without SkeletalMesh' wrapper) into SKELETON_PATH above.\n"
            "  Example: /Game/MetaHumans/Ada/Face/Meshes/m_med_nrw_face_Skeleton"
        )
    unreal.log(f"[EmpathConcierge] Skeleton resolved: {SKELETON_PATH}")
    return skel


# ── Step 5 — Create the AnimSequence asset ───────────────────────────────────

def _create_anim_sequence(skeleton) -> unreal.AnimSequence:
    # Remove pre-existing asset silently to avoid modal dialogs
    if unreal.EditorAssetLibrary.does_asset_exist(OUTPUT_ASSET):
        unreal.log_warning(f"[EmpathConcierge] Deleting pre-existing asset: {OUTPUT_ASSET}")
        unreal.EditorAssetLibrary.delete_asset(OUTPUT_ASSET)

    factory                  = unreal.AnimSequenceFactory()
    factory.target_skeleton  = skeleton

    pkg_path, asset_name = OUTPUT_ASSET.rsplit("/", 1)
    asset_tools          = unreal.AssetToolsHelpers.get_asset_tools()

    anim_seq = asset_tools.create_asset(
        asset_name   = asset_name,
        package_path = pkg_path,
        asset_class  = unreal.AnimSequence,
        factory      = factory,
    )

    if anim_seq is None:
        raise RuntimeError(
            "[EmpathConcierge] AnimSequence asset creation returned None.\n"
            "  Check that the skeleton path is valid and the output package path exists."
        )

    unreal.log(f"[EmpathConcierge] Asset created: {OUTPUT_ASSET}")
    return anim_seq


# ── Step 6 (patched) — multi-strategy curve injection ────────────────────────

def _get_controller(anim_seq):
    """
    UE 5.7.4 Diagnostic Fix: Accessing properties directly as shown in your dir() log.
    """
    try:
        # Based on your diagnostic, these are properties, not functions
        ctrl = anim_seq.controller
        mdl = anim_seq.data_model
        
        if ctrl is not None:
            unreal.log("[EmpathConcierge] Controller obtained via direct property access.")
            return ctrl, mdl
    except Exception as e:
        unreal.log_warning(f"[EmpathConcierge] Property access failed: {e}")
        
    return None, None

def _bake_via_controller(anim_seq, controller, model):
    """
    Final UE 5.7.4 "Resilient" Fix:
    1. Probes for the correct curve identifier property (name, curve_name, or id).
    2. Maintains the FrameNumber struct and non-transacted logic.
    """
    frame_rate = unreal.FrameRate(numerator=FPS, denominator=1)
    
    controller.open_bracket("EmpathConcierge: bake emotion curves")
    
    try:
        controller.set_frame_rate(frame_rate)
        
        # Satisfy the FrameNumber requirement for 5.7.4
        frame_number_struct = unreal.FrameNumber(value=N_FRAMES)
        controller.set_number_of_frames(frame_number_struct)

        with unreal.ScopedSlowTask(N_DIMS, "Baking curves (controller)...") as task:
            task.make_dialog(True)
            
            for dim_idx in range(min(N_DIMS, 260)):
                if task.should_cancel():
                    break
                    
                curve_name_str = INDEX_MAP.get(dim_idx, f"curve_dim_{dim_idx:03d}")
                task.enter_progress_frame(1, f"[{dim_idx+1}/{N_DIMS}] {curve_name_str}")

                fname = unreal.Name(curve_name_str)
                
                # ★ THE RESILIENT FIX: Probe for the correct identifier property ★
                curve_id = unreal.AnimationCurveIdentifier()
                
                # Try every common property name for the identifier
                prop_found = False
                for prop in ['name', 'curve_name', 'identifier', 'id']:
                    if hasattr(curve_id, prop):
                        setattr(curve_id, prop, fname)
                        prop_found = True
                        break
                
                if not prop_found:
                    unreal.log_error(f"[EmpathConcierge] Could not find name property on AnimationCurveIdentifier.")
                    break

                curve_id.curve_type = unreal.RawCurveTrackTypes.RCT_FLOAT

                # Add the curve track to the model
                controller.add_curve(curve_id, unreal.AnimCurveType.RICH_CURVE)

                # Prepare key data
                keys = []
                for frame_i, frame_vals in enumerate(FRAMES):
                    val = float(frame_vals[dim_idx]) if dim_idx < len(frame_vals) else 0.0
                    
                    # Clamp blendshapes (0-250) for the MetaHuman rig
                    if dim_idx < 251:
                        val = max(0.0, min(1.0, val))
                        
                    keys.append(unreal.RichCurveKey(
                        time=frame_i / FPS, 
                        value=val,
                        interp_mode=unreal.RichCurveInterpMode.RCIM_LINEAR
                    ))
                
                # Set all keys for this curve in one call
                controller.set_curve_keys(curve_id, keys)
                
    finally:
        # Finalize the single transaction to update the UI
        controller.close_bracket()
        
    unreal.log(f"[EmpathConcierge] SUCCESS: Baked {N_DIMS} curves × {N_FRAMES} frames.")
def _bake_via_controller(anim_seq, controller, model):
    """
    Final UE 5.7.4 "Encompassing" Fix:
    1. Dynamically probes AnimCurveType for the correct enum (usually ATTRIBUTE).
    2. Handles potential single-argument add_curve() signatures.
    3. Maintains FrameNumber and non-transacted logic for your Omen 16.
    """
    frame_rate = unreal.FrameRate(numerator=FPS, denominator=1)
    
    controller.open_bracket("EmpathConcierge: bake emotion curves")
    
    try:
        controller.set_frame_rate(frame_rate)
        
        # Wrap frame count in the required struct for UE 5.7 nativization
        frame_number_struct = unreal.FrameNumber(value=N_FRAMES)
        controller.set_number_of_frames(frame_number_struct)

        # ★ THE FIX: Resolve the AnimCurveType enum dynamically ★
        curve_type_enum = None
        # Probable candidates in UE 5.7: ATTRIBUTE, VARIABLE, or MORPH_TARGET
        for candidate in ['ATTRIBUTE', 'VARIABLE', 'MORPH_TARGET']:
            if hasattr(unreal.AnimCurveType, candidate):
                curve_type_enum = getattr(unreal.AnimCurveType, candidate)
                break

        with unreal.ScopedSlowTask(N_DIMS, "Baking curves (controller)...") as task:
            task.make_dialog(True)
            
            for dim_idx in range(min(N_DIMS, 260)):
                if task.should_cancel():
                    break
                    
                curve_name_str = INDEX_MAP.get(dim_idx, f"curve_dim_{dim_idx:03d}")
                task.enter_progress_frame(1, f"[{dim_idx+1}/{N_DIMS}] {curve_name_str}")

                fname = unreal.Name(curve_name_str)
                
                # Manual struct initialization confirmed by your dir() diagnostic
                curve_id = unreal.AnimationCurveIdentifier()
                curve_id.set_curve_identifier(fname, unreal.RawCurveTrackTypes.RCT_FLOAT)

                # Add the curve — trying two-arg then one-arg signatures
                try:
                    controller.add_curve(curve_id, curve_type_enum or 0)
                except Exception:
                    try:
                        controller.add_curve(curve_id)
                    except Exception as e:
                        unreal.log_error(f"add_curve failed: {e}")
                        break

                # Process keys for your Joy set (140 frames)
                keys = []
                for frame_i, frame_vals in enumerate(FRAMES):
                    val = float(frame_vals[dim_idx]) if dim_idx < len(frame_vals) else 0.0
                    
                    # Clamp blendshapes (0-250) for MetaHuman rig fidelity
                    if dim_idx < 251:
                        val = max(0.0, min(1.0, val))
                        
                    keys.append(unreal.RichCurveKey(
                        time=frame_i / FPS, 
                        value=val,
                        interp_mode=unreal.RichCurveInterpMode.RCIM_LINEAR
                    ))
                
                # Bulk-set keys
                controller.set_curve_keys(curve_id, keys)
                
    finally:
        # Commit the transaction to avoid editor hang on your Omen 16
        controller.close_bracket()
        
    unreal.log(f"[EmpathConcierge] SUCCESS: Baked {N_DIMS} curves × {N_FRAMES} frames.")
def _bake_curves(anim_seq):
    controller, model = _get_controller(anim_seq)

    if controller is not None:
        unreal.log("[EmpathConcierge] Using IAnimationDataController (fast path).")
        _bake_via_controller(anim_seq, controller, model)
    else:
        unreal.log_warning("[EmpathConcierge] get_controller() unavailable in this UE build.")
        unreal.log("[EmpathConcierge] Falling back to AnimationLibrary (stable path).")
        _bake_via_library(anim_seq)
def _save_and_reveal(anim_seq: unreal.AnimSequence):
    # Ensure the asset is flushed to disk
    unreal.EditorAssetLibrary.save_loaded_asset(anim_seq, only_if_is_dirty=False)
    
    # THE FIX: Get the path string instead of passing the object
    asset_path = anim_seq.get_path_name()
    unreal.EditorAssetLibrary.sync_browser_to_objects([asset_path])
    
    unreal.log(f"[EmpathConcierge] SUCCESS: Saved and revealed → {OUTPUT_ASSET}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    unreal.log("=" * 60)
    unreal.log("  EmpathConcierge — AnimSequence Baker  (UE 5.7.4)")
    unreal.log("  github: syedmuntaquakarim / kaggle: syedmuntaquakarim")
    unreal.log("=" * 60)
    unreal.log(f"  CSV        : {CSV_PATH}")
    unreal.log(f"  Frames     : {N_FRAMES}  ({SEQ_LENGTH_SEC:.2f}s @ {FPS} fps)")
    unreal.log(f"  Curve dims : {N_DIMS}")
    unreal.log(f"  Output     : {OUTPUT_ASSET}")
    unreal.log("=" * 60)

    skeleton = _load_skeleton()
    anim_seq = _create_anim_sequence(skeleton)
    _bake_curves(anim_seq)
    _save_and_reveal(anim_seq)

    unreal.log("[EmpathConcierge] Done. Find the asset in the Content Browser.")
    unreal.log("  Next: drag it onto the MetaHuman Actor → assign to AnimBlueprint.")


main()
