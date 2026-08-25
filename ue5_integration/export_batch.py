import unreal
import csv
import os

# 1. CHANGE THIS to the Windows folder where your MP4s live
output_dir = r"D:\New Dataset\Test"

# 2. CHANGE THIS to the Unreal Engine folder holding your Animations
ue_folder_path = "//Game/Animations/Test" 

unreal.log(f"Scanning Unreal folder: {ue_folder_path}")

asset_paths = unreal.EditorAssetLibrary.list_assets(ue_folder_path)
exported_count = 0

for path in asset_paths:
    asset = unreal.EditorAssetLibrary.load_asset(path)
    
    # Ensure it's an Animation Sequence
    if not isinstance(asset, unreal.AnimSequence):
        continue

    # --- FIX 1: Clean the name to match the MP4 exactly ---
    raw_name = asset.get_name()
    clean_name = raw_name
    if clean_name.startswith("AS_"):
        clean_name = clean_name[3:]
    if clean_name.endswith("_1"):
        clean_name = clean_name[:-2]
        
    csv_path = os.path.join(output_dir, f"{clean_name}.csv")

    # --- FIX 2: Bypass broken UE5 API by explicitly declaring 60 FPS ---
    fps = 60.0
    total_time = asset.get_play_length()
    total_frames = int(total_time * fps)

    curve_names = unreal.AnimationLibrary.get_animation_curve_names(asset, unreal.RawCurveTrackTypes.RCT_FLOAT)
    
    if not curve_names:
        continue

    # --- FIX 3: Add Time_sec to match blendshape_map.py ---
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Frame", "Time_sec"] + [str(name) for name in curve_names]
        writer.writerow(header)

        for frame in range(total_frames):
            time = frame / fps
            row = [frame, time]
            for name in curve_names:
                # --- FIX 4: Updated to UE5 get_float_value_at_time ---
                value = unreal.AnimationLibrary.get_float_value_at_time(asset, name, time)
                row.append(value)
            writer.writerow(row)

    unreal.log(f"Successfully exported: {clean_name}.csv")
    exported_count += 1

unreal.log(f"Batch Export Complete! {exported_count} CSV files generated in {output_dir}.")