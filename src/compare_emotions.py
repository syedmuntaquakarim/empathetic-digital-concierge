import numpy as np
import pandas as pd

def compare_csvs(file1, file2):
    print(f"Loading {file1} and {file2}...")
    
    # Load the data (assuming no header row, based on our inference script)
    joy = pd.read_csv(file1, header=None).values
    grief = pd.read_csv(file2, header=None).values
    
    # Ensure they match in length
    min_len = min(len(joy), len(grief))
    joy = joy[:min_len]
    grief = grief[:min_len]
    
    # 1. OVERALL DIFFERENCE (Mean Absolute Error)
    mae = np.mean(np.abs(joy - grief))
    
    # 2. MAXIMUM DIFFERENCE (The single biggest muscle shift)
    max_diff = np.max(np.abs(joy - grief))
    
    # 3. TOP 5 DIVERGING BLENDSHAPES
    # Calculate the mean difference for each of the 260 columns
    col_diffs = np.mean(np.abs(joy - grief), axis=0)
    # Get the column indices with the highest differences
    top_5_indices = np.argsort(col_diffs)[-5:][::-1]
    top_5_values = col_diffs[top_5_indices]
    
    # 4. KINETIC ENERGY (Frame-to-frame movement)
    # Calculate how much the face moves from one frame to the next
    joy_energy = np.mean(np.abs(np.diff(joy, axis=0)))
    grief_energy = np.mean(np.abs(np.diff(grief, axis=0)))
    
    print("\n" + "="*50)
    print(" 📊 EMOTION SEPARATION REPORT: JOY vs GRIEF")
    print("="*50)
    print(f"Total Frames Analyzed : {min_len}")
    print(f"Overall Mean Shift    : {mae:.4f} (Average distance across all muscles)")
    print(f"Maximum Muscle Shift  : {max_diff:.4f} (The single largest gap)")
    print("-" * 50)
    
    print("Top 5 Most Changed Blendshape Indices:")
    for rank, (idx, val) in enumerate(zip(top_5_indices, top_5_values), 1):
        print(f"  {rank}. Index {idx:>3} shifted by an average of {val:.4f}")
    
    print("-" * 50)
    print("Kinetic Energy (Average Frame-to-Frame Velocity):")
    print(f"  Joy   Energy : {joy_energy:.5f}")
    print(f"  Grief Energy : {grief_energy:.5f}")
    
    if grief_energy < joy_energy:
        print("\n💡 Logic Check Passed: Grief is physically slower/stiller than Joy.")
    else:
        print("\n💡 Note: Grief is surprisingly moving more than Joy in this audio clip.")
    print("="*50)

if __name__ == '__main__':
    compare_csvs('joy_anim.csv', 'grief_anim.csv')