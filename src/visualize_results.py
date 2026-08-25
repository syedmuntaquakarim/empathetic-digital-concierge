import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def generate_plots(joy_csv, grief_csv):
    print("Loading animation data...")
    joy = pd.read_csv(joy_csv, header=None).values
    grief = pd.read_csv(grief_csv, header=None).values
    
    # Ensure they match in length
    min_len = min(len(joy), len(grief))
    joy = joy[:min_len]
    grief = grief[:min_len]
    
    # Calculate means across all frames for each blendshape
    joy_mean = np.mean(joy, axis=0)
    grief_mean = np.mean(grief, axis=0)
    
    # Find the top 5 most divergent blendshapes
    diffs = np.abs(joy_mean - grief_mean)
    top_5_indices = np.argsort(diffs)[-5:][::-1]
    
    labels = [f"Index {idx}" for idx in top_5_indices]
    joy_values = joy_mean[top_5_indices]
    grief_values = grief_mean[top_5_indices]
    diff_values = diffs[top_5_indices]

    # ==========================================
    # PLOT 1: RADAR CHART (Joy vs. Grief)
    # ==========================================
    # Number of variables
    N = len(labels)
    
    # What will be the angle of each axis in the plot?
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Close the loop for values as well
    joy_radar = np.append(joy_values, joy_values[0])
    grief_radar = np.append(grief_values, grief_values[0])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], labels, color='black', size=12, fontweight='bold')
    
    # Plot Joy
    ax.plot(angles, joy_radar, linewidth=2, linestyle='solid', label='Joy', color='#2ca02c')
    ax.fill(angles, joy_radar, '#2ca02c', alpha=0.25)
    
    # Plot Grief
    ax.plot(angles, grief_radar, linewidth=2, linestyle='solid', label='Grief', color='#1f77b4')
    ax.fill(angles, grief_radar, '#1f77b4', alpha=0.25)
    
    plt.title('Emotion Blendshape Activation (Joy vs Grief)', size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    radar_out = 'emotion_radar_v3.png'
    plt.savefig(radar_out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Radar Chart: {radar_out}")
    plt.close()

    # ==========================================
    # PLOT 2: BAR CHART (Absolute Differences)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(labels, diff_values, color=['#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontweight='bold')

    plt.title('Top 5 Most Divergent Facial Muscles (Joy vs Grief)', size=15, fontweight='bold')
    plt.ylabel('Average Absolute Shift (Magnitude)', size=12, fontweight='bold')
    plt.xlabel('Blendshape Index', size=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    bar_out = 'top_blendshapes_bar_v3.png'
    plt.savefig(bar_out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Bar Chart: {bar_out}")
    plt.close()

if __name__ == '__main__':
    generate_plots('joy_anim.csv', 'grief_anim.csv')