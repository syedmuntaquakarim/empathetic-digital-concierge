import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def generate_plots(anger_csv, amaze_csv):
    print("Loading animation data...")
    anger = pd.read_csv(anger_csv, header=None).values
    amaze = pd.read_csv(amaze_csv, header=None).values
    
    min_len = min(len(anger), len(amaze))
    anger = anger[:min_len]
    amaze = amaze[:min_len]
    
    anger_mean = np.mean(anger, axis=0)
    amaze_mean = np.mean(amaze, axis=0)
    
    # Find the top 5 most divergent blendshapes
    diffs = np.abs(anger_mean - amaze_mean)
    top_5_indices = np.argsort(diffs)[-5:][::-1]
    
    labels = [f"Index {idx}" for idx in top_5_indices]
    anger_values = anger_mean[top_5_indices]
    amaze_values = amaze_mean[top_5_indices]
    diff_values = diffs[top_5_indices]

    # ==========================================
    # PLOT 1: RADAR CHART (Anger vs. Amazement)
    # ==========================================
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    anger_radar = np.append(anger_values, anger_values[0])
    amaze_radar = np.append(amaze_values, amaze_values[0])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels, color='black', size=12, fontweight='bold')
    
    ax.plot(angles, anger_radar, linewidth=2, linestyle='solid', label='Anger', color='#d62728')
    ax.fill(angles, anger_radar, '#d62728', alpha=0.25)
    
    ax.plot(angles, amaze_radar, linewidth=2, linestyle='solid', label='Amazement', color='#ff7f0e')
    ax.fill(angles, amaze_radar, '#ff7f0e', alpha=0.25)
    
    plt.title('Emotion Blendshape Activation (Anger vs Amazement)', size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig('anger_vs_amazement_radar.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Radar Chart: anger_vs_amazement_radar.png")
    plt.close()

    # ==========================================
    # PLOT 2: BAR CHART (Absolute Differences)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, diff_values, color=['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'])
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontweight='bold')

    plt.title('Top 5 Most Divergent Facial Muscles (Anger vs Amazement)', size=15, fontweight='bold')
    plt.ylabel('Average Absolute Shift (Magnitude)', size=12, fontweight='bold')
    plt.xlabel('Blendshape Index', size=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('anger_vs_amazement_bar.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Bar Chart: anger_vs_amazement_bar.png")
    plt.close()

if __name__ == '__main__':
    generate_plots('anger_anim.csv', 'amazement_anim.csv')