import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def generate_master_plots():
    # Define our emotions and the color palette for the report
    emotions = ['joy', 'anger', 'grief', 'amazement', 'neutral']
    colors = {
        'joy': '#2ca02c',       # Green
        'anger': '#d62728',     # Red
        'grief': '#1f77b4',     # Blue
        'amazement': '#ff7f0e', # Orange
        'neutral': '#7f7f7f'    # Grey
    }
    
    data = {}
    means = {}
    min_len = float('inf')
    
    print("Loading all 5 emotion CSVs...")
    for emo in emotions:
        file_path = f'{emo}_anim.csv'
        if not os.path.exists(file_path):
            print(f"Error: Could not find {file_path}. Did you run inference for all emotions?")
            return
            
        df = pd.read_csv(file_path, header=None).values
        data[emo] = df
        min_len = min(min_len, len(df))
        
    # Truncate to the same length and calculate the mean frame for each emotion
    for emo in emotions:
        data[emo] = data[emo][:min_len]
        means[emo] = np.mean(data[emo], axis=0) # Shape: (260,)
        
    # ==========================================
    # PLOT 1: BAR CHART (Deviation from Neutral)
    # ==========================================
    print("Generating Mean Deviation Bar Chart...")
    target_emotions = ['joy', 'anger', 'grief', 'amazement']
    deviations = []
    
    for emo in target_emotions:
        # Calculate the absolute mean shift across all 260 blendshapes compared to Neutral
        mae = np.mean(np.abs(means[emo] - means['neutral']))
        deviations.append(mae)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar([e.capitalize() for e in target_emotions], deviations, 
                  color=[colors[e] for e in target_emotions])
    
    # Add the exact numerical values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), 
                ha='center', va='bottom', fontweight='bold', size=11)
        
    plt.title('Mean Deviation from Neutral Baseline', size=15, fontweight='bold', pad=15)
    plt.ylabel('Average Absolute Shift', size=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Increase y-axis limit slightly to make room for text
    plt.ylim(0, max(deviations) * 1.15)
    
    bar_out = 'all_emotion_vs_neutral_bar.png'
    plt.savefig(bar_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {bar_out}")

    # ==========================================
    # PLOT 2: RADAR CHART (All Emotions Overlaid)
    # ==========================================
    print("Generating Multi-Emotion Radar Chart...")
    
    # To find the most interesting axes for the radar chart, we find the top 5 
    # blendshapes that have the highest variance across all 5 emotions.
    all_means_matrix = np.array([means[e] for e in emotions]) # Shape: (5, 260)
    variances = np.var(all_means_matrix, axis=0)
    top_5_indices = np.argsort(variances)[-5:][::-1]
    
    labels = [f"Index {idx}" for idx in top_5_indices]
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1] # Close the circle
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels, color='black', size=12, fontweight='bold')
    
    # Plot each emotion onto the radar
    for emo in emotions:
        emo_values = means[emo][top_5_indices]
        emo_radar = np.append(emo_values, emo_values[0])
        
        # Thicker lines for visibility
        ax.plot(angles, emo_radar, linewidth=2.5, linestyle='solid', label=emo.capitalize(), color=colors[emo])
        # Very light fill so it doesn't become a messy blob of colors
        ax.fill(angles, emo_radar, colors[emo], alpha=0.1) 
        
    plt.title('Emotion Blendshape Activation (Real Audio Inference)', size=16, fontweight='bold', y=1.1)
    
    # Move legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=12)
    
    radar_out = 'real_audio_all_emotions_radar.png'
    plt.savefig(radar_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {radar_out}")
    print("\nAll visualizations complete! Ready for your report.")

if __name__ == '__main__':
    generate_master_plots()