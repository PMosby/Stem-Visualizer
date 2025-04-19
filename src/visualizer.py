"""
visualizer.py - Advanced visualizations for audio stems

This module provides visualization functions that create immersive visuals based on audio stems:
1. Epic nature-inspired visualizations
2. Noir-style visualizations 
3. Dance/movement-inspired visualizations
4. Combined visualization modes

All visualizations are reactive to audio features like amplitude, frequency, rhythm, etc.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageEnhance, ImageFilter
import colorsys
import io
import base64

# Define custom color schemes inspired by nature, noir, and fusion aesthetics
NATURE_COLORS = ["#0D2818", "#1A4B30", "#266D45", "#39A168", "#7AE582"]
NOIR_COLORS = ["#000000", "#222222", "#444444", "#888888", "#FFFFFF"]
FUSION_COLORS = ["#3F0071", "#610094", "#8500BD", "#A100E8", "#BC00FF"]

def create_custom_cmap(name, colors):
    """Create a custom colormap from a list of colors"""
    return LinearSegmentedColormap.from_list(name, colors)

# Create custom colormaps
nature_cmap = create_custom_cmap("nature", NATURE_COLORS)
noir_cmap = create_custom_cmap("noir", NOIR_COLORS)
fusion_cmap = create_custom_cmap("fusion", FUSION_COLORS)

def extract_features(audio_path):
    """Extract audio features needed for visualization"""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract features
    features = {
        "waveform": y,
        "sample_rate": sr,
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
        "rms": librosa.feature.rms(y=y)[0],
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
        "onset_env": librosa.onset.onset_strength(y=y, sr=sr),
        "mel_spec": librosa.feature.melspectrogram(y=y, sr=sr),
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    }
    
    return features

def nature_visualization(stem_features, fig_width=8, fig_height=6):
    """
    Create nature-inspired visualization based on audio features
    
    This visualization uses elements inspired by natural forms:
    - Organic flowing shapes for bass frequencies
    - Leaf/plant-like patterns for middle frequencies
    - Light sparkling effects for higher frequencies
    """
    mel_spec = librosa.power_to_db(stem_features["mel_spec"], ref=np.max)
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create a layered effect with different frequency bands
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap=nature_cmap)
    
    # Add flowing lines based on spectral centroid
    spectral_centroid = stem_features["spectral_centroid"]
    for i in range(3):
        # Smooth the data
        smooth_centroid = gaussian_filter1d(spectral_centroid, sigma=5+i*2)
        # Scale to fit the plot
        scaled_centroid = (smooth_centroid / smooth_centroid.max() * mel_spec.shape[0] * 0.8) + (i * mel_spec.shape[0] * 0.1)
        x = np.linspace(0, mel_spec.shape[1], len(scaled_centroid))
        plt.plot(x, scaled_centroid, linewidth=1.5, alpha=0.7-i*0.2, 
                 color=NATURE_COLORS[-(i+1)], zorder=5)
    
    # Title and styling
    plt.title("Nature-Inspired Visualization")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.tight_layout()
    
    return fig

def noir_visualization(stem_features, fig_width=8, fig_height=6):
    """
    Create noir-style visualization based on audio features
    
    This visualization emphasizes contrast and shadow:
    - High contrast black and white color scheme
    - Sharp geometric forms for rhythm elements
    - Gradual fades for sustained notes
    """
    # Get features for visualization
    mfccs = stem_features["mfcc"]
    onsets = stem_features["onset_env"]
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Plot a stylized MFCC with noir aesthetic
    plt.imshow(mfccs, aspect='auto', origin='lower', cmap=noir_cmap, interpolation='nearest')
    
    # Add rhythm indicators
    onset_times = np.linspace(0, mfccs.shape[1], len(onsets))
    scaled_onsets = (onsets / onsets.max() * mfccs.shape[0] * 0.9)
    
    # Add accent lines on strong beats
    threshold = 0.5 * onsets.max()
    for i, onset in enumerate(onsets):
        if onset > threshold:
            plt.axvline(x=onset_times[i], alpha=onset/onsets.max()*0.8, 
                       color='white', linewidth=1, zorder=3)
    
    # Add smooth curve highlighting rhythm
    plt.plot(onset_times, scaled_onsets, color='white', alpha=0.7, linewidth=1.5, zorder=4)
    
    # Title and styling
    plt.title("Noir Visualization")
    plt.ylabel("Frequency Bands")
    plt.xlabel("Time")
    plt.tight_layout()
    
    return fig

def dance_visualization(stem_features, fig_width=8, fig_height=6):
    """
    Create dance-inspired visualization based on audio features
    
    This visualization emphasizes movement and rhythm:
    - Dynamic shapes that pulse with the beat
    - Flowing lines that follow the melodic contour
    - Color shifts that respond to harmonic changes
    """
    # Get features
    chroma = stem_features["chroma"]
    rms = stem_features["rms"]
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Plot chromagram for harmonic content
    plt.imshow(chroma, aspect='auto', origin='lower', cmap=fusion_cmap, interpolation='bilinear')
    
    # Add amplitude envelope as pulsing element
    rms_times = np.linspace(0, chroma.shape[1], len(rms))
    rms_scaled = (rms / rms.max() * chroma.shape[0] * 0.95)
    
    # Create a "dance" effect with the energy
    for i in range(3):
        offset = i * 0.1 * chroma.shape[0]
        scale_factor = 1.0 - (i * 0.2)
        smoothed_rms = gaussian_filter1d(rms_scaled * scale_factor + offset, sigma=3+i)
        plt.fill_between(rms_times, offset, smoothed_rms, 
                        alpha=0.4-i*0.1, color=FUSION_COLORS[i+1], zorder=i+3)
    
    # Add "movement" lines
    for i in range(12):  # 12 chroma bins
        if i % 2 == 0:  # Only use some bins for cleaner visual
            chroma_line = chroma[i]
            chroma_line = gaussian_filter1d(chroma_line, sigma=2)
            chroma_times = np.linspace(0, chroma.shape[1], len(chroma_line))
            chroma_scaled = chroma_line / chroma_line.max() * chroma.shape[0] * 0.8
            plt.plot(chroma_times, chroma_scaled, linewidth=1, alpha=0.6, 
                    color=FUSION_COLORS[min(4, (i//2)+1)], zorder=10)
    
    # Title and styling
    plt.title("Dance Visualization")
    plt.ylabel("Pitch Class")
    plt.xlabel("Time")
    plt.tight_layout()
    
    return fig

def epic_combined_visualization(stem_features_dict, fig_width=12, fig_height=8):
    """
    Create epic combined visualization using all stems
    
    This visualization creates a complex, layered visualization that:
    - Uses different visual elements for each stem
    - Combines all stems into a cohesive visual experience
    - Creates an epic, immersive representation of the full song
    """
    # Create figure with subplots (one main plot and small plots for each stem)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Set up grid layout
    gs = fig.add_gridspec(3, 4)
    
    # Main combined visualization (spans most of the figure)
    ax_main = fig.add_subplot(gs[:2, :])
    
    # Create combined mel spectrogram
    combined_spec = None
    
    # Add each stem with a different color and effect
    colors = {
        'vocals': '#FF5F5F',  # Red for vocals
        'drums': '#5F5FFF',   # Blue for drums
        'bass': '#5FFF5F',    # Green for bass
        'other': '#FFFF5F'    # Yellow for other
    }
    
    # Process each stem and layer them
    for stem_name, features in stem_features_dict.items():
        if stem_name != 'original':
            # Get mel spectrogram for this stem
            spec = librosa.power_to_db(features["mel_spec"], ref=np.max)
            
            # Display each stem's spectrogram with its color
            ax_main.imshow(spec, aspect='auto', origin='lower', alpha=0.5,
                         cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                             f"{stem_name}_cmap", ['#000000', colors.get(stem_name, '#FFFFFF')]))
    
    # Add epic title
    ax_main.set_title("MYCOTA - \"I am a General\" - Epic Visualization", fontsize=15)
    ax_main.set_ylabel("Frequency")
    ax_main.set_xlabel("Time")
    
    # Individual stem visualizations (smaller plots at bottom)
    stem_names = list(stem_features_dict.keys())
    for i, stem_name in enumerate(stem_names):
        if i < 4 and stem_name != 'original':  # Display up to 4 individual stems
            # Add small subplot
            ax = fig.add_subplot(gs[2, i])
            
            # Get features for this stem
            features = stem_features_dict[stem_name]
            
            # Choose visualization based on stem type
            if stem_name == 'vocals':
                # Show vocals with chroma features
                chroma = features["chroma"]
                ax.imshow(chroma, aspect='auto', origin='lower', 
                        cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                            "vocals_cmap", ['#000000', '#FF5F5F']))
                ax.set_title(f"{stem_name.capitalize()}")
                
            elif stem_name == 'drums':
                # Show drums with onset strength
                onsets = features["onset_env"]
                onset_times = np.arange(len(onsets))
                ax.plot(onset_times, onsets, color='#5F5FFF', linewidth=2)
                ax.fill_between(onset_times, 0, onsets, color='#5F5FFF', alpha=0.5)
                ax.set_title(f"{stem_name.capitalize()}")
                
            elif stem_name == 'bass':
                # Show bass with lower frequencies of mel spectrogram
                spec = librosa.power_to_db(features["mel_spec"], ref=np.max)
                ax.imshow(spec[:spec.shape[0]//4], aspect='auto', origin='lower',
                        cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                            "bass_cmap", ['#000000', '#5FFF5F']))
                ax.set_title(f"{stem_name.capitalize()}")
                
            else:  # 'other' or any additional stems
                # Show other with MFCCs
                mfccs = features["mfcc"]
                ax.imshow(mfccs, aspect='auto', origin='lower',
                        cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                            "other_cmap", ['#000000', '#FFFF5F']))
                ax.set_title(f"{stem_name.capitalize()}")
    
    plt.tight_layout()
    return fig

def create_animated_gif(stem_features_dict, duration=10, fps=10):
    """
    Create an animated visualization from stem features
    Returns a base64 encoded gif image
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Number of frames
    frames = fps * duration
    
    # Create a list to store frames
    frame_data = []
    
    # Get time points
    if 'original' in stem_features_dict:
        features = stem_features_dict['original']
        total_samples = len(features['waveform'])
        sample_rate = features['sample_rate']
        total_duration = total_samples / sample_rate
        time_points = np.linspace(0, total_duration, frames)
    
    # Function to draw one frame
    def draw_frame(frame_idx):
        ax.clear()
        
        # Time position for this frame
        t = time_points[frame_idx]
        sample_idx = int(t * sample_rate)
        
        # Combine elements from each stem for this time point
        for stem_name, features in stem_features_dict.items():
            if stem_name == 'original':
                continue
                
            # Get a slice of the mel spectrogram around this time
            mel_spec = features['mel_spec']
            spec_idx = min(int(t * mel_spec.shape[1] / total_duration), mel_spec.shape[1]-1)
            
            # Get the column of the mel spectrogram at this time
            spec_slice = mel_spec[:, max(0, spec_idx-5):min(mel_spec.shape[1], spec_idx+5)]
            
            # Average across this time window
            if spec_slice.shape[1] > 0:
                spec_col = np.mean(spec_slice, axis=1)
                
                # Scale frequencies logarithmically for more natural visual
                freqs = librosa.mel_frequencies(n_mels=len(spec_col), fmin=0, fmax=sample_rate/2)
                
                # Normalize for visualization
                spec_col = spec_col / np.max(spec_col) if np.max(spec_col) > 0 else spec_col
                
                # Draw differently based on stem type
                if stem_name == 'vocals':
                    # Vocal visualization
                    ax.plot(freqs, spec_col, color='#FF5F5F', alpha=0.7, linewidth=2)
                    
                elif stem_name == 'drums':
                    # Drums visualization - use circular markers for percussion
                    mask = spec_col > 0.5  # Only show strong drum hits
                    ax.scatter(freqs[mask], spec_col[mask], color='#5F5FFF', 
                              s=spec_col[mask]*100, alpha=0.7)
                    
                elif stem_name == 'bass':
                    # Bass visualization - show as filled area for lower frequencies
                    ax.fill_between(freqs, 0, spec_col, color='#5FFF5F', alpha=0.5)
                    
                else:  # 'other'
                    # Other instruments
                    ax.plot(freqs, spec_col, color='#FFFF5F', alpha=0.4, linewidth=1.5)
        
        # Add a time marker
        ax.axvline(x=t*1000, color='white', alpha=0.3, linestyle='--')
        
        # Set title and styling
        ax.set_title(f"MYCOTA - \"I am a General\" - t={t:.2f}s", fontsize=12)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(20, sample_rate/2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('black')
        
        # Capture this frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_data.append(image)
    
    # Generate all frames
    for i in range(frames):
        draw_frame(i)
    
    # Close the figure
    plt.close(fig)
    
    # Create animated GIF
    output = io.BytesIO()
    
    # Convert frames to PIL images and save as GIF
    images = [Image.fromarray(frame) for frame in frame_data]
    
    # Save as GIF
    images[0].save(
        output, 
        format='GIF',
        save_all=True,
        append_images=images[1:],
        duration=1000/fps,  # Duration of each frame in milliseconds
        loop=0  # Loop forever
    )
    
    # Return the base64 encoded GIF
    output.seek(0)
    return base64.b64encode(output.read()).decode('utf-8')

def generate_plotly_visualizations(stem_features_dict):
    """
    Create interactive Plotly visualizations
    Returns a plotly figure that is interactive
    """
    # Create figure with multiple traces for each stem
    fig = go.Figure()
    
    # Track maximum values for normalization
    max_values = {}
    
    # Process each stem
    for stem_name, features in stem_features_dict.items():
        # Skip the original for this visualization
        if stem_name == 'original':
            continue
            
        # Get mel spectrogram
        mel_spec = librosa.power_to_db(features['mel_spec'], ref=np.max)
        
        # Store maximum value
        max_values[stem_name] = np.max(mel_spec)
        
        # Create trace for this stem
        color_map = {
            'vocals': px.colors.sequential.Reds,
            'drums': px.colors.sequential.Blues,
            'bass': px.colors.sequential.Greens,
            'other': px.colors.sequential.Purples
        }
        
        colorscale = color_map.get(stem_name, px.colors.sequential.Viridis)
        
        # Add heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=mel_spec,
                colorscale=colorscale,
                opacity=0.7,
                name=stem_name.capitalize(),
                visible=(stem_name == 'vocals')  # Initially only show vocals
            )
        )
    
    # Add buttons to toggle which stems are visible
    buttons = []
    
    # Button for each individual stem
    for i, stem_name in enumerate(stem_features_dict.keys()):
        if stem_name == 'original':
            continue
            
        visibility = [False] * len(fig.data)
        visibility[i-1] = True  # -1 adjustment because we skipped 'original'
        
        buttons.append(
            dict(
                method="update",
                label=stem_name.capitalize(),
                args=[{"visible": visibility}]
            )
        )
    
    # Button for all stems
    all_visible = [True] * len(fig.data)
    buttons.append(
        dict(
            method="update",
            label="All Stems",
            args=[{"visible": all_visible}]
        )
    )
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.1,
                y=1.15,
                buttons=buttons
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        title="MYCOTA - I am a General - Interactive Visualization",
        xaxis_title="Time",
        yaxis_title="Frequency",
        xaxis=dict(
            title_font=dict(size=14),
            title_standoff=15
        ),
        yaxis=dict(
            title_font=dict(size=14),
            title_standoff=15
        ),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="#FFFFFF")
    )
    
    return fig

def save_visualization_to_html(fig, output_path):
    """Save Plotly visualization to HTML file"""
    fig.write_html(output_path)
    return output_path 