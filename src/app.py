"""
app.py - Streamlit frontend for audio stem separation and visualization.

This app allows users to:
1. Upload audio files
2. Process them into stems using Demucs
3. Play the original and separated stems
4. Visualize waveforms for each stem
5. Generate 3D visualizations with audio reactivity
"""

import os
import tempfile
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import json
import base64
import hashlib  # For creating cache keys
import shutil   # For file operations
from pathlib import Path
from separation import separate_audio, mix_stems

# Configure page
st.set_page_config(
    page_title="Audio Stem Separator",
    page_icon="🎵",
    layout="wide"
)

# Add a cache directory for development
DEV_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "stems")
os.makedirs(DEV_CACHE_DIR, exist_ok=True)

# Caching functions
@st.cache_data
def cached_separate_audio(input_path, output_dir, model_name, device):
    """Cache the stem separation to avoid reprocessing during development"""
    return separate_audio(input_path, output_dir, model_name, device)

def get_cached_stems(input_path, model_name):
    """
    Try to get cached stems for the given input file and model.
    
    Returns:
        Dictionary of stem paths if cached, None otherwise
    """
    # Create a unique key based on the input file and model
    file_name = os.path.basename(input_path)
    file_size = os.path.getsize(input_path)
    
    # Simple cache key based on filename, size and model
    cache_key = f"{file_name}_{file_size}_{model_name}"
    cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # Check if we have this in cache
    cache_dir = os.path.join(DEV_CACHE_DIR, cache_key_hash)
    
    if os.path.exists(cache_dir):
        st.info("Using cached stems for faster development")
        
        # Build dictionary of stem paths
        stem_paths = {}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            stem_path = os.path.join(cache_dir, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                stem_paths[stem_name] = stem_path
        
        if len(stem_paths) > 0:
            return stem_paths
    
    return None

def save_to_cache(input_path, model_name, stem_paths):
    """Save stems to cache for faster development"""
    # Create a unique key based on the input file and model
    file_name = os.path.basename(input_path)
    file_size = os.path.getsize(input_path)
    
    # Simple cache key based on filename, size and model
    cache_key = f"{file_name}_{file_size}_{model_name}"
    cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # Create cache directory
    cache_dir = os.path.join(DEV_CACHE_DIR, cache_key_hash)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Copy stems to cache
    for stem_name, stem_path in stem_paths.items():
        cache_path = os.path.join(cache_dir, f"{stem_name}.wav")
        shutil.copy2(stem_path, cache_path)
    
    st.success(f"Stems cached for faster development")

# Define functions for visualization
def plot_waveform(audio_path, title, color="#1f77b4"):
    """Plot waveform for audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return fig
    except Exception as e:
        st.error(f"Error plotting waveform: {e}")
        # Create empty plot as fallback
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_title(f"{title} (Failed to load)")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig

def create_custom_mix(stem_paths, selected_stems):
    """Create a custom mix from selected stems"""
    if not selected_stems:
        return None
    
    # Create a temporary file for the mix
    try:
        _, temp_file = tempfile.mkstemp(suffix=".wav")
        
        # Use our existing mix_stems function
        return mix_stems(stem_paths, temp_file, selected_stems)
    except Exception as e:
        st.error(f"Error creating custom mix: {e}")
        return None

def create_3d_visualization(stem_paths):
    """
    Create a 3D visualization of audio stems using Three.js
    
    Args:
        stem_paths: Dictionary mapping stem names to file paths
    """
    # Get the base URL for Streamlit
    # For local development this will typically be http://localhost:8501
    if not stem_paths:
        return
    
    # Create URLs for each stem that can be accessed by JavaScript
    stem_urls = {}
    
    for stem_name, path in stem_paths.items():
        try:
            # Create a unique filename for each stem
            filename = f"{stem_name}.wav"
            # Read the file and encode it
            with open(path, "rb") as file:
                audio_bytes = file.read()
            
            # Create a data URL for the audio file
            audio_b64 = base64.b64encode(audio_bytes).decode()
            stem_urls[stem_name] = f"data:audio/wav;base64,{audio_b64}"
        except Exception as e:
            st.error(f"Error creating data URL for {stem_name}: {e}")
    
    # Read the HTML template
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    
    try:
        with open(html_path, "r") as f:
            html_content = f.read()
        
        # Replace the placeholder with actual stem paths
        stem_paths_json = json.dumps(stem_urls)
        html_content = html_content.replace("const stemPaths = {};", f"const stemPaths = {stem_paths_json};")
        
        # Embed the CSS directly
        css_path = os.path.join(os.path.dirname(__file__), "web", "css", "style.css")
        with open(css_path, "r") as f:
            css_content = f.read()
        
        html_content = html_content.replace('<link rel="stylesheet" href="css/style.css">', f'<style>{css_content}</style>')
        
        # Embed the JS directly
        js_path = os.path.join(os.path.dirname(__file__), "web", "js", "visualizer.js")
        with open(js_path, "r") as f:
            js_content = f.read()
        
        html_content = html_content.replace('<script src="js/visualizer.js"></script>', f'<script>{js_content}</script>')
        
        # Display using st.components.html
        st.components.v1.html(html_content, height=700)
    except Exception as e:
        st.error(f"Error creating 3D visualization: {e}")

def show_model_explanation():
    """Display information about the different models"""
    with st.expander("About the Separation Models"):
        st.write("""
        Demucs offers several model variants with different trade-offs:
        
        - **Hybrid Transformer Demucs**: Default model with good overall quality.
        - **Hybrid Transformer Demucs (fine-tuned)**: Further improved quality but slightly larger.
        - **MDX-Extra**: Specialized for music source separation with strong vocal isolation.
        - **MDX-Extra Quantized**: Smaller memory footprint version of MDX-Extra, useful for systems with limited RAM.
        
        The processing time will depend on your CPU capabilities and the audio length. Separation typically takes 
        20-30 seconds per minute of audio on a standard CPU.
        """)

# Add file format info
def show_format_info():
    with st.expander("Supported File Formats"):
        st.write("""
        The app supports the following audio formats:
        - WAV (recommended for highest quality)
        - MP3
        - FLAC
        - OGG
        - M4A/AAC (experimental support)
        
        For best results, prefer uncompressed (WAV) or high-bitrate files.
        If you encounter format issues, try converting your file to WAV using another tool first.
        """)

# Streamlit app
def main():
    st.title("Audio Stem Separator & Visualizer")
    st.write("Upload an audio file to separate it into stems and visualize the waveforms.")
    
    # Add model explanations
    show_model_explanation()
    show_format_info()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])
    
    if uploaded_file is not None:
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, uploaded_file.name)
        output_dir = os.path.join(temp_dir, "stems")
        
        # Save uploaded file to disk
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display file info
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        st.info(f"Uploaded file: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Try to display audio player
        try:
            st.audio(input_path, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
        except Exception as e:
            st.warning(f"Could not preview audio: {e} - Processing will still be attempted.")
        
        # Select model
        model_options = {
            "Hybrid Transformer Demucs": "htdemucs",
            "Hybrid Transformer Demucs (fine-tuned)": "htdemucs_ft",
            "MDX-Extra (faster)": "mdx_extra",
            "MDX-Extra Quantized (smaller memory)": "mdx_extra_q"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            model_name = st.selectbox(
                "Select separation model",
                list(model_options.keys()),
                index=0
            )
        
        with col2:
            use_gpu = st.checkbox("Use GPU (if available)", 
                                value=torch.cuda.is_available(),
                                disabled=not torch.cuda.is_available())
            if not torch.cuda.is_available() and use_gpu:
                st.info("GPU not detected, using CPU instead.")
                use_gpu = False
        
        # Process button
        device = "cuda" if use_gpu else "cpu"
        
        # Estimate processing time based on file size
        estimated_time = file_size_mb * 2  # rough estimate: 2 seconds per MB
        st.info(f"Estimated processing time: {estimated_time:.0f} seconds (may vary based on your system)")
        
        # Add "Use Cached Stems" option for development
        use_cached = st.checkbox("Use cached stems if available (for development)", value=True)
        
        if st.button("Separate Stems"):
            # Try to get cached stems first if enabled
            cached_stems = None
            if use_cached:
                cached_stems = get_cached_stems(input_path, model_options[model_name])
            
            if cached_stems:
                # Use cached stems
                stem_paths = cached_stems
                st.session_state["stem_paths"] = stem_paths
                st.success(f"Using cached stems for faster development!")
                st.experimental_rerun()
            else:
                # No cached stems, do the normal processing
                # Create a progress bar
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                
                try:
                    # Track start time
                    start_time = time.time()
                    
                    # Update status
                    status_text.text("Loading model...")
                    progress_bar.progress(10, text="Loading Demucs model...")
                    
                    # Use our cached separate_audio function
                    status_text.text("Processing audio...")
                    
                    # Process in stages to update progress
                    stem_paths = cached_separate_audio(
                        input_path, 
                        output_dir, 
                        model_options[model_name], 
                        device
                    )
                    
                    # Save to cache for future use
                    save_to_cache(input_path, model_options[model_name], stem_paths)
                    
                    # Update with completion
                    elapsed_time = time.time() - start_time
                    progress_bar.progress(100, text="Processing complete!")
                    status_text.text(f"Completed in {elapsed_time:.1f} seconds.")
                    
                    if stem_paths:
                        st.session_state["stem_paths"] = stem_paths
                        st.success(f"Stems separated successfully in {elapsed_time:.1f} seconds!")
                        st.experimental_rerun()
                    else:
                        st.error("Separation failed. Check the logs for more information.")
                except Exception as e:
                    st.error(f"Error during separation: {str(e)}")
                    st.info("Try a different model or file format if the issue persists.")
                    progress_bar.empty()
        
        # Display stems if available
        if "stem_paths" in st.session_state:
            stem_paths = st.session_state["stem_paths"]
            if stem_paths:  # Check if we have valid stem paths
                st.subheader("Separated Stems")
                
                # Create tabs for original audio, stems, and 3D visualization
                tabs = ["Original", "Stems", "3D Visualization"]
                selected_tab = st.tabs(tabs)
                
                # Original tab
                with selected_tab[0]:
                    st.write("Original audio file")
                    try:
                        st.audio(input_path, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
                    except Exception as e:
                        st.warning(f"Could not play original audio: {e}")
                    
                    st.write("Waveform:")
                    st.pyplot(plot_waveform(input_path, "Original Audio"))
                
                # Stems tab
                with selected_tab[1]:
                    # Create sub-tabs for each stem
                    stem_tabs = list(stem_paths.keys())
                    stem_selected_tab = st.tabs(stem_tabs)
                    
                    # Display each stem in its tab
                    for i, stem_name in enumerate(stem_paths.keys()):
                        with stem_selected_tab[i]:
                            st.write(f"{stem_name.capitalize()} audio")
                            try:
                                st.audio(stem_paths[stem_name], format="audio/wav")
                            except Exception as e:
                                st.warning(f"Could not play {stem_name} stem: {e}")
                            
                            st.write("Waveform:")
                            
                            # Use different colors for different stems
                            colors = {
                                "vocals": "#ff9900",
                                "drums": "#ff0000",
                                "bass": "#0000ff",
                                "other": "#00cc00"
                            }
                            color = colors.get(stem_name, "#1f77b4")
                            
                            st.pyplot(plot_waveform(stem_paths[stem_name], f"{stem_name.capitalize()} Waveform", color))
                
                # 3D Visualization tab
                with selected_tab[2]:
                    st.write("3D Audio Visualization")
                    st.write("Interact with the visualization below. Use the play button to start/stop the audio and visualization.")
                    
                    # Create the 3D visualization
                    create_3d_visualization(stem_paths)
                
                # Custom mix section
                st.subheader("Create Custom Mix")
                st.write("Select stems to include in your custom mix:")
                
                # Checkboxes for each stem
                selected_stems = []
                cols = st.columns(len(stem_paths))
                for i, (stem_name, path) in enumerate(stem_paths.items()):
                    with cols[i]:
                        if st.checkbox(stem_name.capitalize(), value=True, key=f"include_{stem_name}"):
                            selected_stems.append(stem_name)
                
                if st.button("Create Mix", key="create_mix_button"):
                    if selected_stems:
                        with st.spinner("Creating custom mix..."):
                            # Create temporary file for mix
                            mix_file = create_custom_mix(stem_paths, selected_stems)
                            if mix_file:
                                st.session_state["mix_file"] = mix_file
                                st.success("Custom mix created!")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to create mix. Check the logs for details.")
                    else:
                        st.error("Please select at least one stem for mixing")
                
                # Display custom mix if available
                if "mix_file" in st.session_state and os.path.exists(st.session_state["mix_file"]):
                    st.subheader("Custom Mix")
                    try:
                        st.audio(st.session_state["mix_file"], format="audio/wav")
                    except Exception as e:
                        st.warning(f"Could not play custom mix: {e}")
                    
                    st.write("Custom Mix Waveform:")
                    st.pyplot(plot_waveform(st.session_state["mix_file"], "Custom Mix", "#9467bd"))
                    
                    # Add download button for the mix
                    with open(st.session_state["mix_file"], "rb") as f:
                        st.download_button(
                            label="Download Custom Mix",
                            data=f,
                            file_name="custom_mix.wav",
                            mime="audio/wav"
                        )

if __name__ == "__main__":
    main() 