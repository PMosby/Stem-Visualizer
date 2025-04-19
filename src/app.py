"""
app.py - Streamlit frontend for audio stem separation and visualization.

This app allows users to:
1. Upload audio files
2. Process them into stems using Demucs
3. Play the original and separated stems
4. Visualize waveforms for each stem
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
from pathlib import Path
from separation import separate_audio, mix_stems

# Configure page
st.set_page_config(
    page_title="Audio Stem Separator",
    page_icon="🎵",
    layout="wide"
)

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
        
        if st.button("Separate Stems"):
            # Create a progress bar
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            
            try:
                # Track start time
                start_time = time.time()
                
                # Update status
                status_text.text("Loading model...")
                progress_bar.progress(10, text="Loading Demucs model...")
                
                # Use our separate_audio function
                status_text.text("Processing audio...")
                
                # Process in stages to update progress
                stem_paths = separate_audio(
                    input_path, 
                    output_dir, 
                    model_options[model_name], 
                    device
                )
                
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
                
                # Create tabs for original and stems
                tabs = ["Original"] + list(stem_paths.keys())
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
                
                # Stem tabs
                for i, stem_name in enumerate(stem_paths.keys(), 1):
                    with selected_tab[i]:
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