# Audio Stem Separator & Visualizer

A Streamlit application that allows users to separate audio tracks into individual stems (vocals, drums, bass, and other) using Demucs, and visualize the waveforms for each stem. Features advanced visualizations inspired by nature, noir, and dance aesthetics, along with an immersive Three.js experience that synchronizes visuals with the audio playback.

## Features

- Upload audio files in various formats (WAV, MP3, FLAC, OGG, M4A)
- Separate audio into stems (vocals, drums, bass, other) using Demucs models
- Visualize waveforms for the original audio and each stem
- Create custom mixes by selecting which stems to include
- Download the custom mix as a WAV file
- Generate advanced visualizations with different themes:
  - Nature-inspired visualizations with organic flowing shapes
  - Noir-style high-contrast visualizations
  - Dance-inspired dynamic visualizations
  - Epic combined visualizations that represent all stems
- Interactive explorations with real-time controls
- Create animated GIFs from audio visualizations
- Immersive Three.js experience with:
  - Real-time audio-reactive 3D visualizations
  - Different visual elements for each stem
  - Playback controls (play/pause)
  - Synchronized visuals that respond to audio frequencies

## Requirements

- Python 3.7+
- Streamlit
- Librosa
- NumPy
- Matplotlib
- PyTorch
- Demucs (for audio separation)
- Plotly (for interactive visualizations)

## Installation

```bash
git clone https://github.com/PMosby/Stem-Separator.git
cd Stem-Separator
pip install -r requirements.txt
```

## Usage

```bash
streamlit run src/app.py
```

## Visualization Modes

The app offers several visualization modes that are especially suited for music like Mycota's "I am a General":

1. **Nature Theme**: Organic flowing visualizations that represent the natural evolution of sound
2. **Noir Theme**: High-contrast visualizations with sharp lines, ideal for jazz fusion elements
3. **Dance Theme**: Dynamic visualizations that pulse with the rhythm, emphasizing movement
4. **Epic Combined**: A layered visualization showing all stems in a cohesive presentation
5. **Immersive Three.js Experience**: A full-screen audio-reactive 3D visualization with:
   - Flowing particle systems for vocals (nature-inspired)
   - Geometric cube patterns for drums (noir-inspired)
   - Rippling wave surfaces for bass (dance-inspired)
   - Orbital sphere systems for other instruments (epic fusion style)

Each stem (vocals, drums, bass, other) is represented with different visual elements and colors to create a comprehensive visual experience of the music.

## Models

The application supports multiple Demucs models:
- Hybrid Transformer Demucs (default)
- Hybrid Transformer Demucs (fine-tuned)
- MDX-Extra (faster)
- MDX-Extra Quantized (smaller memory footprint)

## Project Structure

```
Im_A_General_Visualizer/
├── data/
│   ├── input/          # Place input audio files here
│   └── stems/          # Output stems will be saved here
├── src/
│   ├── separation.py   # Main script for audio separation
│   ├── app.py          # Streamlit web interface
│   └── generate_sample.py # Generate sample audio for testing
├── venv/               # Virtual environment
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Troubleshooting

- If you encounter file format issues, convert your audio to WAV first
- For large files, expect longer processing times
- The MDX-Extra Quantized model requires the diffq package (`pip install diffq`)
- For .m4a files, the app will attempt to convert them internally, but results may vary 