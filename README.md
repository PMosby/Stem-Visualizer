# Audio Stem Visualizer

A Streamlit application that allows users to separate audio tracks into individual stems (vocals, drums, bass, and other) using Demucs, and visualize the waveforms for each stem.

## Features

- Upload audio files in various formats (WAV, MP3, FLAC, OGG, M4A)
- Separate audio into stems (vocals, drums, bass, other) using Demucs models
- Visualize waveforms for the original audio and each stem
- Create custom mixes by selecting which stems to include
- Download the custom mix as a WAV file

## Requirements

- Python 3.7+
- Streamlit
- Librosa
- NumPy
- Matplotlib
- PyTorch
- Demucs (for audio separation)

## Installation

```bash
git clone https://github.com/PMosby/Stem-Visualizer.git
cd Stem-Visualizer
pip install -r requirements.txt
```

## Usage

```bash
streamlit run src/app.py
```

## Models

The application supports multiple Demucs models:
- Hybrid Transformer Demucs (default)
- Hybrid Transformer Demucs (fine-tuned)
- MDX-Extra (faster)
- MDX-Extra Quantized (smaller memory footprint)

## Project Structure

```
Stem-Visualizer/
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