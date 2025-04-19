"""
separation.py — generate high-quality stems from audio files using Demucs.
Dev notes:
 - Uses Demucs, Facebook's state-of-the-art source separation model
 - Handles common audio formats via torchaudio
 - Can separate into 4 stems: vocals, drums, bass, other
 - Includes functions for custom stem mixes
"""

import os
import sys
import subprocess
import argparse
import tempfile
import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

# Configure default device (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_output_dir(output_dir):
    """Make sure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)

def load_audio(input_file, target_sr=44100):
    """
    Load audio file using multiple backends for better format support.
    Tries torchaudio first, then librosa as fallback.
    """
    print(f"Loading audio file: {input_file}")
    
    try:
        # First attempt: torchaudio native loading
        waveform, sample_rate = torchaudio.load(input_file)
        print(f"Successfully loaded with torchaudio: {input_file}")
    except Exception as e:
        print(f"torchaudio failed to load file, trying librosa: {e}")
        try:
            # Second attempt: use librosa which supports more formats
            y, sample_rate = librosa.load(input_file, sr=None, mono=False)
            # Librosa might return mono, reshape for stereo if needed
            if y.ndim == 1:
                y = np.vstack([y, y])
            waveform = torch.tensor(y)
            print(f"Successfully loaded with librosa: {input_file}")
        except Exception as e:
            # If the file is m4a, aac, or mp4, try converting it to wav first
            ext = os.path.splitext(input_file)[1].lower()
            if ext in ['.m4a', '.aac', '.mp4']:
                print(f"Converting {ext} to temporary WAV for processing...")
                try:
                    # Create a temp WAV file
                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    
                    # Try using subprocess with ffmpeg directly
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", input_file,
                            "-ar", str(target_sr), "-ac", "2",
                            tmp_wav
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # Now load the converted WAV
                        waveform, sample_rate = torchaudio.load(tmp_wav)
                        print(f"Successfully converted and loaded: {input_file}")
                        
                        # Clean up temp file when done
                        os.remove(tmp_wav)
                    except Exception as ffmpeg_error:
                        # If subprocess approach fails, try librosa which can handle more formats
                        print(f"FFmpeg conversion failed: {ffmpeg_error}, trying librosa...")
                        y, sample_rate = librosa.load(input_file, sr=target_sr, mono=False)
                        if y.ndim == 1:
                            y = np.vstack([y, y])
                        waveform = torch.tensor(y)
                        print(f"Successfully loaded with librosa: {input_file}")
                except Exception as conversion_error:
                    raise RuntimeError(f"All attempts to load audio file failed: {conversion_error}")
            else:
                raise RuntimeError(f"Could not load audio file: {e}")
    
    # Resample if needed
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Check channel configuration
    if waveform.shape[0] > 2:
        print(f"Audio has {waveform.shape[0]} channels, truncating to stereo")
        waveform = waveform[:2]  # Take first two channels only
    
    # Make stereo if mono
    if waveform.shape[0] == 1:
        print("Converting mono to stereo")
        waveform = torch.cat([waveform, waveform], dim=0)
    
    return waveform, target_sr

def save_audio(audio, path, sample_rate):
    """Save audio array to a file using torchaudio."""
    # Normalize if float and outside [-1, 1]
    if np.issubdtype(audio.dtype, np.floating) and np.max(np.abs(audio)) > 1:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to torch tensor if numpy array
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    # Ensure it's float32 for torchaudio
    audio = audio.to(torch.float32)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save audio with backend specification to avoid errors
    try:
        torchaudio.save(path, audio, sample_rate, backend="soundfile")
    except Exception as e:
        print(f"Error with soundfile backend: {e}, trying default backend")
        try:
            torchaudio.save(path, audio, sample_rate)
        except Exception as e2:
            print(f"Error saving audio: {e2}")
            # Last resort: Use scipy to save if torchaudio fails
            try:
                from scipy.io import wavfile
                # Convert to numpy for scipy
                audio_np = audio.cpu().numpy()
                # Normalize to int16 range
                if audio_np.dtype != np.int16:
                    audio_np = (audio_np * 32767).astype(np.int16)
                wavfile.write(path, sample_rate, audio_np.T)  # scipy wants (samples, channels)
                print(f"Audio saved using scipy.io.wavfile: {path}")
            except Exception as e3:
                raise RuntimeError(f"All attempts to save audio failed: {e3}")

def separate_audio(input_file, output_dir, model_name="htdemucs", device=DEVICE):
    """
    Separate audio into stems using Demucs.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save stems
        model_name: Demucs model variant to use (default: htdemucs - Hybrid Transformer Demucs)
        device: Computation device ('cpu' or 'cuda')
    
    Returns:
        Dictionary mapping stem names to their file paths
    """
    ensure_output_dir(output_dir)
    
    # Load the model
    print(f"Loading Demucs model: {model_name}...")
    model = get_model(model_name)
    model.to(device)
    
    # Load audio directly with our enhanced loading function
    try:
        audio, sr = load_audio(input_file, model.samplerate)
        audio = audio.to(device)
    except Exception as e:
        print(f"Error loading audio: {e}")
        print("Make sure the file exists and is a valid audio format.")
        return {}
    
    # Apply the model
    print(f"Separating stems with Demucs {model_name} on {device}...")
    stems = apply_model(model, audio.unsqueeze(0), device=device)[0]
    
    # Get output file base name without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join(output_dir, base_name)
    ensure_output_dir(output_folder)
    
    # Save each stem
    stem_paths = {}
    print(f"Saving separated stems to {output_folder}...")
    
    for stem_idx, stem_name in enumerate(model.sources):
        stem_path = os.path.join(output_folder, f"{stem_name}.wav")
        stem_audio = stems[stem_idx].cpu().numpy()
        save_audio(stem_audio, stem_path, model.samplerate)
        stem_paths[stem_name] = stem_path
        print(f"  - Saved {stem_name} stem to {stem_path}")
    
    return stem_paths

def mix_stems(stem_paths, output_path, include_list=None):
    """
    Mix selected stems into a single audio file.
    Uses torchaudio to mix instead of FFmpeg.
    
    Args:
        stem_paths: Dictionary of stem paths
        output_path: Path to save the mixed audio
        include_list: List of stem names to include in the mix
    """
    if not include_list:
        return None
    
    mixed = None
    for stem_name in include_list:
        if stem_name in stem_paths:
            print(f"Adding {stem_name} to mix...")
            try:
                waveform, sr = torchaudio.load(stem_paths[stem_name])
                
                if mixed is None:
                    mixed = waveform
                else:
                    mixed = mixed + waveform
            except Exception as e:
                print(f"Error loading {stem_name} for mixing: {e}")
    
    if mixed is not None:
        # Normalize to prevent clipping
        if mixed.abs().max() > 0:
            mixed = mixed / mixed.abs().max() * 0.9
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save mixed audio
        try:
            torchaudio.save(output_path, mixed, sr, backend="soundfile")
            print(f"Mixed audio saved to {output_path}")
        except Exception as e:
            print(f"Error saving mixed audio: {e}")
            try:
                # Try another backend
                torchaudio.save(output_path, mixed, sr)
                print(f"Mixed audio saved using default backend to {output_path}")
            except Exception as e2:
                print(f"All attempts to save mixed audio failed: {e2}")
                return None
                
        return output_path
    
    return None

def create_stem_mix(input_file, output_dir, output_name="custom_mix.wav", 
                   vocals=True, drums=True, bass=True, other=True,
                   model_name="htdemucs", device=DEVICE):
    """
    Create a custom mix by including or excluding stems.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save stems and mix
        output_name: Filename for the created mix
        vocals/drums/bass/other: Whether to include each stem
        model_name: Demucs model to use
        device: Computation device
    
    Returns:
        Path to the created mix file
    """
    # First separate the audio
    stem_paths = separate_audio(input_file, output_dir, model_name, device)
    if not stem_paths:
        return None
    
    # Determine which stems to include
    stems_to_mix = []
    if vocals and "vocals" in stem_paths:
        stems_to_mix.append("vocals")
    if drums and "drums" in stem_paths:
        stems_to_mix.append("drums")
    if bass and "bass" in stem_paths:
        stems_to_mix.append("bass")
    if other and "other" in stem_paths:
        stems_to_mix.append("other")
    
    # Get base folder path and create mix output path
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join(output_dir, base_name)
    mix_path = os.path.join(output_folder, output_name)
    
    # Mix the stems
    return mix_stems(stem_paths, mix_path, stems_to_mix)

def main():
    parser = argparse.ArgumentParser(
        description="Separate audio into stems using Demucs"
    )
    parser.add_argument("input_file", help="Path to audio file (mp3/m4a/wav/...)")
    parser.add_argument("output_dir", help="Where to save stems")
    parser.add_argument("--model", choices=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"], 
                       default="htdemucs", help="Demucs model to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing")
    parser.add_argument("--mix", action="store_true", help="Create a custom mix")
    parser.add_argument("--no-vocals", action="store_true", help="Exclude vocals from mix")
    parser.add_argument("--no-drums", action="store_true", help="Exclude drums from mix")
    parser.add_argument("--no-bass", action="store_true", help="Exclude bass from mix")
    parser.add_argument("--no-other", action="store_true", help="Exclude other sounds from mix")
    
    args = parser.parse_args()
    
    # Set device
    device = "cpu" if args.cpu else DEVICE
    
    if args.mix:
        # Create custom mix
        create_stem_mix(
            args.input_file, 
            args.output_dir,
            vocals=not args.no_vocals,
            drums=not args.no_drums,
            bass=not args.no_bass,
            other=not args.no_other,
            model_name=args.model,
            device=device
        )
    else:
        # Simply separate
        separate_audio(args.input_file, args.output_dir, args.model, device)

if __name__ == "__main__":
    main() 