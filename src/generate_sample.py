"""Generate a sample audio file for testing."""

import numpy as np
import torchaudio
import torch
import os

def generate_sine_wave(freq, duration, sample_rate):
    """Generate a sine wave at given frequency, duration and sample rate."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def generate_sample(output_path, sample_rate=44100, duration=10):
    """Generate a sample audio with multiple frequencies mixed together."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sine waves at different frequencies (musical notes)
    c_note = generate_sine_wave(261.63, duration, sample_rate)  # C4
    e_note = generate_sine_wave(329.63, duration, sample_rate)  # E4
    g_note = generate_sine_wave(392.00, duration, sample_rate)  # G4
    
    # Add a percussion-like element (pulsed noise)
    percussion = np.zeros(int(sample_rate * duration))
    for i in range(0, len(percussion), int(sample_rate * 0.5)):  # Every half second
        if i + 2000 < len(percussion):
            percussion[i:i+2000] = np.random.random(2000) * 0.2
    
    # Mix them together with different weights to simulate different instruments
    mixed = c_note * 0.5 + e_note * 0.3 + g_note * 0.4 + percussion
    
    # Add some amplitude modulation to simulate vocals
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Slow modulation
    vocal_like = generate_sine_wave(220.0, duration, sample_rate) * am  # A3 with modulation
    
    # Final mix
    final_mix = mixed * 0.8 + vocal_like * 0.4
    
    # Normalize
    final_mix = final_mix / np.max(np.abs(final_mix))
    
    # Convert to stereo
    stereo = np.vstack([final_mix, final_mix])
    
    # Save as WAV using torchaudio
    torchaudio.save(
        output_path, 
        torch.tensor(stereo, dtype=torch.float32), 
        sample_rate
    )
    
    print(f"Sample audio generated and saved to {output_path}")

if __name__ == "__main__":
    output_path = "data/input/sample.wav"
    generate_sample(output_path)
    print(f"You can now run: python src/separation.py {output_path} data/stems") 