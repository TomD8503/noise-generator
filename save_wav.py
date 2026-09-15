# save_best_noise.py

import soundfile as sf
import numpy as np
import math
from project_class import Project
import sys
import soundfile as sf

def save_wav_file(project_file):
    # Load project
    print(f"Loading project: {project_file}")
    proj = Project.load(project_file)
    
    # Check if there's a best candidate
    if proj.best_phases_rad is None:
        print("No best candidate found in project file!")
        return
    
    # Generate signal from best phases
    print("Generating signal from best candidate...")
    waveform = proj.generate_waveform_phases_rad(proj.best_phases_rad)
    cf_db = proj.crest_factor(waveform)
    error = abs(cf_db - proj.target_crest_db)
   
    
    print(f"Best crest factor: {cf_db:.2f} dB")
    print(f"Error from target: {error:.2f} dB")
    
    # Prepare for WAV export
    total_samples = int(proj.sample_rate * proj.duration_seconds)
    repeats = math.ceil(total_samples / len(waveform))
    waveform_out = np.tile(waveform, repeats)[:total_samples].astype(np.float32)
    waveform_out /= np.max(np.abs(waveform_out)) # normalize to 0dB
    
    # Generate filename
    filename = (
        f"best_noise_{proj.noise_color}_"
        f"{int(proj.freq_low)}-{int(proj.freq_high)}Hz_"
        f"crest{cf_db:.1f}dB.wav"
    )
    
    # Save WAV
    sf.write(filename, waveform_out, proj.sample_rate, subtype="PCM_24")
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    # Use command line argument or hardcoded filename
    if len(sys.argv) > 1:
        project_file = sys.argv[1]
    else:
        project_file = "test_project.proj"  # Default filename
    
    save_wav_file(project_file)