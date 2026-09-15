from project_class import Project
import numpy as np
import soundfile as sf
import math

clip_ratio = 0.997

def random_binary_phase_search(proj, n_trials=1000):
    n_bins = proj.fft_size // 2 + 1
    best_cf = np.inf
    best_phases = None

    for i in range(n_trials):
        # Random sequence of 0 or pi
        phases_rad = np.random.choice([0.0, np.pi], size=n_bins)
        
        waveform = proj.generate_waveform_phases_rad(phases_rad)
        cf_db = proj.crest_factor(waveform)

        if cf_db < best_cf:
            best_cf = cf_db
            best_phases = phases_rad.copy()
            print(f"trial {i:5d} | CF: {cf_db:.2f} dB")

    return best_phases, best_cf


def rudin_shapiro(n):
    """
    Generate Rudin-Shapiro sequence as phase values (0 or pi).
    Uses the correct recurrence on pairs (P, Q).
    """
    # RS is defined on pairs: P_n, Q_n
    # P_0 = 1, Q_0 = 1
    # P_{n+1}(k) = P_n(k) if k < 2^n, Q_n(k - 2^n) if k >= 2^n  ... etc
    # Easier: generate bit by bit from definition
    
    result = np.zeros(n, dtype=np.float64)
    for k in range(n):
        # Count consecutive 11 pairs in binary representation of k
        b = bin(k)[2:]  # binary string
        count = sum(1 for i in range(len(b)-1) if b[i] == '1' and b[i+1] == '1')
        result[k] = 0.0 if count % 2 == 0 else np.pi
    
    return result

def main():
	filename = "test_project.proj"


	# Load it back
	proj = Project.load(filename)
	# Get project info
	info = proj.get_info()
	print("\n📊 Project Info:")
	for key, value in info.items():
		print(f"  {key}: {value}")
	print(f"📂 Project loaded: {proj.notes}")
	print(f"  Best error from loaded: {proj.best_error:.2f} dB")
	print(f"  Best crest from loaded: {proj.best_crest_db:.2f} dB")
	
	print("\n✅ Project loaded succesfully!")
	



	n_trials=10000
	n_bins = proj.fft_size // 2 + 1
	best_cf = np.inf
	best_phases = None

	for i in range(n_trials):
		# Random sequence of 0 or pi
		phases_rad = np.random.choice(np.linspace(0, np.pi, 256), size=n_bins)
		waveform = proj.generate_waveform_phases_rad(phases_rad)
		cf_db = proj.crest_factor(waveform)

	if cf_db < best_cf:
		best_cf = cf_db
		best_phases = phases_rad.copy()
		print(f"trial {i:5d} | CF: {cf_db:.2f} dB")

    
		
	waveform_out = proj.generate_waveform_phases_rad(best_phases)
	total_samples = int(proj.sample_rate * proj.duration_seconds)
	repeats = math.ceil(total_samples / len(waveform_out))
	waveform_out = np.tile(waveform_out, repeats)[:total_samples].astype(np.float32)
	waveform_out = waveform_out / np.max(np.abs(waveform_out)) * 10**(-1.05/20)
	filename = (
        f"best_noise_{proj.noise_color}_"
        f"{int(proj.freq_low)}-{int(proj.freq_high)}Hz_"
        f"crest{cf_db:.1f}dB.wav"
    )
	sf.write(filename, waveform_out, proj.sample_rate, subtype="PCM_24")
	print(f"✅ Saved: {filename}")	

			

	# Show best result
	print("\n🏆 Best Result:")
	print(f"  Error: {proj.best_error:.2f} dB")
	print(f"  Crest Factor: {proj.best_crest_db:.2f} dB")
	print(f"  Target: {proj.target_crest_db:.2f} dB")


	# Save project
	filename = "test_project.proj"
	proj.save(filename)
	print(f"\n💾 Project saved to {filename}")

if __name__ == "__main__":
	main()