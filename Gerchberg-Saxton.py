from project_class import Project
import numpy as np
import soundfile as sf
import math

clip_ratio = 0.99

def expand_above_rms(waveform, db_threshold=6.0, expand_ratio=1.2):
    """
    Expands signal peaks above (RMS + db_threshold).
    
    Parameters:
        waveform     : input signal (numpy array)
        db_threshold : dB above RMS at which expansion kicks in
        expand_ratio : how much to amplify the excess (2.0 = doubles it)
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    threshold = rms * (10 ** (db_threshold / 20))  # convert dB to linear and add to RMS

    excess = np.abs(waveform) - threshold
    signs = np.sign(waveform)

    waveform_expanded = np.where(
        np.abs(waveform) > threshold,
        signs * (threshold + excess * expand_ratio),
        waveform
    )
    return waveform_expanded

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
	

	phases_rad = proj.best_phases_rad
	prev_error = None

	if not math.isclose(proj.best_crest_db, proj.target_crest_db, rel_tol=1e-3):
		if proj.best_crest_db > proj.target_crest_db:

			for i in range(25000):
				waveform = proj.generate_waveform_phases_rad(phases_rad)
				peak = np.max(np.abs(waveform))
				cf_db = proj.crest_factor(waveform)
				error = cf_db - proj.target_crest_db	
				if prev_error is not None and np.sign(error) != np.sign(prev_error):
					print("Target crossed (sign change).")
					break
				threshold = clip_ratio * peak
				waveform_clipped = np.clip(waveform, -threshold, threshold)
				S = np.fft.rfft(waveform_clipped)
				phases_rad = np.angle(S)  # radians directly, no np.exp needed here
				print(f"iter {i:3d} | CF: {cf_db:.2f} dB | error: {error:.2f} dB")
				prev_error = error
			
		elif proj.best_crest_db < proj.target_crest_db:
			for i in range(25000):
				waveform = proj.generate_waveform_phases_rad(phases_rad)
				peak = np.max(np.abs(waveform))
				cf_db = proj.crest_factor(waveform)
				error = cf_db - proj.target_crest_db	
				if prev_error is not None and np.sign(error) != np.sign(prev_error):
					print("Target crossed (sign change).")
					break
				S = np.fft.rfft(expand_above_rms(waveform, db_threshold=9.0, expand_ratio=1.003))
				phases_rad = np.angle(S)  # radians directly, no np.exp needed here
				print(f"iter {i:3d} | CF: {cf_db:.2f} dB | error: {error:.2f} dB")
				prev_error = error
		
		proj.is_new_best(phases_rad, np.abs(error), cf_db)
		proj.save(filename)

			

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