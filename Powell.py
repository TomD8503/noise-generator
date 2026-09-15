from project_class import Project
import numpy as np

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


	for i in range(50000):
		waveform, phases, error, cf_db = proj.generate_and_evaluate()
		if proj.is_new_best(phases, error, cf_db):
			print(f"  ⭐ New best #{i+1}: error={error:.2f} dB, crest={cf_db:.2f} dB")

			

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
