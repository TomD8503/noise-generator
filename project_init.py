# test_simple.py

from project_class import Project
import numpy as np

def main():
    print("=" * 50)
    print("SIMPLE PROJECT TEST")
    print("=" * 50)
    
    # Create project with simple config
    proj = Project(
        duration_seconds = 600,
        sample_rate=48000,
        fft_size=65536,  # Smaller for faster testing
        freq_low=20,
        freq_high=20000,
        noise_color="pink",
        phase_levels=16384,
        target_crest_db=6.0,
        octaves=3,
        notes="Simple test"
    )
    
    # Finalize (computes phase_step)
    proj.initialize()
    
    # Get project info
    info = proj.get_info()
    print("\n📊 Project Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    waveform, phases, error, cf_db = proj.generate_and_evaluate()
    proj.is_new_best(phases, error, cf_db)

    
    # Show best result
    print("\n🏆 Best Result:")
    print(f"  Error: {proj.best_error:.2f} dB")
    print(f"  Crest Factor: {proj.best_crest_db:.2f} dB")
    print(f"  Target: {proj.target_crest_db:.2f} dB")
    
    
    # Save project
    filename = "test_project.proj"
    proj.save(filename)
    print(f"\n💾 Project saved to {filename}")
    
    # Load it back
    loaded = Project.load(filename)
    # Get project info
    info = loaded.get_info()
    print("\n📊 Project Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print(f"📂 Project loaded: {loaded.notes}")
    print(f"  Best error from loaded: {loaded.best_error:.2f} dB")
    print(f"  Best crest from loaded: {loaded.best_crest_db:.2f} dB")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
