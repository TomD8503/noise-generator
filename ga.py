# ga.py
import random
import numpy as np
import time
import math
import soundfile as sf
from deap import base, creator, tools, algorithms
from project_class import Project   # your class

def evaluate(individual, project):
    phases_uint = np.array(individual, dtype=np.uint16)
    waveform = project.generate_waveform_phases_uint(phases_uint)
    cf_db = project.crest_factor(waveform)

    error = abs(cf_db - project.target_crest_db) if project.target_crest_db is not None else cf_db

    if project.is_new_best(phases_uint, error, cf_db):
        print(f"  → New best!  error = {error:.6f}   CF = {cf_db:.3f} dB   "
              f"  {time.strftime('%H:%M:%S', time.localtime(project.last_modified))}")

    return (error,)


def setup_toolbox(project):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_phase", random.randint, 0, project.phase_levels - 1)

    n_phases = len(project._freqs) if project._freqs is not None else 32769

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_phase, n=n_phases)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, project=project)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=0, up=project.phase_levels - 1,
                     indpb=5.0 / max(12, n_phases))
    toolbox.register("select", tools.selTournament, tournsize=4)

    # For elitism
    toolbox.register("select_best", tools.selBest, k=3)  # mild elitism: keep top 3

    return toolbox
def _jitter_ga_params(rng, base_pop_size, base_cxpb, base_mutpb):
    """Apply mild uniform jitter to GA hyperparameters. Returns (pop_size, cxpb, mutpb)."""
    pop_size = int(rng.uniform(base_pop_size * 0.80, base_pop_size * 1.20))
    pop_size = max(144, min(256, round(pop_size / 8) * 8))  # snap to multiple of 8
    cxpb     = float(np.clip(rng.uniform(base_cxpb  * 0.87, base_cxpb  * 1.17), 0.50, 0.95))
    mutpb    = float(np.clip(rng.uniform(base_mutpb * 0.75, base_mutpb * 1.40), 0.15, 0.60))
    return pop_size, cxpb, mutpb

def run_ga_forever(project,
                   base_pop_size=300,
                   base_cxpb=0.75,
                   base_mutpb=0.30,
                   print_every=15):
    """
    Runs the GA forever (until Ctrl+C).
    - Uses only numpy.random.default_rng (no python random module)
    - Applies mild random jitter to hyperparameters → different every run
    - No fixed seed → different starting conditions and path every launch
    """

    # Fresh RNG every run → different behaviour every time
    rng = np.random.default_rng()

    pop_size, cxpb, mutpb = _jitter_ga_params(rng, base_pop_size, base_cxpb, base_mutpb)

    print(f"Jittered params:  pop={pop_size}  cxpb={cxpb:.3f}  mutpb={mutpb:.3f}")
    print(f"  (bases were:    pop={base_pop_size}  cxpb={base_cxpb:.3f}  mutpb={base_mutpb:.3f})")
    print("─" * 48)

    # ── Create toolbox using our rng ──
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Use rng for all random operations
    toolbox.register("attr_phase", rng.integers, 0, project.phase_levels)  # exclusive upper

    n_phases = len(project._freqs) if project._freqs is not None else 32769

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_phase, n=n_phases)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, project=project)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=0, up=project.phase_levels-1,
                     indpb=5.0 / max(12, n_phases))

    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("select_best", tools.selBest, k=3)  # mild elitism

    # ── Initial population ──
    pop = toolbox.population(n=pop_size)

    # Insert known best if available
    if project.best_phases_uint is not None:
        try:
            best_ind = toolbox.individual()
            best_ind[:] = project.best_phases_uint.tolist()
            best_ind.fitness.values = (project.best_error,)
            pop[0] = best_ind
            print("→ Inserted loaded best phases into initial population")
        except Exception as e:
            print(f"Warning: Could not insert known best: {e}")

    hof = tools.HallOfFame(5)

    print(f"GA running   |   pop = {pop_size}   |   target = {project.target_crest_db:.1f} dB")
    print("Press Ctrl+C to stop\n")

    gen = 0

    try:
        ERROR_EPS = 1e-3  # stop when error is close enough to zero
        best_cf = None
        while True:
            gen += 1

            offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Mild elitism
            elites = toolbox.select_best(pop + offspring)

            needed = pop_size - len(elites)
            survivors = toolbox.select(offspring + pop, k=needed) if needed > 0 else []

            pop[:] = elites + survivors

            hof.update(pop)

            best_err = hof[0].fitness.values[0]

        
            if abs(best_err) < ERROR_EPS:
                print(f"\nConverged at generation {gen}")
                print(f"Final error = {best_err:.10f}")
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
                break

            if gen % print_every == 0:
                best_cf = project.crest_factor(project.generate_waveform_phases_uint(np.array(hof[0], dtype=np.uint16)))
                print(f"Gen {gen:5d} | best err = {best_err:.6f} | CF = {best_cf:.3f} dB")

    except KeyboardInterrupt:
        print("\n" + "═" * 80)
        print("Stopped by Ctrl+C")
        print("Final known best:")
        print(f"  error         = {project.best_error:.6f}")
        print(f"  crest factor  = {project.best_crest_db:.3f} dB")
        print(f"  target        = {project.target_crest_db:.1f} dB")
        print(f"  last update   = {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(project.last_modified))}")
        if project.best_phases_uint is not None:
            print(f"  phases shape  = {project.best_phases_uint.shape}")
        print("═" * 80)


# ────────────────────────────────────────────────
if __name__ == "__main__":
    filename = "test_project.proj"

    print(f"Attempting to load project from: {filename}\n")

    try:
        project = Project.load(filename)
        print("Project loaded successfully")
        
        info = project.get_info()
        print("\n" + "═" * 60)
        print("📊 Project Info:")
        for key, value in info.items():
            print(f" {key}: {value}")
        print(f"📂 Notes: {project.notes}")
        print(f" Best error from file : {project.best_error:.6f} dB")
        print(f" Best crest factor    : {project.best_crest_db:.3f} dB")
        print("═" * 60 + "\n")

        if project.best_phases_uint is None:
            raise ValueError("Loaded project has no best_phases_uint → cannot continue")

        print(f"→ Starting from known best (error = {project.best_error:.6f})\n")

    except Exception as e:
        print("\n" + "═" * 80)
        print("CRITICAL ERROR – cannot continue without valid project file")
        print(f"Failed to load {filename}:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("Exiting.")
        print("═" * 80)
        import sys
        sys.exit(1)

    project._ensure_initialized()

    run_ga_forever(
    project,
    base_pop_size=250,
    base_cxpb=0.75,
    base_mutpb=0.40,
    print_every=15
)