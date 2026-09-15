import random
import numpy as np
import time
import sys
from deap import base, creator, tools, algorithms
from project_class import Project


def evaluate(individual, project):
    phases_uint = np.array(individual, dtype=np.uint16)
    waveform = project.generate_waveform_phases_uint(phases_uint)
    cf_db = project.crest_factor(waveform)
    error = abs(cf_db - project.target_crest_db) if project.target_crest_db is not None else cf_db

    if project.is_new_best(phases_uint, error, cf_db):
        print(f"  → New best!  error={error:.6f}  CF={cf_db:.3f} dB  "
              f"{time.strftime('%H:%M:%S')}")

    return (error,)


def run_ga(project, base_pop_size=250, base_cxpb=0.75, base_mutpb=0.40,
           print_every=15, error_eps=1e-3):

    rng = np.random.default_rng()

    # Jitter hyperparameters
    pop_size = int(np.clip(round(rng.uniform(base_pop_size * 0.8, base_pop_size * 1.2) / 8) * 8, 144, 256))
    cxpb     = float(np.clip(rng.uniform(base_cxpb  * 0.87, base_cxpb  * 1.17), 0.50, 0.95))
    mutpb    = float(np.clip(rng.uniform(base_mutpb * 0.75, base_mutpb * 1.40), 0.15, 0.60))

    print(f"Jittered params:  pop={pop_size}  cxpb={cxpb:.3f}  mutpb={mutpb:.3f}")
    print("─" * 48)

    # DEAP setup
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    n_phases = len(project._freqs) if project._freqs is not None else 32769

    toolbox = base.Toolbox()
    toolbox.register("attr_phase", rng.integers, 0, project.phase_levels)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_phase, n=n_phases)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, project=project)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=project.phase_levels - 1,
                     indpb=5.0 / max(12, n_phases))
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("select_best", tools.selBest, k=3)

    # Initial population
    pop = toolbox.population(n=pop_size)

    # Seed with known best if available
    if project.best_phases_uint is not None:
        best_ind = creator.Individual(project.best_phases_uint.tolist())
        best_ind.fitness.values = (project.best_error,)
        pop[0] = best_ind
        print(f"→ Seeded population with known best (error={project.best_error:.6f})\n")

    hof = tools.HallOfFame(5)

    print(f"Target: {project.target_crest_db:.1f} dB  |  Running... (Ctrl+C to stop)\n")

    converged = False
    try:
        for gen in range(1, 10_000_000):
            offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            elites   = toolbox.select_best(pop + offspring)
            survivors = toolbox.select(offspring + pop, k=pop_size - len(elites))
            pop[:]   = elites + survivors

            hof.update(pop)
            best_err = hof[0].fitness.values[0]

            if gen % print_every == 0:
                best_cf = project.crest_factor(
                    project.generate_waveform_phases_uint(np.array(hof[0], dtype=np.uint16))
                )
                print(f"Gen {gen:6d} | error={best_err:.6f} | CF={best_cf:.3f} dB")

            if best_err < error_eps:
                print(f"\n✅ Converged at generation {gen}  (error={best_err:.10f})")
                converged = True
                break

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")

    # Save project
    print("\nSaving project...")
    project.save("test_project.proj")
    print(f"✅ Saved  |  best error={project.best_error:.6f}  |  CF={project.best_crest_db:.3f} dB")

    return converged


if __name__ == "__main__":
    filename = "test_project.proj"

    print(f"Loading: {filename}\n")
    try:
        project = Project.load(filename)
    except Exception as e:
        print(f"Failed to load project: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"  Noise color : {project.noise_color}")
    print(f"  Freq range  : {project.freq_low} – {project.freq_high} Hz")
    print(f"  Target CF   : {project.target_crest_db:.1f} dB")
    print(f"  Best error  : {project.best_error:.6f}")
    print(f"  Best CF     : {project.best_crest_db:.3f} dB\n")

    project._ensure_initialized()

    run_ga(project, base_pop_size=250, base_cxpb=0.75, base_mutpb=0.40, print_every=15)