from numba import jit
from pynamical import simulate, cubic_map, singer_map, bifurcation_plot, phase_diagram, phase_diagram_3d


pops = simulate(model=cubic_map, num_gens=1000,
                rate_min=2.99, num_rates=7, num_discard=100)
phase_diagram_3d(pops, xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1,
                 zmax=1, save=False, title='Phase Diagram of Acousticness', xlabel="Acousticness (t)", ylabel="Acousticness (t + 1)", zlabel="Acousticness (t + 2)", color='viridis')

pops = simulate(model=cubic_map, num_gens=3000,
                rate_min=3.5, num_rates=30, num_discard=100)
phase_diagram_3d(pops, xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, save=False, alpha=0.2, color='viridis',
                 azim=330, title='Cubic Map 3D Phase Diagram, r = 3.5 to 4.0')
