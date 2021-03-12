import IPython.display as IPdisplay
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynamical
from pynamical import simulate, bifurcation_plot, save_fig

title_font = pynamical.get_title_font()
label_font = pynamical.get_label_font()

# run the logistic model for 20 generations for 7 growth rates between 0.5 and 3.5 then view the output
pops = simulate(num_gens=20, rate_min=0.5, rate_max=3.5, num_rates=7)
pops.applymap(lambda x: '{:03.3f}'.format(x))


def get_colors(cmap, n, start=0., stop=1., alpha=1., reverse=False):
    '''return n-length list of rgba colors from the passed colormap name and alpha,
       limit extent by start/stop values and reverse list order if flag is true'''
    colors = [cm.get_cmap(cmap)(x) for x in np.linspace(start, stop, n)]
    colors = [(r, g, b, alpha) for r, g, b, _ in colors]
    return list(reversed(colors)) if reverse else colors


# plot the results of the logistic map run for these 7 different growth rates
# color_list = ['#cc00cc', '#4B0082', '#0066cc', '#33cc00', '#cccc33', '#ff9900', '#ff0000']
color_list = get_colors('viridis', n=len(pops.columns), start=0., stop=1)
for color, rate in reversed(list(zip(color_list, pops.columns))):
    ax = pops[rate].plot(kind='line', figsize=[10, 6],
                         linewidth=2.5, alpha=0.95, c=color)
ax.grid(True)
ax.set_ylim([0, 1])
ax.legend(title='Growth Rate', loc=3, bbox_to_anchor=(1, 0.525))
ax.set_title('Logistic Model Results by Growth Rate',
             fontproperties=title_font)
ax.set_xlabel('G', fontproperties=label_font)
ax.set_ylabel('P', fontproperties=label_font)

save_fig('logistic-map-growth-rates')
plt.xlabel('s')
plt.show()
# run the model for 100 generations across 1000 growth rate steps from 0 to 4 then plot the bifurcation diagram
pops = simulate(num_gens=100, rate_min=0, rate_max=4,
                num_rates=1000, num_discard=1)
bifurcation_plot(pops, xmin=2,filename='logistic-map-bifurcation-0',
                 color='#614C99', show=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Growth rate of tempo', size=20)
plt.ylabel('Δtempo/tempo', size=20)
plt.show()
# plot the bifurcation diagram for 200 generations, but this time throw out the first 100 rows
# 200-100=100, so we still have 100 generations in the plot, just like in the previous cell
# this will show us only the attractors (aka, the values that each growth rate settles on over time)
pops = simulate(num_gens=100, rate_min=0, rate_max=4,
                num_rates=1000, num_discard=100)
bifurcation_plot(pops, filename='logistic-map-bifurcation-1',
                 xlabel='Growth rate of tempo', ylabel='tempo')
# run the model for 300 generations across 1,000 growth rate steps from 2.8 to 4, and plot the bifurcation diagram
# this plot is a zoomed-in look at the first plot and shows the period-doubling path to chaos
pops = simulate(num_gens=100, rate_min=2.8, rate_max=4,
                num_rates=1000, num_discard=200, initial_pop=0.1)
bifurcation_plot(pops, xmin=2.8, xmax=4, filename='logistic-map-bifurcation-2')
# run the model for 200 generations across 1,000 growth rate steps from 3.7 to 3.9, and plot the bifurcation diagram
# this plot is a zoomed-in look at the first plot and shows more detail in the chaotic regimes
pops = simulate(num_gens=100, rate_min=3.7, rate_max=3.9,
                num_rates=1000, num_discard=100)
bifurcation_plot(pops, xmin=3.7, xmax=3.9,
                 filename='logistic-map-bifurcation-3')
# run the model for 500 generations across 1,000 growth rate steps from 3.84 to 3.856, and plot the bifurcation diagram
# throw out the first 300 generations, so we end up with 200 generations in the plot
# this plot is a zoomed-in look at the first plot and shows the same structure we saw at the macro-level
pops = simulate(num_gens=200, rate_min=3.84, rate_max=3.856,
                num_rates=1000, num_discard=300)
bifurcation_plot(pops, xmin=3.84, xmax=3.856, ymin=0.445,
                 ymax=0.552, filename='logistic-map-bifurcation-4')
# plot the numeric output of the logistic model for growth rates of 3.9 and 3.90001
# this demonstrates sensitive dependence on the parameter

plt.show()
