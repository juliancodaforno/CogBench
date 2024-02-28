"""
This script creates a legend for the CoT and SB plots for the paper.
"""
import matplotlib.pyplot as plt
plt.figure(figsize=(0.3, 0.1))

colors = ['tab:grey' , 'tab:grey'] + [f'C0{i}' for i in range(6)] 
alphas = [0.3, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Add your desired alpha values here
f = lambda m,c,a: plt.plot([],[],marker=m, color=c, ls="none", alpha=a)[0]
handles = [f("s", colors[i], alphas[i]) for i in range(8)]
labels = [ 'with CoT', 'with SB', 'GPT-4', 'text-bison', 'Claude-2', 'Claude-1', 'LLaMA-2-70', 'Aggregated scores']
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False
                    , bbox_to_anchor=(1.1, 1)
                     )

def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

plt.fontsize = 12

export_legend(legend)
plt.axis('off')  # This line removes the axes
# plt.tight_layout(pad=0.05)  # This line ensures the legend fits in the figure
plt.savefig('plots/cot_sb/legend.pdf', bbox_inches='tight')