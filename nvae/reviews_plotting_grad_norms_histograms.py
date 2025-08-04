'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/reviews_plotting_grad_norms_histograms.py

'''

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator


# Attack types
attck_type1 = "grill_l2_pr"
attck_type2 = "la_l2_pr"

desired_norm_l_inf = 0.05 
step = 99

# Load the gradient norms
all_grad_norms1 = np.load(f"nvae/grad_distribution/grad_norms_list_{attck_type1}_norm_bound_{desired_norm_l_inf}_step_{step}.npy")
all_grad_norms2 = np.load(f"nvae/grad_distribution/grad_norms_list_{attck_type2}_norm_bound_{desired_norm_l_inf}_step_{step}.npy")

# Plotting

colors = ['blue', 'orange', 'green', 'red', 'lime', 'teal', 'indigo', 'gold']


plt.figure(figsize=(6, 5))  # Adjust the width and height as needed

plt.plot(all_grad_norms1, linestyle='-', label='ALMA(L-2)', color = 'lime')
plt.plot(all_grad_norms2, linestyle='-', label='LA(L-2)', color='blue')

#plt.title("L2 Norm of Gradient Over Optimization Steps")
plt.xlabel("Step", fontsize=28)
#plt.ylabel("L2 Norm of ∇(loss) w.r.t. noise_addition")
plt.ylabel(r"$\left\| \nabla ( L )\right\|_2$", fontsize=28)

# Format y-axis in log scale with powers of 10
plt.yscale('log')
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$1e^{{{int(np.log10(y))}}}$'))

plt.grid(False, which='both', axis='y')
#plt.legend(fontsize=20)  # Show legend
plt.xticks(fontsize=28,rotation=45)
plt.yticks(fontsize=28)

handles, labels = plt.gca().get_legend_handles_labels()

# Increase line thickness in the legend
for handle in handles:
    handle.set_linewidth(4)
plt.tight_layout()

# Save and show
plt.savefig(f"nvae/address_reviews/GradL2Norm_vs_Steps_compare_norm_bound_{desired_norm_l_inf}_step_{step}_bp.png")
plt.show()
plt.close()



#Histograms 

#hist_step = 20

for hist_step in range(100): 
    grad_values = np.load("nvae/grad_distribution/grad_values_"+str(attck_type1)+"_norm_bound_"+str(desired_norm_l_inf)+"_step_"+str(hist_step)+".npy")

    plt.figure(figsize=(6, 4))
    #plt.hist(grad_values, bins=100, range=(-0.1, 0.1), density=False, alpha=0.75)
    plt.hist(grad_values, bins=100, range=(-30, 30), density=False, alpha=0.75, color='lime')

    #plt.title("Gradient loss pa wrt perturbation tensor")
    #plt.xlabel("Loss partial derivatives distribution", fontsize=28)
    plt.ylabel("Frequency", fontsize=28)
    plt.grid(False)

    ax = plt.gca()
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))  # Formats as 1e-4, 1e-5, etc.

    '''def clean_sci_notation(x, pos):
        # Format x as scientific notation, remove '+'
        s = f"{x:.0e}"      # e.g., '2e+01'
        return s.replace('e+','e').replace('e0','e')  # Clean: '2e1', '1e3'''

    def clean_sci_notation(x, pos):
        if x == 0:
            return '0'
        s = f"{x:.0e}"
        return s.replace('e+','e').replace('e0','e')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(clean_sci_notation))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(clean_sci_notation))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.xticks(rotation=45, fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()  # Adjust layout to avoid overlap

    plt.show()
    plt.savefig("nvae/address_reviews/Histogram_attack_type"+str(attck_type1)+"_norm_bound_"+str(desired_norm_l_inf)+"_step_"+str(hist_step)+"_.png")   #####this
    plt.close()


for hist_step in range(100): 
    grad_values = np.load("nvae/grad_distribution/grad_values_"+str(attck_type2)+"_norm_bound_"+str(desired_norm_l_inf)+"_step_"+str(hist_step)+".npy")

    plt.figure(figsize=(6, 4))
    #plt.hist(grad_values, bins=100, range=(-0.1, 0.1), density=False, alpha=0.75)
    plt.hist(grad_values, bins=100, range=(-30, 30), density=False, alpha=0.75, color='blue')

    #plt.title("Gradient loss pa wrt perturbation tensor")
    #plt.xlabel("Loss partial derivatives distribution", fontsize=28)
    plt.ylabel("Frequency", fontsize=28)
    plt.grid(False)

    ax = plt.gca()
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))  # Formats as 1e-4, 1e-5, etc.

    '''def clean_sci_notation(x, pos):
        # Format x as scientific notation, remove '+'
        s = f"{x:.0e}"      # e.g., '2e+01'
        return s.replace('e+','e').replace('e0','e')  # Clean: '2e1', '1e3'''

    def clean_sci_notation(x, pos):
        if x == 0:
            return '0'
        s = f"{x:.0e}"
        return s.replace('e+','e').replace('e0','e')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(clean_sci_notation))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(clean_sci_notation))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.xticks(rotation=45, fontsize=28)
    plt.yticks(fontsize=28)

    plt.tight_layout()  # Adjust layout to avoid overlap

    plt.show()
    plt.savefig("nvae/address_reviews/Histogram_attack_type"+str(attck_type2)+"_norm_bound_"+str(desired_norm_l_inf)+"_step_"+str(hist_step)+"_.png")   #####this
    plt.close()
