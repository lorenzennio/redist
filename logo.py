import numpy as np
import scipy.stats as stats 
from matplotlib import pyplot as plt

from matplotlib import rcParams

x = np.linspace (0, 320, 1000) 

aa = np.linspace (6, 13, 30)
sc = np.linspace (6, 13, 30)

fig, ax = plt.subplots(figsize=(10, 5), dpi=800)

for i, (a, s) in enumerate(zip(aa, sc)):
    ax.plot(x, 
            stats.gamma.pdf(x, a=a, scale=s), 
            'darkred', 
            alpha = (30-i)/30,
            linewidth = 1.5)

ax.axis('off')

font = {'family':'serif', 'weight':'light', 'color':'darkred','size':120}
ax.text(0.31, 0.40, 'redist', transform=plt.gca().transAxes, fontdict=font, verticalalignment='top')

plt.savefig('logo.svg', transparent=True)