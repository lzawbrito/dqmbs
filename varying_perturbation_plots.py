import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.interpolate import interp1d




df = pd.read_csv('./data/dqmbs_varying_proj_l=4.csv')
df = df.sort_values(by=['epsilon', 't'])

epsilon = [0.0, 0.25, 1.0]

fids = []
sts = []
for e in epsilon:
	df_e = df[df['epsilon'] == e]	
	fids.append(df_e['fid'])
	sts.append(df_e['st'])

t = df[df['epsilon'] == epsilon[0]]['t'].to_numpy()

# Interpolation so graph looks a little nicer. Just smooths it a bit.
tnew = np.linspace(0, t[-1], 500) 
fids_interp = [] 
sts_interp = [] 
for f, s in zip(fids, sts): 
	fids_interp.append(interp1d(t, f, kind='cubic')(tnew))
	sts_interp.append(interp1d(t, s, kind='cubic')(tnew))

fig, ax = plt.subplots(nrows=2) 

color = ['cornflowerblue', 'lightcoral', 'mediumseagreen']
for a in ax:
	a.set_prop_cycle(color=color)

eps_label = [f"$\\epsilon = {e}$" for e in epsilon]

ax[0].plot(tnew, np.transpose(fids_interp), label=eps_label, alpha=0.85)
ax[1].plot(tnew, np.transpose(sts_interp), label=eps_label, alpha=0.85)
ax[1].legend(loc=(0.65, 0.15))
ax[0].set_ylabel(r'Fidelity')
ax[1].set_ylabel(r'$\mathcal{S}_{V/2}$')
ax[1].set_xlabel(r'$t$')

plt.tight_layout()
plt.savefig('./plots/dqmbs_varying_proj.pdf')
	