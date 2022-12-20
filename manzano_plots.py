import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd 

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath}\n\\usepackage{physics}')

df = pd.read_csv('./data/decohere_2level_manzano.csv')
t = df['t'].to_numpy()
up = df['up'].to_numpy()
down = df['down'].to_numpy()

fig, ax = plt.subplots(nrows=2)
ax[0].plot(t, up**2, color='black', alpha=0.65)
ax[1].plot(t, down**2, color='black', alpha=0.65)
for a in ax:
	a.set_ylim(-0.1, 1.1)
ax[1].set_xlabel('$t$')
ax[0].set_ylabel(r'$|\braket{+1}{\psi}|^2$')
ax[1].set_ylabel(r'$|\braket{-1}{\psi}|^2$')
# ax[0].set_title('Fidelity of Decaying Two-Level System')

plt.tight_layout()
plt.savefig('./plots/manzano_decoherence.pdf')
plt.savefig('./notes/final-report/plots/manzano_decoherence.pdf')
