import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

df = pd.read_csv("./schecter_data.csv")
times = df['times'].to_numpy()
nn_ft = df['nn_ft'].to_numpy()
nf_ft = df['nf_ft'].to_numpy()
fig, ax = plt.subplots(nrows=2) 

ax[0].plot(times, nf_ft, color='cornflowerblue', label='Nematic FM', alpha=0.85)
ax[0].plot(times, nn_ft, color='black', label="Nematic N\\'eel", alpha=0.65)
ax[0].set_xlabel('$\pi t / h$')
ax[0].set_ylabel('Fidelity')

nn_st = df['nn_st'].to_numpy()
nf_st = df['nf_st'].to_numpy()
ax[1].plot(times, nf_st, color='cornflowerblue', label='Nematic FM', alpha=0.85)
ax[1].plot(times, nn_st, color='black', label="Nematic N\\'eel", alpha=0.65)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$\mathcal{S}_{V/2}$')

ax[1].legend(loc=(0.55, 0.15))
plt.tight_layout()
plt.savefig('./plots/schecter_dqmbs.pdf')
plt.savefig('./notes/final-report/plots/schecter_dqmbs.pdf')