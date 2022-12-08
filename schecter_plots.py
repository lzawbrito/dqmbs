import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

df = pd.read_csv("./schecter_data.csv")
times = df['times'].to_numpy()
nn_ft = df['nn_ft'].to_numpy()
nf_ft = df['nf_ft'].to_numpy()
fig, ax = plt.subplots() 
ax.plot(times, nf_ft, color='cornflowerblue', label='Nematic FM')
ax.plot(times, nn_ft, color='black', label="Nematic N\\'eel")
plt.legend(loc='upper right', fancybox=True, framealpha=0.7)
ax.set_xlabel('$\pi t / h$')
ax.set_ylabel('Fidelity')

plt.savefig('./notes/fidelity.pdf')

nn_st = df['nn_st'].to_numpy()
nf_st = df['nf_st'].to_numpy()
fig, ax = plt.subplots() 
ax.plot(times, nf_st, color='cornflowerblue', label='Nematic FM')
ax.plot(times, nn_st, color='black', label="Nematic N\\'eel")
plt.legend(loc='upper right', fancybox=True, framealpha=0.7)
ax.set_xlabel('$t$')
ax.set_ylabel('$\mathcal{S}_{V/2}$')

plt.savefig('./notes/ee.pdf')