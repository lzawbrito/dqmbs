import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Read csvs
dephase = pd.read_csv('./data/dqmbs_dephasing_l=4.csv')
t = dephase['t'].to_numpy()
dephase_nn_fid = dephase['nn_fid'].to_numpy()
dephase_nn_st  = dephase['nn_st'].to_numpy()
dephase_nf_fid = dephase['nf_fid'].to_numpy()
dephase_nf_st  = dephase['nf_st'].to_numpy()

damp = pd.read_csv('./data/dqmbs_damping_l=4.csv')
t = damp['t'].to_numpy()
damp_nn_fid = damp['nn_fid'].to_numpy()
damp_nn_st  = damp['nn_st'].to_numpy()
damp_nf_fid = damp['nf_fid'].to_numpy()
damp_nf_st  = damp['nf_st'].to_numpy()

end = pd.read_csv('./data/dqmbs_end_coupling_l=4.csv')
end_t = damp['t'].to_numpy()
end_nn_fid = end['nn_fid'].to_numpy()
end_nn_st  = end['nn_st'].to_numpy()
end_nf_fid = end['nf_fid'].to_numpy()
end_nf_st  = end['nf_st'].to_numpy()

# Plot fidelity
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(6, 2)) 
ax1.plot(end_t, end_nn_fid, color='plum')
ax2.plot(end_t, end_nf_fid, color='plum', label="Single-site")
ax1.plot(t, dephase_nn_fid, color='black')
ax2.plot(t, dephase_nf_fid, color='black', label="Dephasing")
ax1.plot(t, damp_nn_fid, color='cornflowerblue')
ax2.plot(t, damp_nf_fid, color='cornflowerblue', label="Damping")
for ax in (ax1, ax2): 
	ax.label_outer()
	ax.set_xlim(0, 20)
ax2.legend(frameon=False, loc='upper right')
ax1.set_xlabel('$t$')
ax2.set_xlabel('$t$')
ax2.set_ylabel('Fidelity')
ax1.set_title("Nematic N\\'eel")
ax2.set_title("Nematic FM")
plt.savefig('./notes/final-report-draft/plots/fidelity.pdf')

ax1.cla()
ax2.cla()

# Plot entanglement entropy
ax1.plot(end_t, end_nn_st, color='plum')
ax2.plot(end_t, end_nf_st, color='plum', label="Single-site")
ax1.plot(t, dephase_nn_st, color='black')
ax2.plot(t, dephase_nf_st, color='black', label="Dephasing")
ax1.plot(t, damp_nn_st, color='cornflowerblue')
ax2.plot(t, damp_nf_st, color='cornflowerblue', label="Damping")
for ax in (ax1, ax2): 
	ax.label_outer()
	ax.set_xlim(0, 100)
ax2.legend(frameon=False, loc='center right')
ax1.set_xlabel('$t$')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$\mathcal{S}_{V/2}$')

ax1.set_title("Nematic N\\'eel")
ax2.set_title("Nematic FM")
plt.savefig('./notes/final-report-draft/plots/entanglement_entropy.pdf')