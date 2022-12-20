import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D


df = pd.read_csv('./data/dqmbs_nn_vs_nf_l=4.csv')

t = df['t'].to_numpy()

nn_fid = df['nn_fid'].to_numpy()
nn_st = df['nn_st'].to_numpy()	
nn_fid_noproj = df['nn_fid_noproj'].to_numpy()
nn_st_noproj = df['nn_st_noproj'].to_numpy()

nf_fid = df['nf_fid'].to_numpy()
nf_st = df['nf_st'].to_numpy()	
nf_fid_noproj = df['nf_fid_noproj'].to_numpy()
nf_st_noproj = df['nf_st_noproj'].to_numpy()


# Interpolation so graph looks a little nicer. Just smooths it a bit.
tnew = np.linspace(0, t[-1], 500) 
def interpolate(t, f):
	return interp1d(t, f, kind='cubic')(tnew)

nn_fid_new = interpolate(t, nn_fid)
nn_st_new = interpolate(t, nn_st)
nn_fid_noproj_new = interpolate(t, nn_fid_noproj)
nn_st_noproj_new = interpolate(t, nn_st_noproj)
nf_fid_new = interpolate(t, nf_fid)
nf_st_new = interpolate(t, nf_st)
nf_fid_noproj_new = interpolate(t, nf_fid_noproj)
nf_st_noproj_new = interpolate(t, nf_st_noproj)


fid, ax = plt.subplots(nrows=4, figsize=(4,6.5))
grey_alpha = 0.65
global_alpha = 0.85

ax[0].plot(tnew, nn_fid_new, color='black', alpha=grey_alpha)
ax[0].plot(tnew, nn_fid_noproj_new, color='black', linestyle='dashed', alpha=grey_alpha)
ax[0].set_ylabel(r'Fid.')

ax[1].plot(tnew, nn_st_new, color='black', alpha=grey_alpha)
ax[1].plot(tnew, nn_st_noproj_new, color='black', linestyle='dashed', alpha=grey_alpha)
ax[1].set_ylim(-0.3, 1.95)
ax[1].set_ylabel(r'$\mathcal{S}_{V/2}$')

ax[2].plot(tnew, nf_fid_new, color='cornflowerblue', alpha=global_alpha)
ax[2].plot(tnew, nf_fid_noproj_new, color='cornflowerblue', linestyle='dashed', alpha=global_alpha)
ax[2].set_ylabel(r'Fid.')

ax[3].plot(tnew, nf_st_new, color='cornflowerblue', alpha=global_alpha)
ax[3].plot(tnew, nf_st_noproj_new, color='cornflowerblue', linestyle='dashed', alpha=global_alpha)
ax[3].set_ylim(-0.1, 1.99)

ax[3].set_ylabel(r'$\mathcal{S}_{V/2}$')
ax[3].set_xlabel(r'$t$')

custom_lines = [
	Line2D([0], [0], color='black'),
	Line2D([0], [0], color='black', linestyle='dashed'),
	Line2D([0], [0], color='cornflowerblue'),
	Line2D([0], [0], color='cornflowerblue', linestyle='dashed')
	]


ax[3].legend(custom_lines, [
		r'NN with $P_{ij}$',
		r'NN w/out $P_{ij}$',
		r'NF with $P_{ij}$',
		r'NF w/out $P_{ij}$',
	])
plt.tight_layout()
plt.savefig('./plots/dqmbs_proj_noproj.pdf')