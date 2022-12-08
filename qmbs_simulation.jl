using DataFrames
using CSV

include("./src/qmbs_tools.jl")

l = 4
s = 1
v = 4

# Checks 
printstyled("Check: ", bold=true)
print("Nematic Neél eigenstate of staggered rhombic anisotropy Hamiltonian")
k1 = nematic_neel(v, s, l)
k2 = sra_h(s, l) * k1
print(" -- ")
if round(real(overlap(k1, k2)), digits=14) == 1 
	printstyled("True", color=:green, bold=true)
else 
	printstyled("False", color=:red, bold=true)
end
print("\n\n")


h = coupling_h(local_schecter, l) + field_h(1, s, l) + aniso_h(1, s, l)

printstyled("Check: ", bold=true)
print("Scar state eigenstate of Schecter non-integrable Hamiltonian")

s1 = scar(1, v, s, l)
s2 = h * s1
round.(s2, digits=15)
print(" -- ")
if round(real(overlap(s1, s2)), digits=14) == 1
	printstyled("True", color=:green, bold=true)
else 
	printstyled("False", color=:red, bold=true)
end
print("\n\n")


# Simulation
"""
Compute fidelity and entanglement entropy at given times.
"""
function observables(times, k, h)
	kt = map((t) -> evolve(t * π, k, h), times)
	
	st = map((state) -> s_entgl(state, l/2, s, l), kt)
	ft = map((s) -> fidelity(s, k), kt)
	return ft, st
end

length = 2
n_steps = 200
times = 0:(length/n_steps):length
nn = nematic_neel(v, s, l)
nf = nematic_ferro(s, l)

nn_ft, nn_st = observables(times, nn, h)
nf_ft, nf_st = observables(times, nf, h)

df_sim = DataFrame(times=times, 
					 nn_ft=real(nn_ft),
					 nf_ft=real(nf_ft),
					 nn_st=real(nn_st),
					 nf_st=real(nf_st))

fn = CSV.write("schecter_data.csv", df_sim)