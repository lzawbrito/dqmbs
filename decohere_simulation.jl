using Plots 
using DataFrames
using CSV
using Alert
using ExponentialAction
using MatrixEquations

include("./src/decohere_tools.jl")
include("./src/qmbs_tools.jl")
include("./src/spin_chain_tools.jl")


sx, sy, sz, sp, sm, o, id = s_spinOps(1)


# function lindblad_evolve(t, l_ops)	
# 	h = coupling_h(local_schecter, l, pbc=true) + field_h(1, s, l) + aniso_h(1, s, l)
# 	liouv = liouvillian(h, l_ops)

# 	nn = nematic_neel(v, s, l)
# 	nn_rho = nn * nn'
# 	nn_rho = sparse(flatten(nn * nn'))
# 	dnn = mapreduce((x) -> expv(x, liouv, nn_rho), hcat, t)

# 	nf = nematic_ferro(s, l)
# 	nf_rho = sparse(flatten(nf * nf'))
# 	dnf = mapreduce((x) -> expv(x, liouv, nf_rho), hcat, t)

# 	nn_fid = []
# 	nn_st = []
# 	for col in eachcol(dnn)
# 		append!(nn_fid, vec_expect(col, nn * nn'))
# 		dm = unflatten(copy(col))
# 		append!(nn_st, s_entgl_dm(dm, l/2, s, l))
# 	end

# 	nf_fid = []
# 	nf_st = []
# 	for col in eachcol(dnf)
# 		append!(nf_fid, vec_expect(col, nf * nf'))
# 		dm = unflatten(copy(col))
# 		append!(nf_st, s_entgl_dm(dm, l/2, s, l))
# 	end

# 	s_entgl_dm(unflatten(expv(1000, liouv, nn_rho)), l/2, s, l)

# 	return nn_fid, nn_st, nf_fid, nf_st
# end

function lindblad_evolve(t, state, l_ops)	
	h = coupling_h(local_schecter, l, pbc=true) + field_h(1, s, l) + aniso_h(1, s, l)
	liouv = liouvillian(h, l_ops)

	state_rho = state * state'
	state_rho = sparse(flatten(state * state'))
	dstate = mapreduce((x) -> expv(x, liouv, state_rho), hcat, t)

	state_fid = []
	state_st = []
	for col in eachcol(dstate)
		append!(state_fid, vec_expect(col, state * state'))
		dm = unflatten(copy(col))
		append!(state_st, s_entgl_dm(dm, l/2, s, l))
	end

	s_entgl_dm(unflatten(expv(1000, liouv, state_rho)), l/2, s, l)

	return real(state_fid), real(state_st)
end

# plot!(t, real(ds[4, :]))

l = 4
s = 1
v = 4

n = 0
nn = nematic_neel(v, s, l)
nf = nematic_ferro(s, l)

t_end, n_steps = 20, 150
t = 0:(t_end/n_steps):t_end

function perturbation(h, h_prime, s, l; proj_frac=1) 
	pert = h_prime 
	for i in 1:2:l 
		j = i + 1
		proj = (proj_frac * embed_proj(i, j, s, l) - (1 - proj_frac) * sparse(I, (2s + 1)^l, (2s + 1)^l))
		pert += proj * h(i, j) * proj
	end
	return pert
end

function pert_without_proj(h, h_prime, s, l) 
	pert = h_prime 
	for i in 1:2:l 
		j = i + 1
		pert += h(i, j)
	end
	return pert
end

function higher_order(i, j, l)
	op =  op_tensp(sx, i, l) * op_tensp(sx, j, l)
	# op += (op_tensp(sy, i, l) * op_tensp(sy, j, l))
	# op += (op_tensp(sz, i, l) * op_tensp(sz, j, l))
	return op^2
end

pert = perturbation((i, j) -> higher_order(i, j, l),
		0 * embed_proj(1, 2, s, l), s, l)
no_proj_pert = pert_without_proj((i, j) -> higher_order(i, j, l),
		0 * embed_proj(1, 2, s, l), s, l)

println("Working on NN with projector.")	
nn_fid, nn_st = lindblad_evolve(t, nn, [pert])
println("Working on NN without projector.")	
nn_fid_noproj, nn_st_noproj = lindblad_evolve(t, nn, [no_proj_pert])
println("Working on NF with projector.")	
nf_fid, nf_st = lindblad_evolve(t, nf, [pert])
println("Working on NF without projector.")	
nf_fid_noproj, nf_st_noproj = lindblad_evolve(t, nf, [no_proj_pert])


df = DataFrame(t=t,
			   nn_fid=nn_fid,
			   nn_st=nn_st,
			   nn_fid_noproj=nn_fid_noproj,
			   nn_st_noproj=nn_st_noproj,
			   nf_fid=nf_fid,
			   nf_st=nf_st,
			   nf_fid_noproj=nf_fid_noproj,
			   nf_st_noproj=nf_st_noproj)

fname = "dqmbs_nn_vs_nf_l=$l.csv"
CSV.write("./data/$fname", df)


# data = Dict()
# for frac in 0:1/4:1
# 	println("Working on ϵ = $frac")
	# pert = perturbation((i, j) -> op_tensp(sz, i, l)^3 * op_tensp(sz, j, l)^3, 
	# 	embed_proj(1, 2, s, l), s, l)
	# pert = perturbation((i, j) -> higher_order(i, j, l),
	# 	0*embed_proj(1, 2, s, l), s, l, proj_frac=frac)

	# pert_wout_proj = pert_without_proj((i, j) -> higher_order(i, j, l),
	# 	0*embed_proj(1, 2, s, l), s, l)


	# pert = perturbation((i, j) -> op_tensp(sx, i, l), 0 * embed_proj(1, 2, s, l), s, l)

	# Collapse operator
	# om = 1
	# l_op = om * pert


	# norm(comm(pert, embed_proj(3, 4, s, l)))

	# Evolve
# 	state_fid, state_st = lindblad_evolve(t, state, [l_op])
# 	data[string(frac)] = (real(state_fid), real(state_st))
# end
# println("Done!")

# times = []
# epsilon = []
# fids = []
# sts = []
# for (key, value) in data
# 	append!(times, t)
# 	append!(epsilon, ones(length(t)) * parse(Float64, key))
# 	append!(fids, real(value[1]))
# 	append!(sts, real(value[2]))
# end



# df = DataFrame(t=times, epsilon=epsilon, fid=fids, st=sts)
# fname = "dqmbs_varying_proj_l=$l.csv"
# CSV.write("./data/$fname", df)
# alert("Finished evolving qmbs")
