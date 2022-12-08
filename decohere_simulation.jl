using Plots 
using DataFrames
using CSV
using Alert
using ExponentialAction

include("./src/decohere_tools.jl")
include("./src/qmbs_tools.jl")
include("./src/spin_chain_tools.jl")


l = 4
s = 1
v = 4


sx, sy, sz, sp, sm, o, id = s_spinOps(1)


function lindblad_evolve(t, l_ops)	
	h = coupling_h(local_schecter, l, pbc=true) + field_h(1, s, l) + aniso_h(1, s, l)
	liouv = liouvillian(h, l_ops)

	nn = nematic_neel(v, s, l)
	nn_rho = nn * nn'
	nn_rho = sparse(flatten(nn * nn'))
	dnn = mapreduce((x) -> expv(x, liouv, nn_rho), hcat, t)

	nf = nematic_ferro(s, l)
	nf_rho = sparse(flatten(nf * nf'))
	dnf = mapreduce((x) -> expv(x, liouv, nf_rho), hcat, t)


	nn_fid = []
	nn_st = []
	for col in eachcol(dnn)
		append!(nn_fid, vec_expect(col, nn * nn'))
		dm = unflatten(copy(col))
		append!(nn_st, s_entgl_dm(dm, l/2, s, l))
	end

	nf_fid = []
	nf_st = []
	for col in eachcol(dnf)
		append!(nf_fid, vec_expect(col, nf * nf'))
		dm = unflatten(copy(col))
		append!(nf_st, s_entgl_dm(dm, l/2, s, l))
	end

	s_entgl_dm(unflatten(expv(1000, liouv, nn_rho)), l/2, s, l)

	return nn_fid, nn_st, nf_fid, nf_st
end


# plot!(t, real(ds[4, :]))


om = 0.5
l_ops = [
	om * op_tensp(sp, 1, l)
	]

t_end, n_steps = 50, 250
t = 0:(t_end/n_steps):t_end
nn_fid, nn_st, nf_fid, nf_st = lindblad_evolve(t, l_ops)

plot(t, real(nn_fid))
plot!(t, real(nn_st))
plot!(t, real(nf_fid))
plot!(t, real(nf_st))


df = DataFrame(t=t, nn_fid=real(nn_fid), nn_st=real(nn_st), nf_fid=real(nf_fid), nf_st=real(nf_st))
fname = "dqmbs_end_coupling_l=$l.csv"
CSV.write("./data/$fname", df)
alert("Finished evolving qmbs")
