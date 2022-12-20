using DMRJtensor: spinOps
using LinearAlgebra
using SparseArrays
using Arpack
using FastExpm
using ExponentialAction

# Utilities

normalize(v) = v / norm(v)
overlap(v1, v2) = dot(normalize(v1), normalize(v2))
comm(o1, o2) = o1 * o2 - o2 * o1
fidelity(s1, s2) = abs(dot(s1, s2))^2

"""
Evolve the state `s` to time `t` under Hamiltonian `h`. Can use with 
Lindbladian if you change `L -> (im) * L` to remove (-i) factor
"""
function evolve(t, s, h; threshold=1e-6, nonzero_tol=1e-5)
	if t == 0 return s end 
	return fastExpm(-im * t * h; threshold=threshold, nonzero_tol=nonzero_tol) * s
end

"""
Take the expectation value of the given observable `obs` with respect to the
state density matrix `rho`.
"""
function expect(rho, obs)
	return tr(rho * obs)
end

function vec_expect(rho, obs)
	return rho' * flatten(obs')
end

"""
Take the partial trace with respec to sites 1 -`trace_size` in the chain
"""
function partial_trace(rho, trace_size, s, l)
	trace_dim = round(Int, (2s + 1)^trace_size)
	subsystem_dim = round(Int, (2s + 1)^(l - trace_size))
	id = sparse(I, subsystem_dim, subsystem_dim)

	return sum((i) -> begin
		z = zeros(trace_dim)
		z[i] = 1
		kron(id, z') * rho * kron(id, z)
	end, 1:trace_dim)
end

"""
Eigenvalues 1-(dim(`m`) - 1) of the matrix `m`
"""
function full_eigs(m)
	lmsize = round(Int, size(m)[1])
	a = eigs(m, nev=lmsize - 2, which=:LM)
	return a[1] 
end

"""
Von Neumann entropy
"""
function s_vn(a)
	evals = full_eigs(a)

	function entropy(e)
		if real(e) <= 0
			return 0
		end 
		return e * log(e)
	end

	return - sum(entropy, evals)
end

"""
Entanglement entropy
"""
function s_entgl(k, trace_size, s, l)
	return s_vn(partial_trace(k * k', trace_size, s, l))
end
"""

Entanglement entropy
"""
function s_entgl_dm(dm, trace_size, s, l)
	return s_vn(partial_trace(dm, trace_size, s, l))
end


# OPERATORS --------------------------------------------------------------------

"""
Returns sparse versions of 
sx, sy, sz, sp, sm, o, id 
"""
s_spinOps(s) = map(sparse, spinOps(Float64(s)))

"""
Put operator in tensor product space (`tensp`)
"""
function op_tensp(op::SparseMatrixCSC, i::Int, l::Int)::SparseMatrixCSC
	new_op = I(size(op)[1])

	if i == 1
		new_op = op
	end
	for k in 2:l
		if k == i
			new_op = kron(new_op, op)
		else
			new_op = kron(new_op, I(size(op)[1]))
		end
	end
	return new_op
end

"""
Spin raising operator in emergent SU(2) algebra; Schecter 2019 eq (3).
"""
function total_sp(s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 

	return (1/2) * sum((i) -> exp(im * i * pi) * op_tensp(sp, i, l)^2, 1:l)
end

"""
Spin lowering operator in emergent SU(2) algebra; Schecter 2019 eq (3).
"""
function total_sm(s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 

	return sum((i) -> (1/2) * exp(im * i * pi)* op_tensp(sm, i, l)^2, 1:l)
end

function total_sz(s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 

	return sum((i) -> (1/2) * op_tensp(sz, i, l), 1:l)
end

function total_sx(s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 

	return sum((i) -> (1/2) * op_tensp(sx, i, l), 1:l)
end

function total_sy(s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 

	return sum((i) -> (1/2) * op_tensp(sy, i, l), 1:l)
end

"""
Spin squared operator for emergent SU(2) algebra
"""
function total_ss(s, l)
	t1 = (1/2) * (total_sp(s, l) * total_sm(s, l) + total_sm(s, l) * total_sp(s, l)) 
	t2 = (1/4) * comm(total_sp(s, l), total_sm(s, l))^2 
	return t1 + t2
end


"""
Embedded Hamiltonian projector operator. 
"""
function embed_proj(i, j, s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 
	p = sparse(I, (2s+1)^l, (2s+1)^l) - (3/4) * op_tensp(sz, i, l)^2 * op_tensp(sz, j, l)^2 
	p += (1/8) * op_tensp(sp, i, l)^2 * op_tensp(sm, j, l)^2
	p += (1/8) * op_tensp(sm, i, l)^2 * op_tensp(sp, j, l)^2
	p += - (1/4) * op_tensp(sz, i, l) * op_tensp(sz, j, l)
	return p
end