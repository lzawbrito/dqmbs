using DMRJtensor: spinOps
using LinearAlgebra
using SparseArrays
using Arpack
using FastExpm
using Plots

include("spin_chain_tools.jl")

flatten(v) = vcat(v...) # Column-stacking convention

function unflatten(v)
	dim = round(Int, sqrt(length(v)))
	return sparse(reshape(v, (dim, dim)))
end

"""
Construct the vectorized Liovillian operator corresponding to the given 
Hamiltonian and jump operators `l_ops`.
"""
function liouvillian(h, l_ops)
	dims = size(h)
	id = sparse(I, dims...)
	h_s = sparse(h)

	# Commutator/collapse-free part of Liovillian
	l = -im * (kron(id, h_s) - kron(copy(transpose(h_s)), id))

	# Collapse terms of Liovillian
	function coll_term(l_op) 
		op = kron(conj(l_op), l_op)
		op += - (1/2) * kron(copy(transpose(l_op)) * conj(l_op), id)
		op += - (1/2) * kron(id, copy(l_op') * l_op)
		return op
	end
	l = foldl((a, l_op) -> a + coll_term(l_op), l_ops, init=l) 
	
	return l
end
