using DMRJtensor: spinOps
using LinearAlgebra
using SparseArrays
using Arpack
using FastExpm
using Plots

include("spin_chain_tools.jl")

# HAMILTONIANS -----------------------------------------------------------------

"""
The local coupling for sites i and j in the general case of an exactly scarred
Hamiltonian. 
"""
function local_essh(i, j, s, p::Int, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 
	
	k = round(Int, 2s - p)
	h = op_tensp(sp, i, l)^k * op_tensp(sm, j, l)^k
	h += (factorial(k) / factorial(p))^2 * op_tensp(sp, i, l)^p * op_tensp(sm, j, l)^p 
	return h + h'
end

"""
SU(2) spin-1 eq (1) in Hamiltonian from Schecter 2019. 
"""
function local_schecter(i, j, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(1) 
	return op_tensp(sx, i, l) * op_tensp(sx, j, l) + op_tensp(sy, i, l) * op_tensp(sy, j, l)
end

"""
Sum over local coupling terms given by `local_h`
"""
function coupling_h(local_h, l; pbc=true)
	h = sum((i) -> local_h(i, i + 1, l), 1:l - 1)
	if pbc
		h += local_h(l, 1, l)
	end
	return h
end


"""
The term in the Hamiltonian corresponding to coupling to an external field.
"""
function field_h(field_strength, s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 
	h = reduce((h, i) ->field_strength * op_tensp(sz, i, l) + h, 2:l; 
				init=field_strength * op_tensp(sz, 1, l))
	return h
end

function aniso_h(strength, s, l)
	sx, sy, sz, sp, sm, o, id = s_spinOps(s) 
	return strength * sum((i) -> op_tensp(sz, i, l)^2, 1:l)
end

"""
Staggered rhombic anisotropy Hamiltonian, eq (10). 
"""
function sra_h(s, l)
	return (1/2) * (total_sp(s, l) + total_sm(s, l))
end

# STATES -----------------------------------------------------------------------

"""
A down state in the Spin-S algebra.
"""
function single_down(s)
	k = spzeros(2s + 1)
	k[end] = 1
	return sparse(k)
end

"""
An up state in the Spin-S algebra.
"""
function single_up(s)
	k = spzeros(2s + 1)
	k[1] = 1
	return sparse(k)
end

"""
All down polarized state in a Spin-S `n` particle system.
"""
function many_down(s, l)
	k = spzeros(round(Int, (2s+1)^l))
	k[end] = 1
	return sparse(k)
end

"""
Scheter eq (2)
"""
function scar(n, v, s, l)
	normalization = sqrt(factorial(v - n) / (factorial(n) * factorial(v)))
	return normalization * total_sp(s, l)^n * many_down(s, l)
end

"""
Nematic neel state; i.e., the scarred state of `local_schecter`
Schecter eq (12) 
"""
function nematic_neel(v, s, l)
	sum((n) -> sqrt(binomial(v, n)) * scar(n, v, s, l) / 2^(v / 2), 0:v)
end

"""
Nematic ferromagnetic state
"""
function nematic_ferro(s, l)
	up = single_up(s)
	down = single_down(s)
	return foldl((s1, s2) -> kron(s1, (up - down) / sqrt(2)), 2:l, init=(up - down) / sqrt(2))
end 



