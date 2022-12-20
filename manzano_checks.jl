using Plots 
using DataFrames
using CSV
using Alert
using ExponentialAction

include("./src/decohere_tools.jl")
include("./src/spin_chain_tools.jl")


begin
	sx, sy, sz, sp, sm, o, id = s_spinOps(1/2)
	e = 1
	h_free = sparse([e 0 ; 0 0])
	om = 1
	h_coup = sparse([0 om; om 0])
	
	h = h_free + h_coup
	t = 0:(5/100):5
	s_0 = [1 ; 0]
	s = map((x) -> abs(s_0' * evolve(x, s_0, h))^2, t) # Matches Manzano result
end

# Check Manzano decoherence result
function decohere_2level()
	t_end = 50
	n_steps = 500
	e = 1
	h_free = sparse([e 0 ; 0 0])
	om = 1
	h_coup = sparse([0 om; om 0])

	h = h_free + h_coup
	gam = 1 * sqrt(0.1)
	liouv = liouvillian(h, [gam * sm])

	t = 0:(t_end/n_steps):t_end
	s_0 = flatten([1 0; 0 0])
	ds = mapreduce((x) -> expv(x, liouv, s_0), hcat, t) # Matches Manzano result

	return t, real(ds[1, :]), real(ds[4, :])
end

t, up, down = decohere_2level()
df = DataFrame(t=t, up=up, down=down)
CSV.write("./data/decohere_2level_manzano.csv", df)