using DynamicalSystemsBase, ChaosTools, OrdinaryDiffEq, StaticArrays

include(joinpath(@__DIR__, "_common.jl"))

function rossler_rule(u, p, t)
    a, b, c = p
    du1 = -u[2] - u[3]
    du2 = u[1] + a * u[2]
    du3 = b + u[3] * (u[1] - c)
    return SVector(du1, du2, du3)
end

# Same params as benchmarks/lyapax/rossler.py: dt=1e-2, n_steps=200_000,
# renorm_every=10, t_transient=200.0 -> N=20000, Δt=0.1, Ttr=200.0.
# dtmax=Δt for consistency, see linear_ode.jl's comment.
u0 = [1.0, 1.0, 1.0]
p0 = [0.2, 0.2, 5.7]
ds = CoupledODEs(rossler_rule, u0, p0; diffeq=(alg=Tsit5(), dtmax=0.1))

run_fn = () -> lyapunovspectrum(ds, 20000, 3; u0=u0, Δt=0.1, Ttr=200.0)
first_s, warm_s, exponents = time_and_run(run_fn)

emit("chaostools", "rossler_tier1.2", exponents, first_s, warm_s)
