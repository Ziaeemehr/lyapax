using DynamicalSystemsBase, ChaosTools, OrdinaryDiffEq, StaticArrays

include(joinpath(@__DIR__, "_common.jl"))

function lorenz_rule(u, p, t)
    sigma, rho, beta = p
    du1 = sigma * (u[2] - u[1])
    du2 = u[1] * (rho - u[3]) - u[2]
    du3 = u[1] * u[2] - beta * u[3]
    return SVector(du1, du2, du3)
end

# Same params as benchmarks/lyapax/lorenz.py: dt=1e-2, n_steps=50_000,
# renorm_every=10, t_transient=100.0 -> N=5000 Benettin steps, Δt=0.1, Ttr=100.0.
# dtmax=Δt for consistency with the other ODE benchmarks here (see
# linear_ode.jl's comment) -- doesn't change the result for Lorenz, whose
# bounded chaotic attractor never approaches a fixed point, but keeps the
# solver settings uniform across systems rather than case-by-case.
u0 = [1.0, 1.0, 1.0]
p0 = [10.0, 28.0, 8.0 / 3.0]
ds = CoupledODEs(lorenz_rule, u0, p0; diffeq=(alg=Tsit5(), dtmax=0.1))

run_fn = () -> lyapunovspectrum(ds, 5000, 3; u0=u0, Δt=0.1, Ttr=100.0)
first_s, warm_s, exponents = time_and_run(run_fn)

emit("chaostools", "lorenz_tier1.1", exponents, first_s, warm_s)

# Same-algorithm variants: RK4() and Vern6() (the tableau lyapax's rk6_step
# implements, see lyapax.integrators.rk6_combine's docstring) at lyapax's
# own fixed dt=1e-2, adaptive=false -- a genuine same-method comparison,
# unlike the Tsit5 run above, whose adaptive step-size control means its
# timing reflects a different amount of numerical work, not just a
# different implementation of the same work.
for (alg, tool, dt_raw) in ((RK4(), "chaostools-rk4", 1e-2), (Vern6(), "chaostools-rk6", 1e-2))
    ds_fixed = CoupledODEs(lorenz_rule, u0, p0; diffeq=(alg=alg, adaptive=false, dt=dt_raw))
    local run_fn
    run_fn = () -> lyapunovspectrum(ds_fixed, 5000, 3; u0=u0, Δt=0.1, Ttr=100.0)
    local first_s, warm_s, exponents
    first_s, warm_s, exponents = time_and_run(run_fn)
    emit(tool, "lorenz_tier1.1", exponents, first_s, warm_s)
end
