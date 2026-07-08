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

# Same-algorithm variants: RK4() and Vern6() (the tableau lyapax's rk6_step
# implements, see lyapax.integrators.rk6_combine's docstring) at lyapax's
# own fixed dt=1e-2, adaptive=false -- a genuine same-method comparison,
# unlike the Tsit5 run above, whose adaptive step-size control means its
# timing reflects a different amount of numerical work, not just a
# different implementation of the same work.
for (alg, tool, dt_raw) in ((RK4(), "chaostools-rk4", 1e-2), (Vern6(), "chaostools-rk6", 1e-2))
    ds_fixed = CoupledODEs(rossler_rule, u0, p0; diffeq=(alg=alg, adaptive=false, dt=dt_raw))
    local run_fn
    run_fn = () -> lyapunovspectrum(ds_fixed, 20000, 3; u0=u0, Δt=0.1, Ttr=200.0)
    local first_s, warm_s, exponents
    first_s, warm_s, exponents = time_and_run(run_fn)
    emit(tool, "rossler_tier1.2", exponents, first_s, warm_s)
end
