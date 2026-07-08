using DynamicalSystemsBase, ChaosTools, OrdinaryDiffEq, StaticArrays

include(joinpath(@__DIR__, "_common.jl"))

# Same 4-cycle network as benchmarks/lyapax/network.py / benchmarks/jitcode/network.jl:
# dx_i/dt = gamma*x_i + G*sum_j W_ij*x_j.
function network_rule(u, p, t)
    gamma, G = p
    du1 = gamma * u[1] + G * (u[2] + u[4])
    du2 = gamma * u[2] + G * (u[1] + u[3])
    du3 = gamma * u[3] + G * (u[2] + u[4])
    du4 = gamma * u[4] + G * (u[1] + u[3])
    return SVector(du1, du2, du3, du4)
end

# Same params as benchmarks/lyapax/network.py: dt=1e-3, n_steps=20_000,
# renorm_every=10, t_transient=5.0 -> N=2000, Δt=0.01, Ttr=5.0.
# dtmax=Δt: same fix as benchmarks/chaostools/linear_ode.jl -- this network
# is also globally stable (decays toward the origin), so without capping
# the adaptive step size the tangent dynamics get under-resolved the same
# way. See that file's comment for the full explanation.
u0 = [0.3, -0.1, 0.2, -0.4]
p0 = [-2.0, 0.5]
ds = CoupledODEs(network_rule, u0, p0; diffeq=(alg=Tsit5(), dtmax=0.01))

run_fn = () -> lyapunovspectrum(ds, 2000, 4; u0=u0, Δt=0.01, Ttr=5.0)
first_s, warm_s, exponents = time_and_run(run_fn)

emit("chaostools", "linear_network_tier3.1", exponents, first_s, warm_s)

# Same-algorithm variants: RK4() and Vern6() (the tableau lyapax's rk6_step
# implements, see lyapax.integrators.rk6_combine's docstring) at lyapax's
# own fixed dt=1e-3, adaptive=false -- a genuine same-method comparison,
# unlike the Tsit5 run above, whose adaptive step-size control means its
# timing reflects a different amount of numerical work, not just a
# different implementation of the same work.
for (alg, tool, dt_raw) in ((RK4(), "chaostools-rk4", 1e-3), (Vern6(), "chaostools-rk6", 1e-3))
    ds_fixed = CoupledODEs(network_rule, u0, p0; diffeq=(alg=alg, adaptive=false, dt=dt_raw))
    local run_fn
    run_fn = () -> lyapunovspectrum(ds_fixed, 2000, 4; u0=u0, Δt=0.01, Ttr=5.0)
    local first_s, warm_s, exponents
    first_s, warm_s, exponents = time_and_run(run_fn)
    emit(tool, "linear_network_tier3.1", exponents, first_s, warm_s)
end
