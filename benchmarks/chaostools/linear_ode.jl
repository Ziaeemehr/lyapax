using DynamicalSystemsBase, ChaosTools, OrdinaryDiffEq, StaticArrays

include(joinpath(@__DIR__, "_common.jl"))

function linear_rule(u, p, t)
    return SVector(-1.0 * u[1], -2.0 * u[2], -5.0 * u[3])
end

# Same params as benchmarks/lyapax/linear_ode.py: dt=1e-3, n_steps=20_000,
# renorm_every=10, t_transient=5.0 -> N=2000, Δt=0.01, Ttr=5.0.
#
# dtmax=Δt is essential here, not just a tuning knob: this system's
# trajectory decays toward the origin (unlike Lorenz/Rossler's bounded
# chaotic attractor), and once |u| approaches Tsit5's default abstol, its
# adaptive step-size control -- rightly satisfied by the now-tiny state's
# error budget -- balloons the internal step size far past Δt, silently
# under-resolving the *deviation-vector* (tangent) dynamics riding along
# with it (confirmed directly: without this cap, the reference trajectory
# itself measurably drifted/grew again near t~30 in isolation, and the
# recovered spectrum came out near [0,0,0] instead of [-1,-2,-5]). Capping
# dtmax at the Benettin renormalization interval forces the tangent
# dynamics to be resolved at the same cadence lyapax's fixed dt=1e-3 and
# jitcode's dopri5 default already do implicitly.
u0 = [0.3, -0.2, 0.5]
ds = CoupledODEs(linear_rule, u0, nothing; diffeq=(alg=Tsit5(), dtmax=0.01))

run_fn = () -> lyapunovspectrum(ds, 2000, 3; u0=u0, Δt=0.01, Ttr=5.0)
first_s, warm_s, exponents = time_and_run(run_fn)

emit("chaostools", "linear_ode_tier0.1", exponents, first_s, warm_s)
