using DynamicalSystemsBase, ChaosTools, OrdinaryDiffEq

include(joinpath(@__DIR__, "_common.jl"))

# Same dense (all-to-all) Kuramoto network as
# benchmarks/lyapax/network_scaling.py, same RK4 fixed-step algorithm and
# dt=1e-2, k=5 partial spectrum, n_steps=200, no transient.
function kuramoto_rule!(du, u, p, t)
    omega, n = p
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            if j != i
                s += sin(u[j] - u[i])
            end
        end
        du[i] = omega[i] + s
    end
    return nothing
end

# Only d=50 and d=200 are attempted here, not lyapax's full d=50/200/1000/2000
# sweep. Directly measured (throwaway script, not committed): a single
# build+run at d=50 took 13.5s and at d=200 took 137s -- worse than
# quadratic growth -- and d=1000 was killed after running 14+ minutes
# without finishing. The cause: ChaosTools.jl's tangent propagation forms
# the *full* d x d Jacobian via ForwardDiff every step regardless of k,
# unlike lyapax's jax.jvp, which only ever computes the k=5 requested
# directions. That makes d=1000+ impractical to include here at all, which
# is itself the point of this tier -- see docs/background/benchmarks.md's
# "Network-size scaling" section and docs/background/motivation.md's
# jitcode/ChaosTools.jl bullets.
for n in (50, 200)
    omega = collect(range(-1.0, 1.0, length=n))
    u0 = collect(range(0.0, 2pi, length=n + 1)[1:n])
    local ds, run_fn, first_s, warm_s, exponents
    ds = CoupledODEs(kuramoto_rule!, u0, (omega, n); diffeq=(alg=RK4(), adaptive=false, dt=1e-2))
    run_fn = () -> lyapunovspectrum(ds, 200, 5; u0=u0, Δt=0.1, Ttr=0.0)
    first_s, warm_s, exponents = time_and_run(run_fn)
    emit("chaostools-rk4", "kuramoto_scaling_d$(n)", exponents, first_s, warm_s)
end
