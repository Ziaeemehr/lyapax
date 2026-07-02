using DynamicalSystemsBase, ChaosTools, StaticArrays

include(joinpath(@__DIR__, "_common.jl"))

# Tier 0.2/0.3: same params as benchmarks/lyapax/maps.py.

logistic_rule(u, p, n) = SVector(4.0 * u[1] * (1.0 - u[1]))
u0_logistic = [0.4]
ds = DeterministicIteratedMap(logistic_rule, u0_logistic, nothing)
run_fn = () -> lyapunovspectrum(ds, 500_000, 1; u0=u0_logistic, Δt=1, Ttr=1000)
first_s, warm_s, exponents = time_and_run(run_fn)
emit("chaostools", "logistic_map_tier0.2", exponents, first_s, warm_s)

tent_rule(u, p, n) = SVector(u[1] < 0.5 ? 2.0 * u[1] : 2.0 * (1.0 - u[1]))
u0_tent = [0.4]
ds = DeterministicIteratedMap(tent_rule, u0_tent, nothing)
run_fn = () -> lyapunovspectrum(ds, 500_000, 1; u0=u0_tent, Δt=1, Ttr=1000)
first_s, warm_s, exponents = time_and_run(run_fn)
emit("chaostools", "tent_map_tier0.2", exponents, first_s, warm_s)

function henon_rule(u, p, n)
    a, b = p
    return SVector(1.0 - a * u[1]^2 + u[2], b * u[1])
end
u0_henon = [0.1, 0.1]
p_henon = [1.4, 0.3]
ds = DeterministicIteratedMap(henon_rule, u0_henon, p_henon)
run_fn = () -> lyapunovspectrum(ds, 200_000, 2; u0=u0_henon, Δt=1, Ttr=1000)
first_s, warm_s, exponents = time_and_run(run_fn)
emit("chaostools", "henon_map_tier0.3", exponents, first_s, warm_s)
