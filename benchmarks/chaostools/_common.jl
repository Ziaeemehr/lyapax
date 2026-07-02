"""
Shared plumbing for the ChaosTools.jl side of notes/benchmark_report.md.

No JSON package dependency -- payloads here are simple enough (a flat
object of numbers/strings/arrays) to hand-serialize, matching the schema
`benchmarks/lyapax/_common.py` and `benchmarks/jitcode(dde)/_common.py`
emit: {"tool", "system", "exponents", "first_call_s", "warm_s", ...}.
"""

function json_number(x::AbstractFloat)
    isfinite(x) ? repr(x) : "null"
end
json_number(x::Integer) = repr(x)

function json_array(xs)
    "[" * join([json_number(Float64(x)) for x in xs], ", ") * "]"
end

function emit(tool::String, system::String, exponents, first_s::Float64, warm_s::Float64; extra=Dict{String,Float64}())
    fields = [
        "\"tool\": \"$tool\"",
        "\"system\": \"$system\"",
        "\"exponents\": " * json_array(exponents),
        "\"first_call_s\": " * json_number(first_s),
        "\"warm_s\": " * json_number(warm_s),
    ]
    for (k, v) in extra
        push!(fields, "\"$k\": " * json_number(v))
    end
    println("{" * join(fields, ", ") * "}")
end

function time_and_run(f)
    t0 = time()
    result = f()
    first_s = time() - t0

    t0 = time()
    result = f()
    warm_s = time() - t0

    return first_s, warm_s, result
end
