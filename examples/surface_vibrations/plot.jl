using AcousticPACMAN
using Plots
using LinearAlgebra

# This example reporduces the plots in figure 4 of:
#
#     Harald Ziegelwanger, Paul Reiter, "The PAC-MAN model: Benchmark case for
#     linear acoustics in computational physics," Journal of Computational
#     Physics, Volume 346, 2017, Pages 152-171, ISSN 0021-9991,
#     <https://doi.org/10.1016/j.jcp.2017.06.018>.

function surfacevibrationscatter(N, f, r, ϕ)
    ρ₀ = big"1.2041"
    c₀ = big"342.21"
    Z = ρ₀ * c₀

    k = f * 2 * pi / c₀
    r₀ = big"1.0"
    M = 100
    ic = surfacevibration(BigFloat[one(BigFloat)/10], BigFloat[])
    p = pressure(pacman(M, N, k, r₀, Z, ic))

    a = similar(ϕ, complex(typeof(r)))
    Threads.@threads :static for i in eachindex(a, ϕ)
        a[i] = p.(r, ϕ[i])
    end

    return abs.(a) ./ norm(a, Inf)
end

figs = []
for f in [big"16", big"31.5", big"63", (big"125" .* 2 .^ (0:5))...]

    r = big"2.0"
    ϕ = deg2rad.(range(big"0.0", stop = big"360", length = 1024))

    a2 = surfacevibrationscatter(2, f, r, ϕ)
    a4 = surfacevibrationscatter(4, f, r, ϕ)
    a6 = surfacevibrationscatter(6, f, r, ϕ)

    fig = plot(ϕ, a6; proj = :polar, lims = (0, 1))
    plot!(ϕ, a4; proj = :polar, lims = (0, 1))
    plot!(ϕ, a2; proj = :polar, lims = (0, 1))

    push!(figs, fig)
    display(plot(figs..., layout = (3, 3), legend = false))
end
