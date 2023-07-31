using AcousticPACMAN
using Plots

# This example reproduces the plots in figure 10 of:
#
#     Harald Ziegelwanger, Paul Reiter, "The PAC-MAN model: Benchmark case for
#     linear acoustics in computational physics," Journal of Computational
#     Physics, Volume 346, 2017, Pages 152-171, ISSN 0021-9991,
#     <https://doi.org/10.1016/j.jcp.2017.06.018>.

function planewavescatter(N, f, r, ϕ)
    ρ₀ = 1.2041
    c₀ = 342.21
    Z = ρ₀ * c₀

    k = f * 2 * pi / c₀
    r₀ = 1.0
    M = 100
    ϕₛ = pi / 4.0
    ic = planewave(M, ϕₛ)
    p = pressure(pacman(M, N, k, r₀, Z, ic))

    a = similar(ϕ, complex(typeof(r)))
    Threads.@threads :static for i in eachindex(a, ϕ)
        a[i] = p.(r, ϕ[i])
    end

    return abs.(a) ./ maximum(abs.(a))
end

figs = []
for f in [16.0, 31.5, 63, (125.0 .* 2 .^ (0:5))...]
    r = 2.0
    ϕ = deg2rad.(range(0.0, stop = 360, length = 1024))

    a2 = planewavescatter(2, f, r, ϕ)
    a4 = planewavescatter(4, f, r, ϕ)
    a6 = planewavescatter(6, f, r, ϕ)

    fig = plot(ϕ, a6; proj = :polar, lims = (0, 1))
    plot!(ϕ, a4; proj = :polar, lims = (0, 1))
    plot!(ϕ, a2; proj = :polar, lims = (0, 1))

    push!(figs, fig)
    display(plot(figs..., layout = (3, 3), legend = false))
end
