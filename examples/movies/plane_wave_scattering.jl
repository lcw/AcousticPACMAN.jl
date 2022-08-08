using AcousticPACMAN
using ProgressMeter
using GLMakie
GLMakie.activate!()

function planewavescatter(; gridsize = 1024)
    T = Float64
    x = LinRange(-3, 3, gridsize)
    y = LinRange(-3, 3, gridsize)
    ρ₀ = parse(T, "1.2041")
    c₀ = parse(T, "342.21")
    Z = ρ₀ * c₀
    f = T(500)
    ω = f * 2 * pi
    k = ω / c₀
    r₀ = T(1)
    N = 6
    M = 100
    ϕₛ = pi / T(4)
    ic = planewave(M, k, ϕₛ)
    pac = pacman(M, N, k, r₀, Z, ic)
    p = pressure(pac)
    pinc = pressure(ic)
    val = zeros(Complex{T}, gridsize, gridsize)
    prog = Progress(length(x); desc = "Computing time-harmonic values: ")
    Threads.@threads :static for m in eachindex(x)
        for n in eachindex(y)
            r = hypot(x[m], y[n])
            ϕ = atan(y[n], x[m])
            val[m, n] = p(r, ϕ)
            if r ≥ r₀
                val[m, n] += pinc(r, ϕ)
            end
        end
        next!(prog)
    end
    return (x, y, val, ω, pac)
end

t = Observable(0.0)
x, y, z, ω, pac = planewavescatter()
zt = @lift(real.(exp.($t * im * ω) .* z))

fig = Figure(resolution = (1200, 1200))
axs = [Axis(fig[1, i]; aspect = DataAspect()) for i = 1:1]
hm = heatmap!(axs[1], x, y, zt)
#contour!(axs[2], x, y, zt; levels=20)
#contourf!(axs[3], x, y, zt; levels=20)
#Colorbar(fig[1, 4], hm, height=Relative(0.5))

timesteps = range(0, 5, step = 1 / 240)
prog = Progress(length(timesteps); desc = "Computing frames: ")
record(fig, "time_animation.mp4", timesteps; framerate = 60) do τ
    t[] = τ
    next!(prog)
end
