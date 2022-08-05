using AcousticPACMAN
using CondaPkg
using Pkg
using PythonCall
using Test

CondaPkg.add("scipy")
CondaPkg.add("numpy")
CondaPkg.status()

pyimport("sys").path.append(".")
const pypacman = pyimport("pacman")
const pycylinder = pyimport("cylinder")

@testset "Scattering" begin
    for (rₛ, ϕₛ, Q, V₀) in (
        (Inf, pi / 4, 1, 0), # plane wave scattering
        (0, 0, 0, 1 / 10),   # surface vibration
    )
        for f in [16.0, 1000]
            for N in [4, 6]
                @testset "Case N=$N f=$f rₛ=$rₛ ϕₛ=$ϕₛ Q=$Q V₀=$V₀" begin
                    r₀ = 1.0
                    r = 2.0
                    ρ₀ = 1.2041
                    c₀ = 342.21
                    Z = ρ₀ * c₀
                    k = f * 2 * pi / c₀
                    M = 10

                    pyAA, pyAS = pypacman.build_matrix(k, r₀, N, M)
                    pyrhsA, pyrhsS = pypacman.build_rhs(k, r₀, N, M, rₛ, ϕₛ, Q, 0, Z * V₀)
                    pyaA, pyaS =
                        pypacman.calc_mode_amplitudes(k, r₀, N, M, rₛ, ϕₛ, Q, 0, Z * V₀)
                    pybA, pybS = pypacman.calc_b(pyaA, pyaS, k, r₀, N, M, rₛ, ϕₛ, 0, Q)

                    ic = rₛ == Inf ? planewave(M, ϕₛ) : surfacevibration([V₀], typeof(V₀)[])
                    jlAS, jlAA, jlrhsS, jlrhsA =
                        AcousticPACMAN.getsystem(M, N, k, r₀, Z, ic)

                    @test isapprox(jlAS, pyconvert(Array, pyAS))
                    @test isapprox(jlAA, pyconvert(Array, pyAA))
                    @test isapprox(jlrhsS, pyconvert(Array, pyrhsS))
                    @test isapprox(jlrhsA, pyconvert(Array, pyrhsA))

                    pac = pacman(M, N, k, r₀, Z, ic)
                    jlaS = AcousticPACMAN.outersymmetricmodeamplitudes(pac)
                    jlaA = AcousticPACMAN.outerantisymmetricmodeamplitudes(pac)
                    jlbS = AcousticPACMAN.innersymmetricmodeamplitudes(pac)
                    jlbA = AcousticPACMAN.innerantisymmetricmodeamplitudes(pac)

                    @test isapprox(jlaS, pyconvert(Array, pyaS))
                    @test isapprox(jlaA, pyconvert(Array, pyaA))
                    @test isapprox(jlbS, pyconvert(Array, pybS))
                    @test isapprox(jlbA, pyconvert(Array, pybA))

                    p = pressure(pac)
                    ϕs = range(0.0; stop = 2π, length = 10)

                    pyp = [
                        pyconvert(
                            Complex{Float64},
                            pycylinder.calc_p(k, pyaA, pyaS, [r * cos(ϕ), r * sin(ϕ)]),
                        ) for ϕ in ϕs
                    ]
                    jlp = p.(r, ϕs)

                    @test isapprox(pyp, jlp)
                end
            end
        end
    end
end

if tryparse(Bool, get(ENV, "PACMAN_TEST_EXAMPLES", "false"))
    @testset "examples" begin
        julia = Base.julia_cmd()
        base_dir = joinpath(@__DIR__, "..")

        for example_dir in readdir(joinpath(base_dir, "examples"), join = true)
            @testset "$example_dir" begin
                mktempdir() do tmp_dir
                    # Change to temporary directory so that any files created by the
                    # example get cleaned up after execution.
                    cd(tmp_dir)
                    example_project = Pkg.Types.projectfile_path(example_dir)
                    tmp_project = Pkg.Types.projectfile_path(tmp_dir)
                    cp(example_project, tmp_project)

                    for script in
                        filter!(s -> endswith(s, ".jl"), readdir(example_dir, join = true))
                        cmd = `$julia --project=$tmp_project --threads=auto -e "import Pkg; Pkg.develop(path=raw\"$base_dir\"); Pkg.instantiate(); include(raw\"$script\")"`
                        @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
                    end
                end
            end
        end
    end
end
