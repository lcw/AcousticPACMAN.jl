using AcousticPACMAN
using CSV
using DataFrames
using Test
using Pkg

@testset "Surface Vibration" begin
    referencedata =
        CSV.read("../contrib/reference_solution_surface_vibration.csv", DataFrame)

    function surfacevibrationscatter(f)
        ρ₀ = big"1.2041"
        c₀ = big"342.21"
        Z = ρ₀ * c₀
        k = f * 2 * pi / c₀
        r₀ = big"1.0"
        M = 100
        N = 6
        pac =
            pacman(M, N, k, r₀, Z, surfacevibration(BigFloat[one(BigFloat)/10], BigFloat[]))
        p = pressure(pac)
        r = big"2.0"
        ϕ = deg2rad.(range(big"0.0", stop = big"360", length = 72))
        a = similar(ϕ, complex(typeof(r)))
        Threads.@threads :static for i in eachindex(a, ϕ)
            a[i] = p.(r, ϕ[i])
        end
        return a
    end

    function referencescatter(j)
        return parse.(Complex{BigFloat}, referencedata[:, 2+j])
    end

    for (j, (f, rtol)) in enumerate(
        zip(
            [big"16", big"31.5", big"63", (big"125" .* 2 .^ (0:5))...],
            [0.0023, 0.0034, 0.0054, 0.0084, 0.015, 0.028, 0.054, 0.11, 0.22],
        ),
    )
        @testset "Freq $f" begin
            @test isapprox(surfacevibrationscatter(f), referencescatter(j); rtol = rtol)
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
