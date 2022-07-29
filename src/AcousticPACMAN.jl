module AcousticPACMAN
using SpecialFunctions

export pacman, pressure, velocity, surfacevibration

δ(m, n) = m == n
δ(m) = m == 0 ? 1 : 2

# Here we use a recurrence relationship to compute the derivative of the
# various Bessel functions, see <https://dlmf.nist.gov/10.6#E1> and
# <https://dlmf.nist.gov/10.6#E3>.
for C in (:besselj, :bessely, :hankelh1, :hankelh2)
    Cprime = Symbol(C, "prime")
    @eval begin
        function $Cprime(ν, z)
            if ν == 0
                return -$C(one(ν), z)
            else
                return ($C(ν - 1, z) - $C(ν + 1, z)) / 2
            end
        end
    end
end

function SpecialFunctions.besselh(nu::BigFloat, k::Int, x::BigFloat)
    if k == 1
        return complex(besselj(nu, x), bessely(nu, x))
    elseif k == 2
        return complex(besselj(nu, x), -bessely(nu, x))
    else
        throw(SpecialFunctions.AmosException(1))
    end
end

# These are functions (B.1) and (B.2) from
# <https://doi.org/10.1016/j.jcp.2017.06.018> respectively
# ϕ₀(N) = π / N
# Isin(ϕ₀, x, y) = (sinc((x-y)*ϕ₀/π) - sinc((x+y)*ϕ₀/π)) / 2
# Icos(ϕ₀, x, y) = (sinc((x+y)*ϕ₀/π) + sinc((x-y)*ϕ₀/π)) / 2
#
Isin(Ninv, x, y) = (sinc((x - y) * Ninv) - sinc((x + y) * Ninv)) / 2
Icos(Ninv, x, y) = (sinc((x + y) * Ninv) + sinc((x - y) * Ninv)) / 2

abstract type Fun end

pacman(f::Fun) = f.pac

struct PressureFun{P} <: Fun
    pac::P
end

struct VelocityFun{P} <: Fun
    pac::P
end

struct Pacman{T,C<:AbstractArray{Complex{T}}}
    aA::C
    aS::C
    M::Int
    N::Int
    k::T
    r₀::T
    Z₀::T
end

struct InitialCondition{A}
    VSvib::A
    VAvib::A
    Pinc::A
    Vinc::A
end

vibrationsymmetricmodeamplitudes(ic::InitialCondition) = ic.VSvib
vibrationantisymmetricmodeamplitudes(ic::InitialCondition) = ic.VAvib
incidentpressuremodeamplitudes(ic::InitialCondition) = ic.Pinc
incidentvelocitymodeamplitudes(ic::InitialCondition) = ic.Vinc

wavenumber(pac::Pacman) = pac.k
truncationorder(pac::Pacman) = pac.M
wedgenumber(pac::Pacman) = pac.N
radius(pac::Pacman) = pac.r₀
characteristicimpedance(pac::Pacman) = pac.Z₀
outersymmetricmodeamplitudes(pac::Pacman) = pac.aS
outerantisymmetricmodeamplitudes(pac::Pacman) = pac.aA
pressure(pac::Pacman) = PressureFun(pac)
velocity(pac::Pacman) = VelocityFun(pac)

function pacman(M, N, k, r₀, Z₀, ic::InitialCondition)
    T = typeof(r₀)

    VSvib = vibrationsymmetricmodeamplitudes(ic)
    VAvib = vibrationantisymmetricmodeamplitudes(ic)
    Pinc = incidentpressuremodeamplitudes(ic)
    Vinc = incidentvelocitymodeamplitudes(ic)

    kr₀ = k * r₀
    Ninv = one(T) / N

    Jcoef = [Ninv * δ(η * N) * besseljprime(η * N, kr₀) / besselj(η * N, kr₀) for η = 0:M]
    hp = [hankelh2prime(T(i), kr₀) / δ(i) for i = 0:M]
    h = [hankelh2(T(n), kr₀) for n = 0:M]
    icosN = [Icos(Ninv, n, η * N) for n = 0:M, η = 0:M]

    # Build and solve system for the symmetric modes
    A = zeros(complex(T), M + 1, M + 1)
    Threads.@threads :static for n = 0:M
        for i = 0:M
            J = zero(T)
            for η = 0:M
                J += Jcoef[η+1] * icosN[n+1, η+1] * icosN[i+1, η+1]
            end
            A[i+1, n+1] = δ(n, i) * hp[i+1] - J * h[n+1]
        end
    end

    rhs = zeros(complex(T), M + 1)

    Threads.@threads :static for i = 0:M
        for nstar = 0:(length(Pinc)-1)
            for η = 0:M
                rhs[i+1] +=
                    Pinc[nstar+1] * Jcoef[η+1] * icosN[nstar+1, η+1] * icosN[i+1, η+1]
            end
        end

        if i < length(Vinc)
            rhs[i+1] += -Vinc[i+1] / δ(i)
        end

        if i < length(VSvib)
            rhs[i+1] += -im * Z₀ * VSvib[i+1] / δ(i)
        end

        for nstar = 0:(length(VSvib)-1)
            rhs[i+1] += im * Ninv * Z₀ * VSvib[nstar+1] * Icos(Ninv, i, nstar)
        end
    end

    aS = A \ rhs

    # Build and solve system for the antisymmetric modes
    aA = similar(aS)
    fill!(aA, 0)

    return Pacman(aA, aS, M, N, k, r₀, Z₀)
end

(p::PressureFun)(r, ϕ) = apply(p, r, ϕ)

@inline function apply(p::PressureFun, r, ϕ)
    pac = pacman(p)
    k = wavenumber(pac)
    T = eltype(k)
    kr = k * r

    M = truncationorder(pac)
    aA = outerantisymmetricmodeamplitudes(pac)
    aS = outersymmetricmodeamplitudes(pac)

    val = zero(complex(T))
    for n = 0:M
        aAn = aA[n+1]
        aSn = aS[n+1]
        val += hankelh2(T(n), kr) * (aAn * sin(n * ϕ) + aSn * cos(n * ϕ))
    end

    return val
end

(v::VelocityFun)(r, ϕ) = apply(v, r, ϕ)

@inline function apply(v::VelocityFun, r, ϕ)
    pac = pacman(v)
    k = wavenumber(pac)
    T = eltype(k)
    kr = k * r

    M = truncationorder(pac)
    aA = outerantisymmetricmodeamplitudes(pac)
    aS = outersymmetricmodeamplitudes(pac)

    val = zero(complex(T))
    for n = 0:M
        aAn = aA[n+1]
        aSn = aS[n+1]
        val += hankelh2prime(T(n), kr) * (aAn * sin(n * ϕ) + aSn * cos(n * ϕ))
    end

    return im * val
end

function surfacevibration(VSvib, VAvib)
    Pinc = similar(VSvib, 0)
    Vinc = similar(VSvib, 0)
    return InitialCondition(VSvib, VAvib, Pinc, Vinc)
end

end # module AcousticPACMAN
