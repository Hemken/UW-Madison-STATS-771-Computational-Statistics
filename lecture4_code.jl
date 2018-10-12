using Pkg
Pkg.add("IJulia")
using LinearAlgebra

# Least Squares
function LS(A,b; ϵ = 1e-14)
    """
    Solves a linear regression problem given
    the coefficient matrix A and the constant
    vector b. Return the x hat and the norm-2 of c2
    """
    n, m = size(A)
    F = qr(A, Val(true))
    c = F.Q' * b
    c1 = c[1:m]
    c2 = c[m+1:n]
    return F.P * inv(F.R) * c1, sqrt(c2' * c2)
end

n, m = 10, 4
A = rand(10,4)
x = rand(4)
b = A*x
n, m = size(A)
F = qr(A, Val(true))
c = F.Q' * b
c1 = c[1:m]
c2 = c[m+1:n]

x_hat = F.P * inv(F.R) * c1
c2 = sqrt(c2' * c2)

x_LSQ = inv(A'A)A'b
result = LS(A,b)

norm(x_hat - x_LSQ)

# Under determined Least Squares
function underLS(A,b; ϵ = 1e-14)
    """
    Solves an underdetermined linear system given
    the coefficient matrix A and the constant
    vector b. Returns the least norm solution.
    """
    n, m = size(A)
    s = min(n,m)
    F = qr(A, Val(true))

    #Compute rank approximation r
    #Rtrm = F.R[1:s,1:s]
    #r = maximum(find(abs.(diag(Rtrm)) .>= ϵ))
    r = rank(F.R)
    l = m - r

    #Generate R and S
    R, S = F.R[1:r,1:r], F.R[1:r,r+1:end]
    d, P = inv(R)*F.Q'*b[1:r], inv(R)*S
    z2 = inv(P'*P + Matrix{Float64}(I,l,l)) * P'* d
    z1 = d - P*z2
    return F.P*vcat(z1,z2)
end


n, m = 4, 10
A = rand(n,m)
b = rand(n)
#underLS(A,b)

#test
p, m = 4, 10
A = rand(n,m)

F = qr(A, Val(true))
