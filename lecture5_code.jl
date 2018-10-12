using LinearAlgebra

# Gram-Schmidt QR decompostion
# With one loop
function gramschmidtQR(A)
    """
    Implement the gram-schmidt procedure.
    Input a full column rank matrix A.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != n
        println("The input matrix is not full column rank")
    else
        u1 = A[:,1]
        e1 = u1 ./ norm(u1)
        e = e1
        R = zeros(n,1)
        R[1] = A[:,1]'e1
        for i in 2:n
            ui = A[:, i] - sum(e.* (A[:,i]'*e), dims = 2)
            e = hcat(e, ui ./ norm(ui))
            t = zeros(n,1)
            t[1:i] = A[:,i]'*e
            R = hcat(R, t)
        end
    end
    return e, R
end

A = [1 1 0; 1 0 1; 0 1 1]
Q, R = gramschmidtQR(A)

m, n = 10, 4
A = rand(m, n)
Q, R = gramschmidtQR(A)

# Gram-Schmidt QR decompostion
# With two loops
function gramschmidtQR2(A)
    """
    Implement the gram-schmidt procedure.
    Input a full column rank matrix A.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != n
        println("The input matrix is not full column rank")
    else
        Q = zeros(m, n)
        R = zeros(n,n)
        R[1,1] = norm(A[:,1])
        Q[:,1] = A[:,1]./R[1,1]
        for k = 2:n
            z = A[:, k]
            for i = 1:k-1
                R[i,k] = A[:,k]'Q[:,i]
                z = z - R[i,k]Q[:,i]
            end
            R[k,k] = norm(z)
            Q[:, k] = z ./ R[k,k]
        end
    end
    return Q, R
end

A = [1 1 0; 1 0 1; 0 1 1]
Q, R = gramschmidtQR2(A)


#Modified gram-schmidt QR decompostion
# with two loops
function modifiedGSQR(A)
    """
    Implement the modifies gram-schmidt procedure.
    Input a full column rank matrix A.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != n
        println("The input matrix is not full column rank")
    else
        Q = float(A)
        R = zeros(n,n)
        for k = 1:n
            R[k,k] = norm(Q[:,k])
            Q[:,k] = Q[:,k] ./ R[k,k]
            for j = k+1:n
                R[k,j] = A[:,j]'Q[:,k]
                Q[:,j] = Q[:,j] - R[k,j]Q[:,k]
            end
        end
    end
    return Q, R
end

A = [1 1 1; 1 1 0 ; 1 0 2;1 0 0]
Q, R = modifiedGSQR(A)
