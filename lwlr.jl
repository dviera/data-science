using CSV
using LinearAlgebra
using Plots

function read_dat(dat)
    x = []

    f = open(dat)
    for line in readlines(f)
        push!(x, parse.(Float64, split(line)))
    end
    close(f)
    
    return Matrix(hcat(x...)')
end

X = read_dat("data/x.dat")
y = read_dat("data/y.dat")

function sigmoid(z)
    1 ./ (1 + exp.(-z))
end

function hypo(z, theta)
    sigmoid.(z * theta)
end

function norm2(A; dims)
    B = sum(A.^2; dims=dims)
    B .= sqrt.(B)
end

function weights(X, x, tau)
    exp.(-norm2(X .- x', dims=2).^2 ./ (2 * tau^2))
end

# main function
function lwlr(X, y, x, tau; lambda=0.001, tol=1e-6)
    m, n = size(X)
    theta_new = ones(Float64, 2, 1)
    theta_old = zeros(Float64, 2, 1)

    # === Newton's method
    while norm2(theta_new - theta_old, dims=1)[1] > tol
        w = weights(X, x, tau)
        h = hypo(X, theta_new)
        z = w .* (y .- h)
        g = X' * z .- lambda * theta_new
        D = - w .* h .* (1 .- h)
        D = diagm(vec(D))
        H = X' * D * X - I * lambda
        theta_old = theta_new
        theta_new = theta_new - LinearAlgebra.inv(H) * g
        # @show theta_old
        # @show theta_new
        # println("---------------------------------------------------")
    end
    # ===
    return theta_new
end

tt = lwlr(X, y, [2,3], 6, lambda=0.001, tol=1e-6)

function pred(x, tt)
    """
    x a matrix or raw vector
    """
    Float64.(hypo(x, tt) .> 0.5)
end

function lwrl_plot(X, y, tau, res)

    m, n = size(X)

    x_min, x_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    y_min, y_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    # === meshgrid
    xx = range(x_min, stop=x_max, length=res)
    len_xx = length(xx)

    yy = range(y_min, stop=y_max, length=res)
    len_yy = length(yy)
    
    xx = xx' .* ones(len_yy)
    yy = ones(len_xx)' .* yy
    # ===

    predictions = zeros(res, res)

    for i in 1:res
        for j in 1:res
            x = [xx[i, j], yy[i, j]]
            theta = lwlr(X, y, x, tau)
            predictions[i, j] = pred(x', theta)[1]
        end 
    end

    scatter(xx, yy,
     marker_z=predictions, 
     alpha=0.05, 
     color=:jet, 
     markerstrokewidth=0, 
     aspect_ratio=1, 
     label=false, 
     legend=:none)

    scatter!(X[:, 1], X[:, 2], 
    markerstrokewidth=0, 
    markersize=6, 
    marker_z=y, 
    color=:jet, 
    legend=:none)

    title!("Decision Boundary with τ = $(tau)")
    xlabel!("Feature 1")
    ylabel!("Feature 2")

end


taus = [0.05, 0.10, 0.50, 1.00, 5.00]

anim = @animate for tau in taus
    lwrl_plot(X, y, tau, 200)
end

gif(anim, "lwlr.gif", fps=1)