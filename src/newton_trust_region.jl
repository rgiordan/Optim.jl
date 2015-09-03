macro newton_tr_trace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["h(x)"] = copy(H)
                dt["delta"] = copy(delta)
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end

# Choose a point in the trust region for the next step using
# the interative (nearly exact) method of section 4.3 of Nocedal and Wright.
# This is appropriate for Hessians that you factorize quickly.
#
# TODO: Allow the user to specify their own function for the subproblem.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size
#  tolerance: The convergence tolerance for newton's method
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  s, the step to take from the current x, is updated in place.
function solve_tr_subproblem!{T}(gr::Vector{T},
                                 H::Matrix{T},
                                 delta::T,
                                 s::Vector{T};
                                 tolerance=1e-12, verbose=false)
    n = length(gr)
    @assert n == length(s)
    @assert (n, n) == size(H)

    H_eig = eigfact(H)
    lambda_1 = H_eig[:values][1]

    # Cache the inner products between the eigenvectors and the gradient.
    qg2 = Array(T, n)
    for i=1:n
      qg2[i] = _dot(H_eig[:vectors][:, i], gr) ^ 2
    end

    hard_case = false
    for i = 1:n
      if (H_eig[:values][i] == lambda_1) &&  (qg2[i] <= 1e-10)
         hard_case = true
      end
    end

    @assert(!hard_case, "The hard case is not currently implemented.")

    # Function 4.39 in N&W
    function p(lambda::Real)
      p_sum = 0.
      for i = 1:n
        p_sum = p_sum + qg2[i] / (lambda + H_eig[:values][i])
      end
      p_sum
    end

    interior = false
    delta2 = delta ^ 2
    if p(0.0) <= delta2
        # No shrinkage is necessary, and -(H \ gr) is the solution.
        s[:] = -(H_eig[:vectors] ./ H_eig[:values]') * H_eig[:vectors]' * gr
        lambda = 0.0
        interior = true
    else
      # Algorithim 4.3 of N&W, with s insted of p_l to be consistent with
      # the rest of otpim.

      newton_diff = Inf
      max_iters = 20
      iter = 1
      B = copy(H)

      # Start at the absolute value of the smallest eigenvalue.
      # TODO: is there something better?
      lambda = abs(lambda_1)
      for i=1:n
        B[i, i] = H[i, i] + lambda
      end
      while (newton_diff > tolerance) && (iter <= max_iters)
        R = chol(B)
        s[:] = -R \ (R' \ gr)
        q_l = R' \ s
        norm2_s = norm2(s)
        lambda_previous = lambda
        lambda = (lambda_previous +
                  norm2_s * (sqrt(norm2_s) - delta) / (delta * norm2(q_l)))
        # Check that lambda is not less than -lambda_1, and if so, go half the
        # distance to -lambda_1.
        if lambda < -lambda_1
          lambda = 0.5 * (lambda_previous - lambda_1)
        end
        newton_diff = abs(lambda - lambda_previous)
        iter = iter + 1
        for i=1:n
          B[i, i] = H[i, i] + lambda
        end
      end
      @assert(iter > 1, "Bad tolerance -- no iterations were computed")
      if iter > max_iters
        warn(string("In the trust region subproblem max_iters ($max_iters) ",
                    "was exceeded.  Diff vs tolerance: ",
                    "$(newton_diff) > $(tolerance)"))
      end
      interior = true
    end

    m = zero(T)
    if interior
      m = _dot(gr, s) + 0.5 * _dot(s, H * s)
    else
      m = _dot(gr, s) + 0.5 * _dot(s, B * s)
    end

    return m, interior
end


function newton_tr{T}(d::TwiceDifferentiableFunction,
                       initial_x::Vector{T};
                       initial_delta::T=1.0,
                       delta_hat::T = 10.0,
                       eta::T = 0.1,
                       xtol::Real = 1e-32,
                       ftol::Real = 1e-8,
                       grtol::Real = 1e-8,
                       iterations::Integer = 1_000,
                       store_trace::Bool = false,
                       show_trace::Bool = false,
                       extended_trace::Bool = false)

    @assert(delta_hat > 0, "delta_hat must be strictly positive")
    @assert(0 < initial_delta < delta_hat, "delta must be in (0, delta_hat)")
    @assert(0 <= eta < 0.25, "eta must be in [0, 0.25)")

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(T, n)

    # The current search direction
    # TODO: Try to avoid re-allocating s
    s = Array(T, n)

    # Store f(x), the function value, in f_x
    f_x_previous, f_x = NaN, d.fg!(x, gr)
    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Store the hessian in H
    H = Array(T, n, n)
    d.h!(x, H)

    # Keep track of trust region sizes
    delta = copy(initial_delta)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace
    @newton_tr_trace

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Find the next step direction.
        m, interior = solve_tr_subproblem!(gr, H, delta, s)

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        for i in 1:n
            @inbounds x[i] = x[i] + s[i]
        end

        # Update the function value and gradient
        f_x_previous, f_x = f_x, d.fg!(x, gr)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        x_converged,
        f_converged,
        gr_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       gr,
                                       xtol,
                                       ftol,
                                       grtol)

        if !converged
          # Update the trust region size based on the discrepancy between
          # the predicted and actual function values.  (Algorithm 4.1 in N&W)
          @assert(m < 0,
                  "unconverged solution failed to decrease quadratic objective")
          rho = (f_x_previous - f_x) / m
          if rho < 0.25
              delta *= 0.25
          elseif rho > 0.75 && interior
              delta = min(2 * delta, delta_hat)
          # else leave delta unchanged.
          end

          if rho > eta
              # Update the Hessian and accept the point
              d.h!(x, H)
          else
              # The improvement is too small and we won't take it.
              x, f_x = x_previous, f_x_previous
          end
        end

        @newton_tr_trace
    end

    return MultivariateOptimizationResults("Newton's Method with Trust Region",
                                           initial_x,
                                           x,
                                           @compat(Float64(f_x)),
                                           iteration,
                                           iteration == iterations,
                                           x_converged,
                                           xtol,
                                           f_converged,
                                           ftol,
                                           gr_converged,
                                           grtol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
