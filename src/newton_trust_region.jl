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
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  s, the step to take from the current x, is updated in place.
function solve_tr_subproblem!{T}(gr::Vector{T},
                                 H::Matrix{T},
                                 delta::T,
                                 s::Vector{T})
    n = length(gr)
    @assert n == length(s)
    @assert (n, n) == size(H)

    H_eig = eigfact(H)
    lambda_1 = H_eig[:values][1]
    hard_case = false
    for i = 1:n
      if (H_eig[:values][i] == lambda_1) &&
         (abs(_dot(gr, H_eig[:vectors][:, i])) <= 1e-6)
         hard_case = true
      end
    end

    @assert(!hard_case, "The hard case is not currently implemented.")

    m = min_value
    s[:] = min_s

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
    @assert(0 < initial_ < delta_hat, "delta must be in (0, delta_hat)")
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

    return MultivariateOptimizationResults("Newton's Method",
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
