from cartpole import State, Target, Parameters

import casadi as cs


def find_parameters(states, gravity=9.81) -> Parameters:
    '''
    Find dynamics parameters from given control sequence and state trajectory.
    Use collocation method to solve nonlinear optimization problem (Hermit-Simpson method).

    Args:
        state: cartpole trajectory.

        gravity: gravity constant.
    
    Returns:
        Estimated dynamics parameters.
    '''

    assert len(states) > 1


    # For more information see:
    #   https://cartpole.robotics-lab.ru/3.0.0/dynamics-and-control

    # declare model variables
    b = cs.SX.sym('b')
    k = cs.SX.sym('k')

    # state variables, use only theta and theta_dot for collocation
    theta = cs.SX.sym('theta')
    theta_dot = cs.SX.sym('theta_dot')
    q = cs.vertcat(theta, theta_dot) 

    # control variables
    u = cs.SX.sym('u')

    # system dynamics
    q_dot = cs.vertcat(
        theta_dot,
        -b * theta_dot - k * (u * cs.cos(theta) + gravity * cs.sin(theta))
    )

    F = cs.Function('f', [q, u, b, k], [q_dot], ['q', 'u', 'b', 'k'], ['q_dot'])

    x, x0 = [], []
    x_min, x_max = [], []
    g = []
    g_min, g_max = [], []
    J = 0

    # add decision variables
    eps = 1e-6

    b = cs.MX.sym('b')
    x.append(b)
    x0.append(eps)
    x_min.append(0)
    x_max.append(cs.inf)

    k = cs.MX.sym('k')
    x.append(k)
    x0.append(eps)
    x_min.append(0)
    x_max.append(cs.inf)

    q_prev = None
    f_prev = None
    s_prev = None

    for i in range(0, len(states)-1):
        s = states[i]

        q = cs.MX.sym(f'q_{i:03d}', 2)
        x.append(q)
        x0 += [s.pole_angle, s.pole_angular_velocity]

        x_min += [-cs.inf, -cs.inf]
        x_max += [cs.inf, cs.inf]

        # minimize error of predicted state with dynamics model and measured state
        q_true = cs.vertcat(s.pole_angle, s.pole_angular_velocity)
        J += cs.sumsqr(q - q_true)

        f = F(q, s.cart_acceleration, b, k)

        # Hermite-Simpson collocation method
        if q_prev is not None and f_prev is not None and s_prev is not None:
            q_c = cs.MX.sym(f'qc_{i:03d}', 2)
            x.append(q_c)
            x0 += [s_prev.pole_angle, s_prev.pole_angular_velocity]
            x_min += [-cs.inf, -cs.inf]
            x_max += [cs.inf, cs.inf]

            h = s.stamp - s_prev.stamp
            f_c = F(q_c, s_prev.cart_acceleration, b, k)

            g.append(h / 6 * (f_prev + 4 * f_c + f) - (q - q_prev))
            g_min += [0, 0]
            g_max += [0, 0]

            g.append((q_prev + q)/2 + h/8 * (f - f_prev) - q_c)
            g_min += [0, 0]
            g_max += [0, 0]

        q_prev = q
        f_prev = f
        s_prev = s

    # Create an NLP solver
    x = cs.vertcat(*x)
    g = cs.vertcat(*g)

    nlp = {
        'x': x, # decision variables
        'f': J, # objective function
        'g': g, # constraints
    }

    opts = {
        'ipopt.print_level':0,
        'print_time':0
    }

    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)

    # solve
    solution = solver(x0=x0, lbx=x_min, ubx=x_max, lbg=g_min, ubg=g_max)
    
    b = float(solution['x'][0])
    k = float(solution['x'][1])

    return Parameters(friction_coef=b, mass_coef=k, gravity=gravity)
