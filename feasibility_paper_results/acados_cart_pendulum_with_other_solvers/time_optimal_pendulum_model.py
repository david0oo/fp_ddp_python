from acados_template import AcadosModel
import casadi as cs

def export_free_time_pendulum_ode_model() -> AcadosModel:

    model_name = 'free_time_pendulum'

    # constants
    m_cart = 1.0 # mass of the cart [kg]
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]

    # set up states & controls
    T = cs.SX.sym('T')
    x1      = cs.SX.sym('x1')
    theta   = cs.SX.sym('theta')
    v1      = cs.SX.sym('v1')
    dtheta  = cs.SX.sym('dtheta')

    x = cs.vertcat(T, x1, theta, v1, dtheta)

    F = cs.SX.sym('F')
    u = cs.vertcat(F)

    # xdot
    T_dot = cs.SX.sym('T_dot')
    x1_dot      = cs.SX.sym('x1_dot')
    theta_dot   = cs.SX.sym('theta_dot')
    v1_dot      = cs.SX.sym('v1_dot')
    dtheta_dot  = cs.SX.sym('dtheta_dot')

    xdot = cs.vertcat(T_dot, x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters
    p = cs.SX.sym('moon_x')

    # dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = m_cart + m - m*cos_theta*cos_theta
    f_expl = cs.vertcat(0,
                        T*v1,
                        T*dtheta,
                        T*((-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator),
                        T*((-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(m_cart+m)*g*sin_theta)/(l*denominator))
                        )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model




