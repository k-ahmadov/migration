from fracinj import physics
from fracinj.types import Parameters

params = Parameters(
    k_n=200e9,
    L=3000.0,
    mu=1e-3,
    w_i=1e-5,
    E=60e9,
    nu=0.25,
    Nx_p=1024,
    Nx_sn=512,
    Nt=2000,
    m_q=1e-13,
    # T=500.0,
)

scales = physics.ramp_rate_fixed_aperture_cubic(params)
scales.t_c / 86400

physics.ramp_rate_fixed_length_cubic(params).t_c / 86400 / 4
