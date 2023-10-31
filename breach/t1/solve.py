import numpy as np

# Constantes físicas
GRAVITY = 9.81 # g = 9.81 m/s²
RHO_WATER = 1000 # rho = 1000 kg/m³

# Geometria do problema
B_0 = 0.015 # b(x, 0) = 1.5cm = 0.015m (Largura inicial)
H_RESERVOIR_0 = 0.45 # H(0) = 0.45m (Altura do reservatório)
H_DEPTH_0 = 0.015 # h(x, 0) = 1.5cm = 0.015m (Profundidade)
MANNING = 3.6287999711930750e-3 # Calculada no artigo
X_1 = 1.3 # x_1 = 1.3m (Comprimento do canal)
Z_B_0 = H_RESERVOIR_0 - H_DEPTH_0 # Elevação inicial do leito

# Primeiro caso, obtidos a partir de processo de otimização
ALPHA_B = 2.6135999360121783e-5
ALPHA_Z = 1.1388600032660172e-4
TAU_C = 0.0

def solve(
    dt:float,
    nx:int,
    t1:float,
    ):
    
    # Domínio espacial
    x = np.linspace(0, X_1, nx)
    dx = x[1] - x[0]
    n_points = len(x)
    # Domínio temporal
    t = np.arange(0, t1 + dt, dt)
    n_steps = len(t)

    # Variáveis necessárias para simulação
    A = np.ones(n_points) * (B_0 * X_1)
    Q = np.zeros(n_points)
    z_b = np.ones(n_points) * Z_B_0
    b = np.ones(n_points) * B_0
    h = np.ones(n_points) * H_DEPTH_0
    # Séries temporais armazenadas
    H = np.zeros(n_steps)
    Q0 = np.zeros(n_steps)
    
    # Funções de atualização de erosão
    phi = lambda x: 1 + 0.5 * (x - 0.17)**4
    psi = lambda x: 2*x + x**2
    
    for step in range(n_steps):
        # Termo da fricção
        R:float
        S_f = (MANNING**2 * Q * np.abs(Q)) / (R**(4/3) * A**2)
        
        # Velocidade(m/s) = Vazão(m³/s) / Área(m²)
        V = Q / A
        # Tensão de cisalhamento no leito
        tau_b = RHO_WATER * GRAVITY * MANNING**2 * (V**2 / R**(1/3))

        # Atualização de A e Q a partir do esquema upwind
        A[1:] = A[1:] - dt / dx * (Q[1:] - Q[:-1])
        Q2A = Q**2 / A
        z = h + z_b
        Q[1:] = Q[1:] - dt / dx * (Q2A[1:] - Q2A[:-1]) + dt * GRAVITY * A[1:] * (z[1:] - z[:-1]) / dx + dt * GRAVITY * A[1:] * S_f[1:]
        
        # Atualização de b e z_b a partir das derivadas temporais
        erosion_term = np.maximum(tau_b - TAU_C, 0)
        b += dt * ALPHA_B * erosion_term * phi(x)
        z_b -= dt * ALPHA_Z * erosion_term * psi(x)
        
        H[step] = h[0] + z_b[0]
        Q0[step] = Q[0]
        
    return H, Q0
