import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas
GRAVITY = 9.81  # g = 9.81 m/s²
RHO_WATER = 1000  # rho = 1000 kg/m³

# Geometria do problema
B_0 = 0.015  # b(x, 0) = 1.5cm = 0.015m (Largura inicial)
H_RESERVOIR_0 = 0.45  # H(0) = 0.45m (Altura do reservatório)
H_DEPTH_0 = 0.015  # h(x, 0) = 1.5cm = 0.015m (Profundidade)
MANNING = 3.6287999711930750e-3  # Calculada no artigo
X_1 = 1.3  # x_1 = 1.3m (Comprimento do canal)

# Primeiro caso, obtidos a partir de processo de otimização
ALPHA_B = 2.6135999360121783e-5
ALPHA_Z = 1.1388600032660172e-4
TAU_C = 0.0


def solve(
    dt: float,
    nx: int,
    t1: float,
):

    # Domínio espacial
    x = np.linspace(0, X_1, nx)
    dx = x[1] - x[0]
    n_points = len(x)
    # Domínio temporal
    t = np.arange(0, t1 + dt, dt)
    n_steps = len(t)

    # Variáveis necessárias para simulação
    A = np.ones(n_points) * B_0 * H_DEPTH_0
    Q = np.zeros(n_points)
    z_b = np.empty(n_points)
    b = np.ones(n_points) * B_0
    h = np.ones(n_points) * H_DEPTH_0
    # Outros vetores
    dQdx = np.zeros(n_points)
    dQ2Adx = np.zeros(n_points)
    d_zdx = np.zeros(n_points)
    # Séries temporais armazenadas
    H = np.zeros(n_steps)
    Q0 = np.zeros(n_steps)
    Q[0] = 0.00063002

    # Função que descreve a altura do leito inicial
    def z_b_0_func(x):
        BASE_HEIGHT = H_RESERVOIR_0 - H_DEPTH_0
        if x <= 0.17:
            return BASE_HEIGHT
        elif x == 1.3:
            return 0
        else:
            return BASE_HEIGHT * (1 - (x - 0.17) / 1.13)
    z_b = np.vectorize(z_b_0_func)(x)

    # Funções de atualização de erosão
    def phi(x): return 1 + 0.5 * (x - 0.17)**4
    def psi(x): return 2*x + x**2

    for step in range(n_steps):
        p = b + 2 * z_b
        # Raio hidráulico
        R = A / p
        # Termo da fricção
        S_f = (MANNING**2 * Q * np.abs(Q)) / (R**(4/3) * A**2)

        # Velocidade(m/s) = Vazão(m³/s) / Área(m²)
        V = Q / A
        # Tensão de cisalhamento no leito
        tau_b = RHO_WATER * GRAVITY * MANNING**2 * (V**2 / R**(1/3))

        # Atualização de A e Q a partir do esquema upwind
        dQdx[:-1] = (Q[1:] - Q[:-1]) / dx
        A -= dt * dQdx 

        Q2A = Q**2 / A
        # Derivada espacial de Q²/A
        dQ2Adx[:-1] = (Q2A[1:] - Q2A[:-1]) / dx
        # Derivada espacial de z
        z = z_b + h
        d_zdx[:-1] = (z[1:] - z[:-1]) / dx
        Q -= dt * (GRAVITY * A * (d_zdx + S_f) + dQ2Adx)

        h = A / b

        # Atualização de b e z_b a partir das derivadas temporais
        erosion_term = np.maximum(tau_b - TAU_C, 0)
        b += dt * ALPHA_B * erosion_term * phi(x)
        b = np.minimum(b, 1.2)
        z_b -= dt * ALPHA_Z * erosion_term * psi(x)

        H[step] = z_b[0] + h[0]
        Q0[step] = Q[0]

    return H, Q0


def show_plot(H: np.ndarray, Q0: np.ndarray):
    plt.plot(H, label='H')
    plt.show()
    plt.plot(Q0, label='Q0')
    plt.show()


if __name__ == '__main__':
    dt = 0.01
    nx = 1000
    t1 = 100
    H, Q0 = solve(dt, nx, t1)
    show_plot(H, Q0)
