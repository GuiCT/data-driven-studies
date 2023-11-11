import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulation physical parameters
BARRIER_POSITION = 500.
DEPTH_DOWNSTREAM = 2.
DEPTH_UPSTREAM = 10.
GRAVITY = 9.81
LENGTH = 1000.

# Numeric parameters
ANALISYS_ON_POINT = 50
DT = 0.001
FILE_NAME = './data/breach_simplified.csv'
FINAL_TIME = 20.0
NUMBER_POINTS = 101

def solve(n: int, dt: float, final_t: float):
    x = np.linspace(0, LENGTH, n)
    dx = x[1] - x[0]
    print('Courant number: ', dt / dx)
    t = np.arange(0, final_t + dt, dt)
    number_of_steps = len(t) - 1
    # Initial condition for h

    def h0(x):
        if x <= BARRIER_POSITION:
            return DEPTH_UPSTREAM
        else:
            return DEPTH_DOWNSTREAM
    h = np.vectorize(h0)(x)
    # Initial condition for q
    q = np.zeros(n)
    h_history = np.zeros((number_of_steps + 1, n))
    q_history = np.zeros((number_of_steps + 1, n))
    h_history[0, :] = h
    q_history[0, :] = q
    for i in range(number_of_steps):
        # Auxiliary term (q^2/h + (g*h^2)/2)
        aux = q**2 / h + (GRAVITY * h**2) / 2
        # Auxiliary first derivative wrt x, with backward difference
        aux_dx = np.zeros_like(aux)
        aux_dx[1:] = (aux[1:] - aux[:-1]) / dx
        # (qt+1 - qt)/dt + aux_dx = -g*((n^2 * q * |q|)/h^(10/3))
        q[1:] = q[1:] - dt * aux_dx[1:]
        # (ht+1 - ht)/dt + (qx - qx-1)/dx = 0
        h[:-1] = h[:-1] - dt * (q[1:] - q[:-1]) / dx
        # Boundary conditions don't need to be enforced, as they are already satisfied
        # h[-1] = Initial downstream depth
        # q[0] = 0.0
        h_history[i + 1, :] = h
        q_history[i + 1, :] = q
    return x, t, h_history, q_history


if __name__ == '__main__':
    x, t, h_history, q_history = solve(NUMBER_POINTS, DT, FINAL_TIME)
    fig = plt.figure(figsize=(20, 5))
    ax = fig.subplots(1, 2)
    ax[0].plot(t, h_history[:, ANALISYS_ON_POINT])
    ax[1].plot(t, q_history[:, ANALISYS_ON_POINT])
    fig.suptitle('Histórico de altura e vazão no ponto da barragem')
    plt.show()
    df = pd.DataFrame(
        {'water_depth': h_history[:, ANALISYS_ON_POINT], 'inlet_discharge': q_history[:, ANALISYS_ON_POINT]})
    df.to_csv(FILE_NAME, index=False)
