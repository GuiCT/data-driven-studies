import optuna
from keras.backend import clear_session
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scipy.integrate import odeint
import numpy as np


N_TRAIN_EXAMPLES = 500
N_VALID_EXAMPLES = 100
EPOCHS = 300


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    # Gerando dados de treino
    # ================================================================================
    integration_domain = np.arange(0, 5, 0.01)
    batch_size = integration_domain.shape[0] - 1
    # Condições iniciais aleatórias [-1, 1]
    x0_training = -1 + 2 * np.random.random((N_TRAIN_EXAMPLES, 2))
    # Função do Espectro Simples
    def odefun(x1_x2, t0):
        x1, x2 = x1_x2
        return [- 0.05 * x1, - 1.0 * (x2 - x1 ** 2)]
    # Solucionando sistemas
    x_training = np.asarray([odeint(odefun, x0_j, integration_domain) for x0_j in x0_training])
    nn_input_training = np.zeros((N_TRAIN_EXAMPLES * (len(integration_domain) - 1), 2))
    nn_output_training = np.zeros_like(nn_input_training)

    for j in range(N_TRAIN_EXAMPLES):
        nn_input_training[j * (len(integration_domain) - 1) : (j + 1) * (len(integration_domain) - 1), :] = x_training[j, :-1, :]
        nn_output_training[j * (len(integration_domain) - 1) : (j + 1) * (len(integration_domain) - 1), :] = x_training[j, 1:, :]
    # ================================================================================

    # Gerando dados de validação
    # ================================================================================
    # Condições iniciais aleatórias [-5, 5]
    x0_validation = -5 + 10 * np.random.random((N_VALID_EXAMPLES, 2))
    # Solucionando sistemas
    x_validation = np.asarray([odeint(odefun, x0_j, integration_domain) for x0_j in x0_validation])
    # ================================================================================
    
    # Construindo modelo com os parâmetros sugeridos pelo Optuna
    # ================================================================================
    # Rede terá 4 camadas densas, com número de neurônios e funções de ativação definidos pelo Optuna
    model = Sequential()
    model.add(Dense(trial.suggest_int("n_units_l1", 8, 32, log=True), activation=trial.suggest_categorical("activation_l1", ["relu", "sigmoid"])))
    model.add(Dense(trial.suggest_int("n_units_l2", 8, 32, log=True), activation=trial.suggest_categorical("activation_l2", ["relu", "linear"])))
    model.add(Dense(trial.suggest_int("n_units_l3", 8, 32, log=True), activation=trial.suggest_categorical("activation_l3", ["relu", "linear"])))
    model.add(Dense(2, activation="linear"))

    # Otimizador e taxa de aprendizado definidos pelo Optuna
    # ================================================================================
    # Interrompe o treinamento se a função de *loss* não diminuir por 50 épocas
    callback = EarlyStopping(monitor='loss', patience=50)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=learning_rate),
    )

    model.fit(
        nn_input_training,
        nn_output_training,
        shuffle=True,
        batch_size=batch_size,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=0,
    )

    x_prediction = np.zeros_like(x_validation)
    x_prediction[:, 0, :] = x0_validation
    for j in range(1, len(integration_domain)):
        x_prediction[:, j, :] = model.predict(x_prediction[:, j - 1, :])
    
    # ================================================================================
    # Penaliza o número de neurônios na rede
    penalty = 0
    penalty += trial.suggest_int("n_units_l1", 8, 32, log=True)
    penalty += trial.suggest_int("n_units_l2", 8, 32, log=True)
    penalty += trial.suggest_int("n_units_l3", 8, 32, log=True)
    penalty -= 24
    penalty /= 72
    return np.mean(np.abs(x_prediction[:, 1:, :] - x_validation[:, 1:, :])) * ((1 + penalty) ** 2)


if __name__ == "__main__":
    # Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
    try:
        study = optuna.create_study(study_name='ode_feed_forward', direction='minimize', storage='sqlite:///ode_feed_forward.db')
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name='ode_feed_forward', storage='sqlite:///ode_feed_forward.db')
    study.optimize(objective, n_trials=1000, timeout=None)
