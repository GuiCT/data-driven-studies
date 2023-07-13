import time

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.integrate import odeint

import optuna


def separate_groups(x: np.ndarray, size: int):
    if (len(x.shape) != 2):
        raise ValueError(
            f'Expected ndims=2, but got ndims={len(x.shape)}.')

    n_rows = x.shape[0]
    n_columns = x.shape[1]

    if (n_rows <= size):
        raise ValueError(
            f'Unable to split input and output group.')

    n_groups = n_rows - size
    groups = np.zeros((n_groups, size, n_columns))
    groups_output = np.zeros((n_groups, 1, n_columns))
    for i in range(n_groups):
        groups[i, :, :] = x[i:i + size, :]
        groups_output[i, :, :] = x[i + size, :]
    return groups, groups_output


def random_with_scale(shape: tuple, scale: float):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def simple_spectrum(x1_x2, t0):
    x1, x2 = x1_x2
    return [- 0.05 * x1, - 1.0 * (x2 - x1 ** 2)]


ODEFUN = simple_spectrum
ODEFUN_SIZE = 2
MIN_LSTM_CELLS = 8
MAX_LSTM_CELLS = 256
TRAINING_SYSTEMS = 256
VALIDATION_SYSTEMS = 64
EPOCHS = 50
X0_SCALE_TRAINING = 10.0
INTEGRATION_DOMAIN = np.arange(0, 10 + 0.1, 0.1)
TIMESTEPS_PER_SYSTEM = len(INTEGRATION_DOMAIN)


def generate_training_data(trailing_timesteps):
    x0_training = random_with_scale(
        (TRAINING_SYSTEMS, 2), X0_SCALE_TRAINING)
    x_training = np.asarray(
        [odeint(ODEFUN, x0_j, INTEGRATION_DOMAIN) for x0_j in x0_training])
    samples_per_system = TIMESTEPS_PER_SYSTEM - trailing_timesteps
    total_samples = samples_per_system * TRAINING_SYSTEMS
    training_input = np.zeros(
        (total_samples, trailing_timesteps, ODEFUN_SIZE))
    training_output = np.zeros((total_samples, 1, ODEFUN_SIZE))
    for system_index in range(TRAINING_SYSTEMS):
        system_timesteps = x_training[system_index, :, :]
        system_input_groups, system_outputs = separate_groups(
            system_timesteps, trailing_timesteps)
        lower_bound = system_index * samples_per_system
        upper_bound = (system_index + 1) * samples_per_system
        training_input[lower_bound:upper_bound, :, :] = system_input_groups
        training_output[lower_bound:upper_bound, :, :] = system_outputs
    return training_input, training_output.squeeze()


def generate_validation_data():
    x0_validation = random_with_scale(
        (VALIDATION_SYSTEMS, 2), X0_SCALE_TRAINING)
    x_validation = np.asarray(
        [odeint(ODEFUN, x0_j, INTEGRATION_DOMAIN) for x0_j in x0_validation])
    return x_validation


def model_network(trial: optuna.Trial, trailing_timesteps):
    # Hiperparâmetros estudados
    cell_count = trial.suggest_int(
        "n_units", MIN_LSTM_CELLS, MAX_LSTM_CELLS, log=True)
    activation_function = trial.suggest_categorical(
        "activation_function", ["relu", "tanh", "sigmoid", "selu"])
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-1, log=True)
    beta_1 = trial.suggest_float("adam_beta_1", 0.1, 0.999, step=0.001)
    beta_2 = trial.suggest_float("adam_beta_2", 0.1, 0.999, step=0.001)
    use_amsgrad = trial.suggest_categorical("use_amsgrad", [False, True])

    model = Sequential()
    model.add(LSTM(cell_count, activation=activation_function, input_shape=(
        trailing_timesteps, ODEFUN_SIZE), return_sequences=False))
    model.add(Dense(ODEFUN_SIZE, activation="linear"))

    model.compile(
        loss="mse",
        optimizer=Adam(
            beta_1=beta_1,
            beta_2=beta_2,
            amsgrad=use_amsgrad,
            learning_rate=learning_rate),
    )
    return model, cell_count


def predict(x_validation, model, trailing_timesteps):
    prediction = np.zeros_like(x_validation)
    prediction[:, 0:trailing_timesteps,
               :] = x_validation[:, 0:trailing_timesteps, :]
    for k in range(trailing_timesteps, TIMESTEPS_PER_SYSTEM):
        lower_bound = k - trailing_timesteps
        upper_bound = k
        prediction[:, k, :] = model.predict(
            prediction[:, lower_bound:upper_bound, :], verbose=0)
    return prediction


def calculate_penalty_multiplier(cell_count):
    return (1 + ((cell_count - MIN_LSTM_CELLS) / (MAX_LSTM_CELLS - MIN_LSTM_CELLS))) ** 2


TRAINING_INPUT, TRAINING_OUTPUT = generate_training_data(5)
VALIDATION_DATA = generate_validation_data()


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", [256, 128, 64, 32])

    model, _ = model_network(trial, 5)
    start = time.time()
    try:
        history = model.fit(
            TRAINING_INPUT,
            TRAINING_OUTPUT,
            shuffle=True,
            batch_size=batch_size,
            epochs=EPOCHS,
            verbose=1,
        )
        duration = time.time() - start
        trial.set_user_attr("training_duration", duration)
        final_loss = history.history["loss"][-1]
        trial.set_user_attr("final_loss", final_loss)
    except KeyboardInterrupt:
        raise optuna.exceptions.TrialPruned()

    # penalty_multiplier = calculate_penalty_multiplier(cell_count)
    x_prediction = predict(VALIDATION_DATA, model, 5)
    mean_error = np.mean(np.abs(VALIDATION_DATA - x_prediction))
    # Checa se score é NaN
    # Se sim, retorna maxfloat para desmotivar uso de hiperparâmetros que causem esse comportamento
    if (np.isnan(mean_error)):
        return np.finfo(np.float32).max
    else:
        return mean_error


if __name__ == "__main__":
    # Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
    try:
        study = optuna.create_study(
            study_name='lstm_layer', direction='minimize', storage='sqlite:///lstm_layer.db')
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name='lstm_layer', storage='sqlite:///lstm_layer.db')

    study.optimize(objective, n_trials=29, timeout=60 * 30)
    exit(0)
