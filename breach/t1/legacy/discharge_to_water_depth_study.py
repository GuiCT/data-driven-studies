from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, InputLayer, Reshape, Bidirectional
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import optuna


depth_df = pd.read_csv('data/water_depth_reservoir_Test1.csv')
print(depth_df)


depth_df = depth_df.drop(columns=[depth_df.columns[-1]])
print(depth_df)


depth_df = depth_df.drop(columns=[depth_df.columns[0]])
print(depth_df)


discharge_df = pd.read_csv('data/discharge_inlet_Test1.csv')
print(discharge_df)


discharge_df = discharge_df.drop(columns=[discharge_df.columns[0]])
print(discharge_df)


unified_df = pd.concat([discharge_df, depth_df], axis=1)
print(unified_df)

EWM_ALPHA = 0.125
smoothed_df = unified_df.ewm(alpha=EWM_ALPHA).mean()
print(smoothed_df)


unified = smoothed_df.to_numpy()


min_max_scaler = MinMaxScaler()
unified = min_max_scaler.fit_transform(unified)
print(unified)


SEQUENCE_SIZE = 50
SEQUENCE_GAP = 25


i = 0
lower = list()
upper = list()
while i + SEQUENCE_SIZE < len(unified):
    lower.append(i)
    upper.append(i + SEQUENCE_SIZE)
    print(f"[{i}, {i + SEQUENCE_SIZE})")
    i += SEQUENCE_GAP


number_of_sequences = len(lower)
print("Quantidade de sequências", number_of_sequences)

TRAINING_SIZE = int(2 * number_of_sequences / 3)
print("Quantidade de sequências de treinamento", TRAINING_SIZE)

inputs = np.zeros((number_of_sequences, SEQUENCE_SIZE, 1))
outputs = np.zeros((number_of_sequences, SEQUENCE_SIZE, 1))
for i in range(number_of_sequences):
    inputs[i, :] = unified[lower[i]:upper[i], 0].reshape(SEQUENCE_SIZE, 1)
    outputs[i, :] = unified[lower[i]:upper[i], 1].reshape(SEQUENCE_SIZE, 1)

inputs_train = inputs[:TRAINING_SIZE]
outputs_train = outputs[:TRAINING_SIZE]
inputs_test = inputs[TRAINING_SIZE:]
outputs_test = outputs[TRAINING_SIZE:]
print("Shapes", inputs_train.shape, outputs_train.shape,
      inputs_test.shape, outputs_test.shape)


def optuna_trial(trial: optuna.Trial):
    number_of_lstm_layers = trial.suggest_int('number_of_hidden_layers', 1, 3)
    return_sequences_last = trial.suggest_categorical(
        'return_sequences_last', [True, False])
    generated_model = Sequential()
    generated_model.add(InputLayer(
        input_shape=(SEQUENCE_SIZE, 1), name='input'))
    for i in range(number_of_lstm_layers - 1):
        is_bidirectional = trial.suggest_categorical(
            f'lstm_{i}_bidirectional', [True, False])
        number_of_neurons = trial.suggest_int(
            f'lstm_{i}_units', 4, 64)
        lstm_layer = LSTM(
            number_of_neurons,
            activation=trial.suggest_categorical(
                f'lstm_{i}_activation_function', ['tanh', 'relu', 'linear', 'sigmoid']),
            return_sequences=True,
            name=f'lstm_{i}'
        )
        if is_bidirectional:
            lstm_layer = Bidirectional(lstm_layer, name=f'bidirectional_{i}')
        generated_model.add(lstm_layer)
    is_bidirectional = trial.suggest_categorical(
        f'lstm_{number_of_lstm_layers - 1}_bidirectional', [True, False])
    number_of_neurons = trial.suggest_int(
        f'lstm_{number_of_lstm_layers - 1}_units', 4, 64)
    lstm_layer = LSTM(
        number_of_neurons,
        activation=trial.suggest_categorical(
            f'lstm_{number_of_lstm_layers - 1}_activation_function',
            ['tanh', 'relu', 'linear', 'sigmoid']
        ),
        return_sequences=return_sequences_last,
        # return_sequences=False,
        name=f'lstm_{number_of_lstm_layers - 1}'
    )
    if is_bidirectional:
        lstm_layer = Bidirectional(lstm_layer, name=f'bidirectional_{number_of_lstm_layers - 1}')
    generated_model.add(lstm_layer)
    if return_sequences_last:
        generated_model.add(Dense(
            1, trial.suggest_categorical(
                'dense_activation_function', ['tanh', 'relu', 'linear', 'sigmoid']),
            name='dense')
        )
    else:
        generated_model.add(Dense(
            SEQUENCE_SIZE, trial.suggest_categorical(
                'dense_activation_function', ['tanh', 'relu', 'linear', 'sigmoid']),
            name='dense')
        )
        generated_model.add(Reshape(
            (SEQUENCE_SIZE, 1), name='reshape'))
    # generated_model.add(Dense(
    #     1, trial.suggest_categorical(
    #         'dense_activation_function', ['linear', 'relu']),
    #     name='dense')
    # )
    generated_model.compile(loss='mse', optimizer='adam')
    generated_model.summary()
    history = generated_model.fit(
        inputs_train, outputs_train,
        epochs=50, batch_size=16,
        shuffle=True, validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
    )
    best_training_loss = sorted(history.history['loss'])[0]
    best_val_loss = sorted(history.history['val_loss'])[0]
    trial.set_user_attr('best_training_loss', best_training_loss)
    trial.set_user_attr('best_val_loss', best_val_loss)
    number_of_parameters = generated_model.count_params()
    trial.set_user_attr('number_of_parameters', number_of_parameters)
    return best_val_loss, number_of_parameters


if __name__ == '__main__':
    study = optuna.create_study(
        study_name='discharge_to_water_depth_seq2seq',
        directions=['minimize', 'minimize'],
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        storage='sqlite:///study.db',
        load_if_exists=True
    )
    study.set_user_attr('SEQUENCE_SIZE', SEQUENCE_SIZE)
    study.set_user_attr('SEQUENCE_GAP', SEQUENCE_GAP)
    study.set_user_attr('EMW_ALPHA', EWM_ALPHA)
    study.optimize(optuna_trial, n_trials=100)
