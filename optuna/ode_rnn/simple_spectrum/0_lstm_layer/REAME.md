# 0_lstm_layer

Essa foi a primeira rede LSTM modelada via otimização de hiperparâmetros, ainda utilizando o Optuna.

O _score_ dessa vez não aplica penalização por número de células, o que impacta de forma considerável o resultado final apresentado pelo _Notebook_ onde é realizado o treinamento e teste da rede modelada.

---

Nesse diretório:

- run.py: contém o _script_ utilizado para configurar e executar as _trials_ utilizando o _framework_ do Optuna.
- train.ipynb: contém o _Jupyter Notebook_ avaliando os resultados obtidos a partir dos hiperparâmetros dados pelo estudo.
- trained_models: diretório contendo os modelos treinados salvos.
- predictions_result: diretório contendo os resultados preditos salvos.
