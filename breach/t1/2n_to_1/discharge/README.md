# _Discharge_

Tenta prever o próximo valor da vazão com base em $n$ valores anteriores da vazão **e** da altura da lâmina (portanto, $2n$).

Cada _Notebook_ gera:
- Model do Keras (representação visual)
- Histórico de _loss_ utilizando SGD e Adam
- Modelo HDF5 resultante do uso de ambos otimizadores
- Resultado das predições sobre os dados de treinamento e de teste para ambos otimizadores

Diretórios:
- **0_first_try**: primeira tentativa. 32 células LSTM na primeira camada e 4 na segunda.
- **1_very_low_lstm_count**: segunda tentativa. 4 células LSTM em cada camada.
- (...) A partir da terceira tentativa, os nomes são autoexplicativos.
