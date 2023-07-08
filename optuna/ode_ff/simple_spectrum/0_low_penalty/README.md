# 0_low_penalty

Essa foi a primeira tentativa de utilizar o Optuna para descobrimento de hiperparâmetros.

Foi utilizado o conceito de penalização de *score* por números de neurônios presentes no modelo.

Dessa forma, o *score* de cada *trial* do Optuna é calculado por:

$\mu(\hat{x}-x)\left(1 + \dfrac{\sum n}{\max\sum n}\right)$, onde:
- $\mu(\hat{x} - x)$ indica a **média aritmética** da diferença entre o valor predito pela rede neural e o valor calculado utilizando *scipy.integrate*
- $\sum n$ representa a soma dos neurônios nas camadas ocultas
- $\max\sum n$ representa o valor máximo da soma dos neurônios nas camadas ocultas

Nota-se que o penalty não é zerado quando o número mínimo de neurônios é utilizado, isso é corrigido na próxima iteração do estudo (**1_increasing_penalty**).

---
Nesse diretório:
- run.py: contém o *script* utilizado para configurar e executar as *trials* utilizando o *framework* do Optuna.
- train.ipynb: contém o *Jupyter Notebook* avaliando os resultados obtidos a partir dos hiperparâmetros dados pelo estudo.
- ode_feed_forward.db: Banco de Dados SQLite contendo informação de todas as *trials* realizadas, das melhores às piores.
- trained_models: diretório contendo os modelos treinados salvos e seus respectivos tempos de treinamento, em segundos.
- predictions_result: diretório contendo os resultados preditos salvos 