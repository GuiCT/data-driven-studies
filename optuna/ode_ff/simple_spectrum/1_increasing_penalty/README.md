# 1_increasing_penalty

Segunda iteração do uso do Optuna para descobrimento de hiperparâmetros. Em relação ao que foi apresentado no [README.md anterior](../0_low_penalty//README.md), foram mudados:

- O *score* passou a ser calculado por:

$\mu(\hat{x}-x)\left(1 + \dfrac{\sum n - \min\sum n}{\max\sum n}\right)^2$

Dessa forma, o número de neurônios é mais penalizado.

- No *Jupyter Notebook*, o número de *epochs* utilizadas para treinamento, assim como a tolerância utilizada para interrupção por não redução do *loss* foram intensamente reduzidos, isto pois a primeira iteração demonstrou que esses modelos convergem de forma rápida, e um número muito maior de *epochs* do que o necessário acaba causando *overfitting* e tempos de treinamento muito longos.

O diretório possui a mesma estrutura do diretório citado no README.md anterior.