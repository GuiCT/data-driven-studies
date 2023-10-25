# legacy

Alguns estudos preliminares que não deram muito certo, mas eu mantive os arquivos para fins de registro (saber o que não dá certo é tão importante quanto saber o que dá certo!). Infelizmente os notebooks nãoe estão muito organizados e por isso não irão conter tudo que foi registrado nesse Markdown, pois muitas mudanças eram feitas *inplace*, sobrescrevendo testes anteriores.

---
- Previsão da altura da lâmina com base apenas na vazão
- Previsão da vazão com base apenas na altura da lâmina

Dentre os testes realizados, estão:
- Prever $n$ valores de uma série com base nos $n$ valores de uma outra série no mesmo intervalo de tempo (também chamado de problema _seq2seq_, utilizando células LSTM).
- Prever 1 valor de uma série com base em $n$ valores anteriores de outra série (problema mais comum em que as células LSTM são utilizadas, por conta de sua capacidade de aproveitar contexto anterior)
- Prever 1 valor de uma série com base em apenas 1 valor de outra série (utilizando uma abordagem _feed-forward_ tradicional)

Foram feitos três estudos Optuna para tentar encontrar um modelo viável nos casos de previsões _seq2seq_ e previsão do próximo valor com base em uma sequência anterior. Os valores estão disponíveis no banco de dados SQLite3 localizado nessa pasta: `study.db`, note que os três estudos ficam no mesmo banco de dados.

Também foram testados diferentes métodos de redução de ruídos, como Média móvel, suavização exponencial e filtro _Butterworth_ do tipo passa baixa, que elimina ruídos de alta frequência. O filtro _butterworth_ foi o que reduziu o ruído de maneira mais eficaz, sem perder muitas informações dos dados originais.

---
Apesar da dificuldade em obter um resultado com um _loss_ aceitável, esses primeiros experimentos ajudaram a desenvolver uma compreensão maior sobre tratamento de dados obtidos experimentalmente, assim como as limitações de utilizar células LSTM em conjuntos de dados não muito extensos, isto pois a grande maioria dos testes realizados com células LSTM apresentaram uma tendência de _overfitting_. Embora o _loss_ de treinamento não fosse muito bom também, o _loss_ de teste era muito mais elevado. 