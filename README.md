# Optimization_GradientBoosting

Neste trabalho abordamos o Gradient Boosting e algumas modificações. 

Este método consiste em uma técnica de aprendizagem de máquinas supervisionado para problemas tanto de classificação como de regressão que produz um modelo de previsão na forma de um ensemble de modelos de previsão fracos, geralmente árvores de decisão. Ele constrói o modelo em etapas, como outros métodos de boosting, e os generaliza. Tal estratégia utiliza ideias provenientes do método do gradiente descendente com o intuito de minimizar uma função de perda associada ao problema.

Tendo em vista a possível lentidão na convergência do método do gradiente descendente, estudamos artigos que propõem modificações do gradiente boosting baseadas em dois algoritmos conhecidos da área de Otimização: o Momentum e o método de aceleração de Nesterov. O intuito de tais modificações é tentar acelerar a convergência, ou seja, utilizar menos árvores, porém mantendo o bom desempenho do gradient boosting e sem aumentar a variância. 

**Referências**:

FENG, Z.; XU, C.; TAO, D. **Historical Gradient Boosting Machine**. EPiC Series Incomputing. 2018. p(68-80).

FRIEDMAN, J.; HASTIE, T.; TIBSHIRANI, R. **The Elements of Statistical Learning**: Data Mining, Inference, and Prediction. 2. ed. New York: Springer, 2009.

JAMES, G.; WITTEN, D.; HASTIE, T.; TIBSHIRANI, R. **An Introduction to Statistical Learning**: With Applicaton in R. 1. ed. New York: Springer, 2013.
