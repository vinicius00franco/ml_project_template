import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt

import pandas as pd


# Carregando os dados
dados = pd.read_csv('data/emp_automovel.csv')

# Separando as variáveis
x = dados.drop('inadimplente', axis=1)
y = dados['inadimplente']

# Dividindo os dados em treino, validação e teste
x, x_teste, y, y_teste = train_test_split(
    x, y, test_size=0.15, stratify=y, random_state=5)
x_treino, x_val, y_treino, y_val = train_test_split(
    x, y, stratify=y, random_state=5
    )

# Definir os ranges de parâmetros para busca aleatória
param_distributions = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

# Criando o modelo de DecisionTree
modelo = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5)

# Utilizar uma amostra menor dos dados (30% dos dados de treino)
x_treino_sample, _, y_treino_sample, _ = train_test_split(x_treino, y_treino, train_size=0.3, stratify=y_treino)

# Configurando o RandomizedSearchCV com 100 iterações e validação cruzada de 3 folds
random_search = RandomizedSearchCV(
    modelo,
    param_distributions=param_distributions,
    n_iter=1000,  # Número de iterações aleatórias
    cv=3,  # Validação cruzada com 3 folds
    n_jobs=-1,  # Utilizar todos os núcleos disponíveis
    random_state=42,
    return_train_score=True 
)

# Treinando o modelo com a busca aleatória nos parâmetros
random_search.fit(x_treino_sample, y_treino_sample)

# Visualizando os melhores parâmetros e score
print("Melhores parâmetros encontrados:", random_search.best_params_)
print("Melhor score de validação:", random_search.best_score_)

# Gráfico de comparação entre os scores de treino e validação
scores_train = random_search.cv_results_['mean_train_score']
scores_val = random_search.cv_results_['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(scores_val, label="Validação")
plt.plot(scores_train, label="Treino")
plt.xlabel('Combinacão de parâmetros (Random Search)')
plt.ylabel('Score')
plt.legend()
plt.title('Score em relação aos parâmetros (Random Search)')
plt.show()
