import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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

# Definir os parâmetros e os ranges para testar
depth_range = np.arange(1, 20, 1)
split_range = np.arange(2, 20, 1)
leaf_range = np.arange(1, 20, 1)

# Função para treinar e avaliar o modelo
def treinar_e_avaliar(max_depth, min_samples_split, min_samples_leaf, x_treino, y_treino, x_val, y_val):
    modelo = DecisionTreeClassifier(max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
    modelo.fit(x_treino, y_treino)

    score_train = modelo.score(x_treino, y_treino)
    score_val = modelo.score(x_val, y_val)

    return (max_depth, min_samples_split, min_samples_leaf, score_train, score_val)

# Usar joblib.Parallel para paralelizar o processo
resultados = Parallel(n_jobs=-1)(delayed(treinar_e_avaliar)(max_depth, min_samples_split, min_samples_leaf, x_treino, y_treino, x_val, y_val)
                                for max_depth in depth_range
                                for min_samples_split in split_range
                                for min_samples_leaf in leaf_range)

# Separar os resultados de treino e validação
scores_train = np.array([(r[0], r[1], r[2], r[3]) for r in resultados])
scores_val = np.array([(r[0], r[1], r[2], r[4]) for r in resultados])

# Gerar o gráfico para visualizar a performance em relação aos parâmetros
plt.figure(figsize=(10, 6))
plt.plot(scores_val[:, 3], label="Validação")
plt.plot(scores_train[:, 3], label="Treino")
plt.xlabel('Combinacão de parâmetros')
plt.ylabel('Score')
plt.legend()
plt.title('Score em relação aos parâmetros (max_depth, min_samples_split e min_samples_leaf)')
plt.show()

# Encontrar o ponto máximo de performance na validação
max_index = np.argmax(scores_val[:, 3])
best_params = scores_val[max_index, :3]
print(f"Melhor combinação de parâmetros: max_depth={int(best_params[0])}, min_samples_split={int(best_params[1])}, min_samples_leaf={int(best_params[2])}")
print(f"Melhor score de validação: {scores_val[max_index, 3]}")