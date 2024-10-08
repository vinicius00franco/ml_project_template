# Documentação do Pipeline de Modelos Supervisionados

## Visão Geral

Este pipeline foi criado para treinar e avaliar modelos supervisionados usando um processo completo de carregamento de dados, tratamento de desbalanceamento de classes, treinamento de modelos e avaliação de performance com métricas e gráficos. O pipeline também permite o salvamento do modelo treinado.

---

# Fluxo de Funcionamento do Modelo

O fluxo de funcionamento do modelo implementado no código pode ser dividido em etapas claras, desde o carregamento dos dados até a avaliação e salvamento do modelo. Abaixo está uma explicação passo a passo do processo:

## 1. Carregamento de Dados (DataLoader)

A classe `DataLoader` é responsável por carregar os dados do arquivo CSV fornecido pelo caminho (`filepath`). Essa classe também permite a separação dos dados em conjuntos de treino, validação e teste.

- A função `split_data()` divide o conjunto de dados de forma estratificada para manter a proporção das classes.
  - Os dados são inicialmente divididos em treino e teste, e em seguida, o conjunto de treino é dividido em treino e validação.

## 2. Construção do Modelo (ModelBuilder)

A classe `ModelBuilder` define o tipo de modelo a ser utilizado (Árvore de Decisão ou Random Forest). O usuário pode escolher o tipo de modelo e o parâmetro `max_depth` para controlar a profundidade do modelo.

- A função `build_pipeline()` permite configurar uma pipeline de processamento de dados e modelo, com estratégias de balanceamento de classes como **oversampling** (com SMOTE) ou **undersampling** (com NearMiss).
  - Essas técnicas são úteis quando há desbalanceamento entre as classes no conjunto de dados.
- A função `train_model()` é responsável pelo treinamento do modelo com os dados de treino.
- A função `cross_validate_model()` realiza validação cruzada para medir a performance do modelo em diferentes partes do conjunto de dados.

## 3. Treinamento e Avaliação do Modelo (TrainerWorkflow)

A classe `TrainerWorkflow` integra os componentes anteriores e controla o processo completo.

- O pipeline é definido, os dados são carregados, e o modelo é treinado.
- Após o treinamento, a classe `ModelEvaluator` é usada para avaliar a performance do modelo em termos de métricas como:

  - Acurácia
  - F1 Score
  - Precisão
  - Recall
  - ROC AUC

- Durante a avaliação, gráficos de **matriz de confusão** e **curva ROC** são gerados e salvos automaticamente em arquivos PNG, facilitando a visualização dos resultados.

## 4. Salvamento do Modelo (ModelSaver)

Após o treinamento e avaliação, se o parâmetro `save_model` for **True**, o modelo treinado é salvo em um arquivo `.pkl` usando a classe `ModelSaver`.

- O arquivo é nomeado de acordo com o tipo de modelo, estratégia de balanceamento e a data/hora do treinamento.
- O modelo salvo pode ser carregado posteriormente para fazer previsões em novos dados, utilizando o método `load_model()`.

## 5. Execução do Pipeline

A função `run_pipeline()` é a entrada principal do sistema, onde todo o processo é coordenado.

- Ela define os componentes, como o carregador de dados, o modelo e o avaliador, e executa o treinamento e avaliação do modelo.
- O pipeline também permite que o modelo seja salvo após o treinamento, dependendo da escolha do usuário.

### 1. **Carregamento de Dados (`DataLoader`)**

- **Responsabilidade**: Carregar os dados do arquivo CSV e dividi-los em conjuntos de treino, validação e teste.
- **Métodos**:
  - `load_data()`: Carrega os dados a partir do caminho fornecido (arquivo CSV).
  - `split_data()`: Divide os dados em três conjuntos: treino, validação e teste. A divisão é estratificada para manter a proporção entre as classes.
- **Parâmetros**:
  - `x`: Variáveis independentes (features).
  - `y`: Variável dependente (label).
  - `test_size`: Proporção dos dados reservada para o teste (padrão: 15%).
  - `val_size`: Proporção dos dados de treino reservada para a validação (padrão: 15%).

### 2. **Construção do Modelo (`ModelBuilder`)**

- **Responsabilidade**: Definir o modelo e o processo de balanceamento de dados (oversampling ou undersampling).
- **Métodos**:
  - `__init__(model_type, max_depth)`: Inicializa o tipo de modelo (Árvore de Decisão ou Random Forest).
  - `build_pipeline(balance_strategy)`: Cria uma pipeline com uma estratégia de balanceamento (oversampling ou undersampling) e o modelo.
  - `train_model(x_train, y_train)`: Treina o modelo com os dados de treino.
  - `cross_validate_model(x, y, scoring)`: Executa validação cruzada para avaliar o modelo.
- **Parâmetros**:
  - `model_type`: Tipo de modelo (Árvore de Decisão ou Random Forest).
  - `balance_strategy`: Estratégia de balanceamento de dados (oversample ou undersample).
  - `x_train`: Conjunto de treino.
  - `y_train`: Labels de treino.

### 3. **Treinamento e Avaliação do Modelo (`TrainerWorkflow`)**

- **Responsabilidade**: Coordenar o fluxo de treinamento e avaliação do modelo.
- **Métodos**:
  - `train_and_evaluate()`: Carrega os dados, divide-os, treina o modelo e realiza a avaliação.
  - O modelo é treinado com o conjunto de treino e avaliado com o conjunto de validação.
- **Métricas de Avaliação**:
  - Acurácia
  - F1 Score
  - Precisão
  - Recall
  - ROC AUC
- **Visualizações**:
  - Matriz de Confusão
  - Curva ROC

### 4. **Salvamento do Modelo (`ModelSaver`)**

- **Responsabilidade**: Salvar o modelo treinado e carregar um modelo salvo.
- **Métodos**:
  - `save_model(model, model_type, balance_strategy, filename)`: Salva o modelo treinado em um arquivo `.pkl`.
  - `load_model(filename)`: Carrega um modelo previamente salvo.
- **Parâmetros**:
  - `model`: O modelo treinado a ser salvo.
  - `model_type`: Tipo de modelo (Árvore de Decisão ou Random Forest).
  - `balance_strategy`: Estratégia de balanceamento utilizada (oversample ou undersample).
  - `filename`: Nome do arquivo para salvar o modelo.

---
