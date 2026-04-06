# GalaxyNet — Classificacao Morfologica de Galaxias

Pipeline completo de aprendizado profundo para classificacao morfologica de galaxias
utilizando fotometria do SDSS DR17 e rotulos do Galaxy Zoo 2 (GZ2).
Tres arquiteturas sao treinadas e comparadas: **MLP**, **CNN** e **Hibrido (CNN + MLP)**.

> Projeto academico — UESC, orientador Prof. Andre Ribeiro.

---

## Indice

1. [Visao Geral do Projeto](#visao-geral-do-projeto)
2. [Requisitos](#requisitos)
3. [Instalacao](#instalacao)
4. [Estrutura de Diretorios](#estrutura-de-diretorios)
5. [Pipeline Completo](#pipeline-completo)
   - [Etapa 1 — Download da Fotometria SDSS](#etapa-1--download-da-fotometria-sdss)
   - [Etapa 2 — Cross-Match SDSS + Galaxy Zoo 2](#etapa-2--cross-match-sdss--galaxy-zoo-2)
   - [Etapa 3 — Download das Imagens](#etapa-3--download-das-imagens)
   - [Etapa 4 — Analise Exploratoria (EDA)](#etapa-4--analise-exploratoria-eda)
   - [Etapa 5 — Preprocessing](#etapa-5--preprocessing)
   - [Etapa 6 — Modelo MLP](#etapa-6--modelo-mlp)
   - [Etapa 7 — Modelo CNN](#etapa-7--modelo-cnn)
   - [Etapa 8 — Modelo Hibrido](#etapa-8--modelo-hibrido)
   - [Etapa 9 — Comparacao de Modelos](#etapa-9--comparacao-de-modelos)
6. [Arquiteturas dos Modelos](#arquiteturas-dos-modelos)
7. [Metricas de Avaliacao](#metricas-de-avaliacao)
8. [Dataset](#dataset)
9. [Figuras Geradas](#figuras-geradas)

---

## Visao Geral do Projeto

O objetivo e classificar galaxias em tres classes morfologicas — **Elliptical**,
**Spiral** e **Irregular** — a partir de dados fotometricos e imagens do
Sloan Digital Sky Survey (SDSS DR17), combinados com rotulos de consenso do
Galaxy Zoo 2.

O pipeline cobre todas as etapas, desde a aquisicao de dados ate a avaliacao
comparativa dos modelos, incluindo visualizacao via Grad-CAM para
interpretabilidade.

### Fluxo resumido

```
SDSS DR17 Query ─┐
                  ├─> Cross-Match (1") ─> Catalogo ─> Download Imagens
GZ2 (Zenodo)   ──┘                         │                │
                                            v                v
                                    Features Tabulares   Imagens (64x64x3)
                                            │                │
                                    ┌───────┴───────┐        │
                                    v               v        v
                                   MLP            CNN    Hibrido (CNN+MLP)
                                    │               │        │
                                    └───────┬───────┘        │
                                            v                v
                                      Comparacao de Modelos (nb06)
```

---

## Requisitos

- **Python** 3.9+
- **Sistema operacional**: Linux, macOS ou Windows
- **Conexao com internet** para download de dados do SDSS e imagens FITS
- **GPU** (opcional): recomendado para treinamento dos modelos CNN e Hibrido

### Dependencias principais

| Pacote | Uso |
|---|---|
| TensorFlow >= 2.8 | Definicao e treinamento dos modelos |
| Astropy / Astroquery | Consultas SQL ao SDSS, download de FITS |
| Pandas / NumPy | Processamento de dados tabulares |
| Scikit-learn / imbalanced-learn | Scaling, split, class weights |
| Matplotlib / Seaborn | Graficos e visualizacoes |
| OpenCV / Photutils | Preprocessamento de imagens |

---

## Instalacao

### 1. Clonar o repositorio

```bash
git clone https://github.com/<seu-usuario>/galaxy_classification.git
cd galaxy_classification
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv galaxy_env
source galaxy_env/bin/activate   # Linux/macOS
# galaxy_env\Scripts\activate    # Windows
```

### 3. Instalar as dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar a instalacao

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import astroquery; print('Astroquery OK')"
python -c "import sklearn; print('Scikit-learn OK')"
```

Se voce tiver GPU disponivel, verifique:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 5. Obter o catalogo Galaxy Zoo 2

Baixe manualmente o arquivo `gz2_hart16.csv` do Zenodo e coloque-o em `data/raw/`:

```bash
mkdir -p data/raw
# Baixe de: https://zenodo.org/record/3565489
# Coloque o arquivo gz2_hart16.csv em data/raw/
```

---

## Estrutura de Diretorios

```
galaxy_classification/
├── download_sdss.py            # Etapa 1: query fotometria SDSS DR17
├── load_gz2_and_merge.py       # Etapa 2: cross-match SDSS + GZ2
├── download_images.py          # Etapa 3: download imagens FITS
├── requirements.txt            # Dependencias Python
├── README.md                   # Este arquivo
│
├── src/                        # Biblioteca reutilizavel
│   ├── data_loader.py          #   Download, cross-match, merge
│   ├── preprocessing.py        #   Feature engineering, image preprocessing
│   ├── models.py               #   MLP, CNN, Hybrid (Keras)
│   ├── evaluation.py           #   Metricas, classification report, confusion matrix
│   └── visualization.py        #   Grad-CAM (single e dual-input)
│
├── notebooks/                  # Workflow sequencial (Jupyter)
│   ├── 01_eda.ipynb            #   Analise exploratoria
│   ├── 02_preprocessing.ipynb  #   Feature engineering + image preprocessing
│   ├── 03_mlp_model.ipynb      #   Treinamento MLP
│   ├── 04_cnn_model.ipynb      #   Treinamento CNN
│   ├── 05_hybrid_model.ipynb   #   Treinamento Hibrido + Grad-CAM
│   └── 06_model_comparison.ipynb  # Comparacao final dos 3 modelos
│
├── data/
│   ├── raw/                    # sdss_galaxies.csv, gz2_hart16.csv
│   ├── processed/              # merged_catalog.csv, arrays .npy, scaler.pkl
│   └── images/                 # <objid>.npy (64x64x3, float32, bandas g,r,i)
│
├── models/                     # Modelos treinados (.h5), LabelEncoder, test data
└── reports/
    └── figures/                # Figuras fig01–fig25 (.png, .csv)
```

---

## Pipeline Completo

Execute cada etapa na ordem indicada. Os scripts de linha de comando sao
executados no terminal; os notebooks no Jupyter.

---

### Etapa 1 — Download da Fotometria SDSS

```bash
python download_sdss.py
```

**O que faz:**
Executa uma query SQL ao SDSS DR17 SkyServer buscando dados fotometricos
e espectroscopicos de galaxias numa regiao do ceu. Os parametros de busca
estao definidos como constantes no script:

| Parametro | Valor padrao | Descricao |
|---|---|---|
| `RA_CENTER` | 180.0 | Ascensao reta central (graus) |
| `DEC_CENTER` | 0.0 | Declinacao central (graus) |
| `RADIUS_DEG` | 2.0 | Raio de busca (graus) |
| `MAX_RECORDS` | 1000 | Limite de registros (usar 5000+ para versao final) |

**Colunas obtidas:** `objid`, `ra`, `dec`, magnitudes `u/g/r/i/z`,
`redshift`, `velDisp`, parametros de perfil (`deVAB_r`, `expAB_r`,
`fracDeV_r`, `petroR50_r`, `petroR90_r`), entre outros.

**Saida:** `data/raw/sdss_galaxies.csv`

Para ajustar a regiao do ceu, edite as constantes no inicio do script.

---

### Etapa 2 — Cross-Match SDSS + Galaxy Zoo 2

```bash
python load_gz2_and_merge.py
```

**Pre-requisito:** o arquivo `data/raw/gz2_hart16.csv` deve estar presente
(download manual do Zenodo — veja secao [Instalacao](#instalacao)).

**O que faz:**
1. Carrega o catalogo SDSS (`sdss_galaxies.csv`) e o GZ2 (`gz2_hart16.csv`)
2. Realiza cross-match espacial com tolerancia de **1 arcseg** usando
   `SkyCoord.match_to_catalog_sky()` — necessario porque o GZ2 usa objids
   do DR7 enquanto o pipeline usa o DR17
3. Atribui classes morfologicas baseadas nas fracoes de voto do GZ2:
   - **Elliptical**: `t01_smooth_or_features_a01_smooth_fraction >= 0.8`
   - **Spiral**: `t01_smooth_or_features_a02_features_or_disk_fraction >= 0.8`
   - **Irregular**: `t06_odd_a01_yes_fraction >= 0.8`
   - Thresholds: `min_vote_fraction=0.8`, `min_total_votes=20`
   - Entradas classificadas como `Uncertain` sao removidas
4. Salva o catalogo mesclado

**Saida:** `data/processed/merged_catalog.csv` (~156 galaxias, 33 colunas)

---

### Etapa 3 — Download das Imagens

```bash
python download_images.py --max 500
```

Ou para baixar todas as imagens do catalogo:

```bash
python download_images.py
```

**O que faz:**
Para cada galaxia no catalogo mesclado, baixa recortes FITS nas bandas
`g`, `r`, `i` do SDSS SkyServer e salva como arrays NumPy.

| Parametro | Descricao |
|---|---|
| `--max N` | Limite de imagens a baixar (padrao: 500). Omita para baixar todas |

- Formato: `float32`, shape `(64, 64, 3)`, canais = bandas `[g, r, i]`
- Imagens ja baixadas sao automaticamente ignoradas (skip-existing)
- Cada imagem e salva como `data/images/<objid>.npy`

**Saida:** `data/images/<objid>.npy` (um arquivo por galaxia)

---

### Etapa 4 — Analise Exploratoria (EDA)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**O que faz:**
Exploracao detalhada do catalogo mesclado:

1. **Estatisticas descritivas** — resumo numerico de todas as colunas
2. **Distribuicao de classes** — grafico de barras e pizza mostrando o
   desbalanceamento severo: Elliptical ~86%, Spiral ~11%, Irregular ~3%
3. **Distribuicao de redshift** — histograma geral e por classe
4. **Indices de cor** — calcula `u-g`, `g-r`, `r-i`, `i-z` e indice de
   concentracao `C = petroR90_r / petroR50_r`
5. **Diagramas cor-magnitude** — `g-r` vs `r` e `u-r` vs `r`, com
   separacao por classe
6. **Boxplots por classe** — compara distribuicoes de features entre classes
7. **Heatmap de correlacao** — identifica relacoes entre as 15 features
8. **Dispersao de velocidade** — `velDisp` vs `redshift`

**Figuras geradas:** `fig01` a `fig06` em `reports/figures/`

**Insight principal:** galaxias Elliptical tem `g-r` medio de 0.94, Spiral
0.68 e Irregular 0.82 — forte separacao por indice de cor.

---

### Etapa 5 — Preprocessing

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**O que faz:**
Prepara os dados tabulares e de imagem para treinamento:

**Features tabulares (15 features):**

| Grupo | Features |
|---|---|
| Magnitudes brutas | `u`, `g`, `r`, `i`, `z` |
| Indices de cor (derivados) | `u_g`, `g_r`, `r_i`, `i_z` |
| Espectroscopicas | `redshift`, `velDisp` |
| Morfologicas/estruturais | `concentration`, `deVAB_r`, `expAB_r`, `fracDeV_r` |

- Feature engineering: calcula indices de cor e concentracao
- Normalizacao com `StandardScaler` (media=0, desvio=1)
- Remove linhas com valores ausentes

**Imagens:**

1. **Arcsinh stretch**: `arcsinh(pixel / 1000)` — realca features difusas sem
   saturar nucleos brilhantes
2. **Min-max por canal**: normaliza cada banda para `[0, 1]`
3. **Resize** para `64 x 64` pixels

O notebook inclui visualizacoes comparativas do efeito do stretch e um grid
de galaxias RGB por classe.

**Artefatos salvos em `data/processed/`:**

| Arquivo | Shape | Descricao |
|---|---|---|
| `X_tabular.npy` | `(156, 15)` | Features tabulares normalizadas |
| `y_labels.npy` | `(156,)` | Labels como string |
| `objids.npy` | `(156,)` | Object IDs alinhados |
| `scaler.pkl` | — | StandardScaler ajustado |
| `X_images.npy` | `(156, 64, 64, 3)` | Imagens preprocessadas |
| `img_objids.npy` | `(156,)` | Object IDs das imagens |

**Figuras geradas:** `fig07` a `fig15`

---

### Etapa 6 — Modelo MLP

```bash
jupyter notebook notebooks/03_mlp_model.ipynb
```

**O que faz:**
Treina um Multi-Layer Perceptron usando apenas as 15 features tabulares.

**Configuracao do treinamento:**

| Parametro | Valor |
|---|---|
| Split | 70% train / 15% val / 15% test (estratificado) |
| Batch size | 32 |
| Epocas maximas | 100 |
| EarlyStopping | patience=10, restore_best_weights |
| ReduceLROnPlateau | patience=5, factor=0.5, min_lr=1e-7 |
| Class weights | `balanced` (compensa desbalanceamento) |

**Artefatos salvos:**

| Arquivo | Descricao |
|---|---|
| `models/mlp_galaxy_classifier.h5` | Modelo treinado |
| `models/label_encoder.pkl` | LabelEncoder (reutilizado nos notebooks 04 e 05) |
| `models/mlp_test_data.pkl` | Dados de teste para comparacao |
| `fig16_mlp_training_curves.png` | Curvas de loss e accuracy |
| `fig17_mlp_confusion_matrix.png` | Matriz de confusao |

O notebook exibe o classification report completo e as metricas cientificas
(Completeness e Reliability) por classe.

---

### Etapa 7 — Modelo CNN

```bash
jupyter notebook notebooks/04_cnn_model.ipynb
```

**O que faz:**
Treina uma Rede Neural Convolucional usando as imagens `(64, 64, 3)`.

**Data Augmentation:**
Galaxias nao possuem orientacao preferencial no ceu, o que justifica:

| Transformacao | Valor | Justificativa |
|---|---|---|
| Rotacao | 360 graus | Qualquer angulo e fisicamente valido |
| Flip horizontal | Sim | Simetria do ceu |
| Flip vertical | Sim | Simetria do ceu |
| Zoom | 0.8 - 1.2 | Simula diferentes distancias angulares |
| Brilho | 0.8 - 1.2 | Simula variacoes de exposicao |
| Fill mode | `constant` (0.0) | Bordas preenchidas com ceu escuro |

**Configuracao do treinamento:**

| Parametro | Valor |
|---|---|
| Split | 70/15/15 estratificado |
| Batch size | 32 |
| Epocas maximas | 100 |
| EarlyStopping | patience=15, restore_best_weights |
| ReduceLROnPlateau | patience=7, factor=0.5 |
| Class weights | `balanced` |

**Artefatos salvos:**

| Arquivo | Descricao |
|---|---|
| `models/cnn_galaxy_classifier.h5` | Modelo treinado |
| `models/cnn_test_data.pkl` | Dados de teste |
| `fig18_cnn_data_augmentation.png` | Exemplos de augmentation |
| `fig19_cnn_training_curves.png` | Curvas de treinamento |
| `fig17_cnn_confusion_matrix.png` | Matriz de confusao |

---

### Etapa 8 — Modelo Hibrido

```bash
jupyter notebook notebooks/05_hybrid_model.ipynb
```

**O que faz:**
Treina um modelo de **fusao tardia** (*late fusion*) que combina:
- **Branch CNN** — extrai features espaciais das imagens
- **Branch MLP** — processa features tabulares fotometricas
- **Fusao** — concatena as representacoes e classifica

**Pipeline especifico:**

1. Carrega dados tabulares e imagens simultaneamente, alinhados por `objid`
2. Usa o `LabelEncoder` do notebook 03 (consistencia entre modelos)
3. Split por indices compartilhados — garante que a mesma galaxia esta no
   mesmo conjunto (train/val/test) para ambas as modalidades
4. Generator customizado `hybrid_generator` — aplica data augmentation
   **apenas nas imagens**, passando features tabulares sem alteracao
5. Mesmos callbacks e class weights da CNN

**Grad-CAM (interpretabilidade):**

Apos o treinamento, o notebook aplica **Grad-CAM** (Gradient-weighted Class
Activation Mapping) na ultima camada convolucional do branch CNN. Isso gera
mapas de calor que mostram as regioes da imagem mais relevantes para a
decisao do modelo:

- **Elliptical**: ativacao concentrada no nucleo (distribuicao suave de luz)
- **Spiral**: ativacao nos bracos espirais e regioes de formacao estelar
- **Irregular**: ativacao dispersa e assimetrica

**Artefatos salvos:**

| Arquivo | Descricao |
|---|---|
| `models/hybrid_galaxy_classifier.h5` | Modelo treinado |
| `models/hybrid_test_data.pkl` | Dados de teste (tabular + imagem) |
| `fig20_hybrid_training_curves.png` | Curvas de treinamento |
| `fig17_hybrid_confusion_matrix.png` | Matriz de confusao |
| `fig21_hybrid_gradcam.png` | Grid de Grad-CAM por classe |

---

### Etapa 9 — Comparacao de Modelos

```bash
jupyter notebook notebooks/06_model_comparison.ipynb
```

**O que faz:**
Carrega os tres modelos treinados e seus dados de teste (sem retreinar) e
produz uma comparacao unificada:

1. **Tabela comparativa** — Accuracy, F1 Macro, F1 Weighted e metricas
   por classe (Completeness/Reliability) lado a lado
2. **Barplot agrupado** — Completeness e Reliability por classe e modelo;
   permite visualizar se o Hybrid recupera melhor Spiral e Irregular
3. **Confusion matrices side-by-side** — tres matrizes na mesma escala de
   cor para comparacao visual direta dos padroes de erro
4. **Grafico de accuracy/F1** — resumo geral do desempenho
5. **Classification reports** — relatorios detalhados por modelo

**Figuras geradas:**

| Arquivo | Descricao |
|---|---|
| `fig22_model_comparison_table.csv` | Tabela de metricas (CSV) |
| `fig23_completeness_reliability_comparison.png` | Barplot por classe |
| `fig24_confusion_matrices_comparison.png` | Confusion matrices 1x3 |
| `fig25_accuracy_f1_comparison.png` | Accuracy e F1 por modelo |

---

## Arquiteturas dos Modelos

### MLP (Multi-Layer Perceptron)

```
Input(15) --> Dense(256)/ReLU --> BN --> Dropout(0.3)
          --> Dense(128)/ReLU --> BN --> Dropout(0.3)
          --> Dense(64)/ReLU  --> BN --> Dropout(0.2)
          --> Dense(3, softmax)
```

- **Input**: 15 features tabulares normalizadas
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy

### CNN (Convolutional Neural Network)

```
Input(64,64,3) --> [Conv2D(32) --> BN --> MaxPool --> Dropout(0.25)] x1
               --> [Conv2D(64) --> BN --> MaxPool --> Dropout(0.25)] x1
               --> [Conv2D(128)--> BN --> MaxPool --> Dropout(0.25)] x1
               --> [Conv2D(256)--> BN --> MaxPool --> Dropout(0.25)] x1
               --> Flatten --> Dense(512)/ReLU --> BN --> Dropout(0.5)
               --> Dense(3, softmax)
```

- **Input**: imagens 64x64x3 (bandas g, r, i)
- **Filtros**: 3x3, padding='same'
- **Pooling**: MaxPool 2x2

### Hibrido (CNN + MLP — Late Fusion)

```
Image Input(64,64,3) ──> CNN Branch (4 conv blocks) ──> Dense(256) ─┐
                                                                     ├─> Concatenate
Tabular Input(15)    ──> Dense(128) --> Dense(64)  ─────────────────┘
                                                                     │
                                                              Dense(128)/ReLU/BN/Dropout
                                                                     │
                                                              Dense(3, softmax)
```

- **Dois inputs**: `tabular_input` e `image_input`
- **Functional API**: necessaria para multiplos inputs
- Branch CNN tem 4 blocos convolucionais (identicos a CNN standalone)
- Branch MLP com 2 camadas densas

---

## Metricas de Avaliacao

Todas as metricas sao calculadas no conjunto de teste (15% dos dados).

| Metrica | Descricao |
|---|---|
| **Accuracy** | Fracao de classificacoes corretas |
| **F1 Macro** | Media nao ponderada do F1 por classe (trata classes igualmente) |
| **F1 Weighted** | Media ponderada pelo suporte de cada classe |
| **Completeness (Recall)** | TP / (TP + FN) por classe — "que fracao desta classe foi encontrada?" |
| **Reliability (Precision)** | TP / (TP + FP) por classe — "das classificadas como X, quantas realmente sao?" |

Completeness e Reliability sao termos padrao em astronomia observacional,
equivalentes a Recall e Precision respectivamente.

---

## Dataset

### Composicao

| Classe | N | Fracao |
|---|---|---|
| Elliptical | ~135 | ~86% |
| Spiral | ~17 | ~11% |
| Irregular | ~4 | ~3% |
| **Total** | **~156** | **100%** |

### Desbalanceamento

O dataset apresenta desbalanceamento severo. Estrategias adotadas:
- **Class weights** (`balanced`): pesos inversamente proporcionais a frequencia
- **Data augmentation**: expande artificialmente o conjunto de treino (CNN/Hybrid)
- **Split estratificado**: mantem proporcoes em todos os conjuntos

SMOTE nao e viavel como estrategia primaria — Irregular possui apenas 4 amostras.

### Origem dos dados

- **SDSS DR17**: fotometria e espectroscopia via SQL query ao SkyServer —
  https://skyserver.sdss.org/dr17/
- **Galaxy Zoo 2**: rotulos de consenso (Hart et al. 2016) — classificacao
  visual por cidadaos cientistas com thresholds rigorosos (80% de concordancia,
  minimo 20 votos) — https://zenodo.org/record/3565489

---

## Figuras Geradas

O pipeline produz 25 figuras ao longo dos 6 notebooks:

| Fig | Notebook | Descricao |
|---|---|---|
| 01–06 | 01_eda | Distribuicoes, diagramas cor-magnitude, correlacoes |
| 07–15 | 02_preprocessing | Features derivadas, comparacao de stretch, grid de galaxias |
| 16 | 03_mlp | Curvas de treinamento MLP |
| 17 | 03/04/05 | Matrizes de confusao (uma por modelo) |
| 18 | 04_cnn | Exemplos de data augmentation |
| 19 | 04_cnn | Curvas de treinamento CNN |
| 20 | 05_hybrid | Curvas de treinamento Hibrido |
| 21 | 05_hybrid | Grad-CAM — regioes de atencao por classe |
| 22 | 06_comparison | Tabela comparativa de metricas (CSV) |
| 23 | 06_comparison | Completeness/Reliability por classe e modelo |
| 24 | 06_comparison | Confusion matrices side-by-side (MLP vs CNN vs Hybrid) |
| 25 | 06_comparison | Accuracy e F1 por modelo |

Todas as figuras sao salvas automaticamente em `reports/figures/`.

---

## Execucao Rapida (TL;DR)

```bash
# Setup
python -m venv galaxy_env && source galaxy_env/bin/activate
pip install -r requirements.txt

# Colocar gz2_hart16.csv em data/raw/ (download manual do Zenodo)

# Pipeline de dados
python download_sdss.py
python load_gz2_and_merge.py
python download_images.py

# Notebooks (executar em ordem no Jupyter)
jupyter notebook
# Abrir e executar: 01 -> 02 -> 03 -> 04 -> 05 -> 06
```
