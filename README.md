# GalaxyNet — Classificador Morfológico de Galáxias

Projeto Prático de Deep Learning Aplicado à Astrofísica  
**Prof. André Ribeiro — UESC**

## Descrição
Pipeline completo para classificação morfológica de galáxias usando dados do SDSS DR17 e Galaxy Zoo 2, com modelos MLP, CNN e Híbrido.

## Estrutura do Projeto
```
galaxy_classification/
├── data/
│   ├── raw/          # Dados brutos (gz2_hart16.csv, dados SDSS)
│   ├── processed/    # Dados processados (catálogo mesclado)
│   └── images/       # Imagens FITS baixadas
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_mlp_model.ipynb
│   ├── 04_cnn_model.ipynb
│   └── 05_hybrid_model.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── models/           # Modelos treinados (.h5 / SavedModel)
├── reports/          # Relatório final, figuras e apresentações
├── requirements.txt
└── README.md
```

## Cronograma
| Semana | Tema | Entregável |
|--------|------|------------|
| 1 | EDA e Aquisição de Dados | Notebook EDA, dados baixados |
| 2 | Pré-processamento | Dados limpos e imagens pré-processadas |
| 3 | Modelo MLP | MLP treinado e avaliado |
| 4 | Modelo CNN | CNN treinada com data augmentation |
| 5 | Modelo Híbrido | Modelo híbrido + Grad-CAM |
| 6 | Análise e Relatório | Relatório final e apresentação |

## Instalação
```bash
python -m venv galaxy_env
source galaxy_env/bin/activate
pip install -r requirements.txt
```

## Datasets
- **SDSS DR17**: https://skyserver.sdss.org/dr17/
- **Galaxy Zoo 2**: https://zenodo.org/record/3565489 (gz2_hart16.csv)
