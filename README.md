# 1. MotoJeopardyNER
A specialized dataset curation pipeline for Named Entity Recognition (NER) validation using Jeopardy! questions. This project processes ~217K Jeopardy questions to create high-quality validation subsets targeting specific NER challenges.

# 2. Project Overview
Goal: Curate specialized subsets from the Jeopardy Questions dataset to validate NER model performance across three challenging categories:

- Phrases containing numbers
- Phrases containing non-English words
- Phrases containing unusual proper nouns

# 3. Project structure sketch:

MOTOJEOPARDYNER/
├── data/
│   ├── processed/
│   ├── raw/
│   └── subsets/
├── outputs/
│   ├── logs/
│   ├── processed/
│   ├── reports/
│   └── validation/
├── scripts/
│   ├── main_pipeline.py
│   └── test_data_pipeline.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   └── handler.py
│   └── factories/
│       ├── __init__.py
│       ├── base_factory.py
│       ├── entities_factory.py
│       ├── language_factory.py
│       └── numbers_factory.py
├── .gitignore
├── README.md
└── requirements.txt

## 3.1 Data preprocessing
- [x] dataloader
- [x] validation/clean

Target: Preserves meaningful content while cleaning formatting issues
1. HTML processing
2. Field specific cleaning
3. Text normalization

## 3.2 Data generation
### 3.2.1 Datafactory
- [x] phrases containing numbers
- [x] phrases containing non-English words
- [x] phrases containing unusual proper nouns

Implementation analysis and packages used are detailed in each ***factory docstring***.

# 4. Infrastructure
- [x] Logging
- [x] Reports
- [x] Validation

# 5. Further consideration
- [ ] Module reusable
- [ ] Caching and database

---

# 6. Get Start
## 6.1 Setup
1. Install packages
```bash
pip3 install -r requirements.txt
```
2. Run main_pipeline
```bash
python3 scripts/main_pipeline.py
```

1. cleaned data saved to data/processed/jeopardy_cleaned.jsonl
2. sampled subsets saved to data/subsets/*
3. logging and report saved to outputs/*

Console outputs are saved to ['console output'](./console_log)

## 6.2 Module validation
- dataloader
```bash
python3 -m src.data.handler
```

- data cleaner
```bash
python3 scripts/test_data_pipeline.py
```

- datafactory
```bash
python3 -m src.factories.numbers_factory
python3 -m src.factories.language_factory
python3 -m src.factories.entities_factory
```

