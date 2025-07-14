# 1. MotoJeopardyNER
A specialized dataset curation pipeline for Named Entity Recognition (NER) validation using Jeopardy! questions. This project processes ~217K Jeopardy questions to create high-quality validation subsets targeting specific NER challenges.

# 2. Project Overview
Goal: Curate specialized subsets from the Jeopardy Questions dataset to validate NER model performance across three challenging categories:

- Phrases containing numbers
- Phrases containing non-English words
- Phrases containing unusual proper nouns

# 3. Project structure sketch:
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

# 4. Infrastructure
## 4.1 Logging
## 4.2 Reports
## 4.3 Validation

# 5. Further consideration
- [] Module reusable
- [] Caching and database

---

# 6. Get Start
## 6.1 Setup
## 6.2 Module validation
- dataloader
```bash
python3 -m src.data.handler
```

- datafactory
```bash
python3 -m src.factories.numbers_factory
python3 -m src.factories.language_factory
python3 -m src.factories.entities_factory
```