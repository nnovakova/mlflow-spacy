# NER Spacy with MLFlow

Example project to use MLFlow for tracking machine learning experiments when training NER model with Spacy library. It trains a model for German language.

## Train

Create new experiment via MLFlow CLI:

```bash
mlflow experiments create -n ner-medicine-spacy
```

Use experiment id from the above command output:

```bash
mlflow run -e main . -P drop_rate=0.25 -P iterations=30 --experiment-id 1
```

## Test

```bash
mlflow run -e test . \
    -P run_id=<put run id from the train output> \
    --experiment-id 1
```