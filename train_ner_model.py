import argparse
import json
import spacy
import random
import warnings
import mlflow.spacy
from mlflow import log_metric, log_param, log_artifacts

# Read json file and transform it to array
def transform_json(json_file_path):
    corpus = open(json_file_path, 'r')
    lines = corpus.readlines()

    training_data = []  # array of train data with marked up medical entities
    # read lines from file and parsing them into special array
    for line in lines:
        res = json.loads(line)
        text = res['content']
        entities = []
        for annotation in res['annotation']:
            point = annotation['points'][0]
            labels = annotation['label']
            if not isinstance(labels, list):
                labels = [labels]
            for label in labels:
                entities.append((point['start'], point['end'] + 1, label))
        training_data.append((text, {"entities": entities}))
    return training_data


def train_spacy(data, iterations, drop_rate):
    nlp = spacy.blank('de')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    ner_pipe = 'ner'
    if ner_pipe not in nlp.pipe_names:
        ner = nlp.create_pipe(ner_pipe)
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe(ner_pipe)

    # add labels
    for _, annotations in data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    log_param("drop_rate", drop_rate)
    log_param("iterations", iterations)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != ner_pipe]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(data)
            losses = {}
            for text, annotations in data:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=drop_rate,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            log_metric("ner.losses", value=losses[ner_pipe], step=itn)
            print(losses)
    return nlp


parser = argparse.ArgumentParser(description='NER Model training')
parser.add_argument('-d', dest='drop_rate', type=float,
                    help='SpaCy drop rate for training [0.0..0.9]')
parser.add_argument('-i', dest='iterations', type=int,
                    help='Number of iterations')
args = parser.parse_args()


training_data = transform_json('med-corpus.json')
prdnlp = train_spacy(training_data, args.iterations, args.drop_rate)
mlflow.spacy.log_model(spacy_model=prdnlp, artifact_path="model")

print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
