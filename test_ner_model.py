import argparse
import mlflow.spacy

parser = argparse.ArgumentParser(description='NER Model training')
parser.add_argument('-r', dest='run_id', type=str,
                    help='MLFlow run id')
args = parser.parse_args()

model_uri = "runs:/{run_id}/{artifact_path}".format(
    run_id=args.run_id, artifact_path="model"
)
prdnlp = mlflow.spacy.load_model(model_uri=model_uri)

test_text = input("Enter your testing text: ")

doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
