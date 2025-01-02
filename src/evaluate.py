import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
import yaml
import os
from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Madhuvamsi-iitg/Machinelearning-pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="Madhuvamsi-iitg"
os.environ['MLFLOW_TRACKING_PASSWORD']="c2ac17ec637c41bbbffdb043389efdcbc59a4882"

#load the parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/Madhuvamsi-iitg/Machinelearning-pipeline.mlflow")

    #load the model from the disk
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    #log metrics to MLFLOW
    mlflow.log_metric("accuracy", accuracy)
    print("Model accuracy: {accuracy}")
if __name__ == '__main__':
    evaluate(params['data'],params['model'])