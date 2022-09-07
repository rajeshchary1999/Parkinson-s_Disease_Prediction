import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle
import pandas as pd
import numpy as np


# load the pickle model
model = pickle.load(open("model_history.pkl", "rb"))
predictions_classes = {0: "Healthy", 1: "Parkinsons Disease"}

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # access the data from form
        # ph
        Fo = float(request.form["Fo"])
        Fhi = float(request.form["Fhi"])
        Flo = float(request.form["Flo"])
        Shimmer = float(request.form["Shimmer"])
        HNR = float(request.form["HNR"])
        RPDE = float(request.form["RPDE"])
        spread1 = float(request.form["spread1"])
        D2 = float(request.form["D2"])

        # get prediction

        columns = [

            [Fo, Fhi, Flo, Shimmer, HNR, RPDE, spread1, D2]]

    results = model.predict(columns)
    results = results.tolist()[0]

    pred = {"Predicted quality": predictions_classes[results]}

    return pred


if __name__ == "__main__":
    app.run(debug=True)