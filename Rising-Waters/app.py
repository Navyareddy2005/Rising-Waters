from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/floods.save")
scaler = joblib.load("model/transform.save")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Predict")
def predict_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["MonsoonIntensity"]),
        float(request.form["TopographyDrainage"]),
        float(request.form["RiverManagement"]),
        float(request.form["Deforestation"]),
        float(request.form["Urbanization"]),
        float(request.form["ClimateChange"]),
        float(request.form["DamsQuality"]),
        float(request.form["Siltation"]),
        float(request.form["AgriculturalPractices"]),
        float(request.form["Encroachments"]),
        float(request.form["IneffectiveDisasterPreparedness"]),
        float(request.form["DrainageSystems"]),
        float(request.form["CoastalVulnerability"]),
        float(request.form["Landslides"]),
        float(request.form["Watersheds"]),
        float(request.form["DeterioratingInfrastructure"]),
        float(request.form["PopulationScore"]),
        float(request.form["WetlandLoss"]),
        float(request.form["InadequatePlanning"]),
        float(request.form["PoliticalFactors"])
    ]

    data = pd.DataFrame([features], columns=[
        "MonsoonIntensity",
        "TopographyDrainage",
        "RiverManagement",
        "Deforestation",
        "Urbanization",
        "ClimateChange",
        "DamsQuality",
        "Siltation",
        "AgriculturalPractices",
        "Encroachments",
        "IneffectiveDisasterPreparedness",
        "DrainageSystems",
        "CoastalVulnerability",
        "Landslides",
        "Watersheds",
        "DeterioratingInfrastructure",
        "PopulationScore",
        "WetlandLoss",
        "InadequatePlanning",
        "PoliticalFactors"
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        return render_template("chance.html")
    else:
        return render_template("no_chance.html")


if __name__ == "__main__":
    app.run(debug=True)