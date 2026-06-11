from flask import Flask ,request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("covid_model.pkl","rb"))

@app.route("/")

def home():
    return "covid prediction  API is running "

@app.route("/predict", methods=["POST"])
def predict():
    pass

def predict():

    data = request.json

    features = np.array([[
        data["age"],
        data["gender"],
        data["fever"],
        data["cough"],
        data["city"],
    ]])

    prediction = model.predict(features)[0]

    return jsonify({"has_covid" : int(prediction)})



if __name__ == "__main__":

    app.run(debug=True)
