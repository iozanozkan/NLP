from flask import Flask, request
from flask_cors import CORS
import joblib
from flask import jsonify

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods = ["POST"])
def tahmin_et():
    model = open("model.pkl","rb")
    clf = joblib.load(model)
    if request.method == "POST":
        text = request.json["data"]
        data = [text]
        sonuc = clf.predict(data).astype(int)
        if sonuc==0:
            prediction = "Yanlış"
        else:
            prediction = "Doğru"    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run()