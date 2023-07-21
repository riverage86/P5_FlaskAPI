from flask import Flask, request, jsonify
import joblib
import pickle
import nltk
import re

app = Flask(__name__)

# Charger le modèle scikit-learn
# model = joblib.load('modele_bowLG.pkl')
with open('modele_bowLG.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('mlb.pkl', 'rb') as mlb_file:
    mlb = pickle.load(mlb_file)

nltk.download('punkt')  # Only required once to download the tokenizer data

def prepare_features(ride):

    # Use re.sub to remove non-alphabetic characters and replace with spaces
    processed_string = re.sub('[^a-zA-Z_]', ' ', ride)

    # Tokenize the string using nltk.word_tokenize
    tokenized_string = nltk.word_tokenize(processed_string)

    print("features =", tokenized_string)

    features = vectorizer.transform(tokenized_string)
    return features

@app.route('/')
def hello():
    return jsonify(message='Hello, world!')

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()  # Récupérer les données du corps de la requête

    features = prepare_features(data['Title'])

    print("features bin =", features)
    # Effectuer la prédiction
    prediction_bin = model.predict(features)

    prediction = mlb.inverse_transform(prediction_bin) 

    # Renvoyer les prédictions sous forme de réponse JSON
    return jsonify({'prediction': prediction}), 200

#    return jsonify({'prediction': prediction.tolist()}), 200

@app.route('/api/test')
def test():
    return jsonify(message='Hello, world! test')


if __name__ == '__main__':
    app.run(debug=True)
