from flask import Flask, request, jsonify
import joblib
import pickle
import nltk
import re

app = Flask(__name__)

# Charger le modèle scikit-learn
# model = joblib.load('modele_bowLG.pkl')
with open('lda_model.pkl', 'rb') as model_file:
    lda_model = pickle.load(model_file)

with open('train_dictionary.pkl', 'rb') as dictionary_file:
    dictionary = pickle.load(dictionary_file)

nltk.download('punkt')  # Only required once to download the tokenizer data

def prepare_features(ride):

    # Use re.sub to remove non-alphabetic characters and replace with spaces
    processed_string = re.sub('[^a-zA-Z_]', ' ', ride)

    # Tokenize the string using nltk.word_tokenize
    output = nltk.word_tokenize(processed_string)

#    array_of_unicode_tokens = [token for token in output]

    return output

@app.route('/')
def hello():
    return jsonify(message='Hello, world!')

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()  # Récupérer les données du corps de la requête

    features = prepare_features(data['Title'])

    print("features =", features)
# création bow pour X_test_title avec les dictionary train  (avec test_body et option)    
    test_bow_corpus = [dictionary.doc2bow(features)]

#   test_bow_corpus = [dictionary.doc2bow(doc) for doc in features]
    test_topics_distributions = lda_model[test_bow_corpus] 

    print("test_bow_corpus =", test_bow_corpus)
    print("test_topics_distributions =", test_topics_distributions)
    print("test_topics_distributions type =", type(test_topics_distributions))

# Effectuer la prédiction

# Extract keywords from the inferred topic distributions
    test_keywords = []
    for doc_topics in test_topics_distributions:       
    # trier les topics par la probailité en order décroissant (x[1] de doc_topics)
        sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    # extraire le premier mot clé des top topics 
        top_keywords = [dictionary[word_id] for word_id, _ in sorted_topics[:10]]  # Adjust the number of keywords as needed
        test_keywords.append(top_keywords)

    print("test_keywords =", test_keywords)
    # Renvoyer les prédictions sous forme de réponse JSON
    return jsonify({'prediction': test_keywords})

@app.route('/api/test')
def test():
    return jsonify(message='Hello, world! test')


if __name__ == '__main__':
    app.run(debug=True)
