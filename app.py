from flask import Flask, render_template, request
import pickle
import nltk
import re

app = Flask(__name__)

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

    return output

#Create the routes that will handle the interactions with the dashboard. 
# create an HTML template file (index.html) to display the dashboard.
@app.route('/')
def index():
    return render_template('index.html')

# Define another route to handle the prediction request when the user submits data through the form. 
# This route will call the prediction model function with the user input and return the result to be displayed on the dashboard.

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']

    features = prepare_features(input_data)

# création bow pour X_test_title avec les dictionary train  (avec test_body et option)    
    test_bow_corpus = [dictionary.doc2bow(features)]

#   test_bow_corpus = [dictionary.doc2bow(doc) for doc in features]
    test_topics_distributions = lda_model[test_bow_corpus] 

# Effectuer la prédiction

# Extract keywords from the inferred topic distributions
    test_keywords = []
    for doc_topics in test_topics_distributions:       
    # trier les topics par la probailité en order décroissant (x[1] de doc_topics)
        sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    # extraire le premier mot clé des top topics 
        top_keywords = [dictionary[word_id] for word_id, _ in sorted_topics[:5]]  # Adjust the number of keywords as needed
        test_keywords.append(top_keywords)

    return render_template('index.html', prediction_result=test_keywords)

if __name__ == '__main__':
    app.run(debug=True)


