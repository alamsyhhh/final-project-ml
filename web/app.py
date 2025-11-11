from flask import Flask, request, render_template
import pandas as pd
import pickle
from urllib.parse import urlparse
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import hashlib
import re

app = Flask(__name__)

def hash_sha256_to_int(url):
    sha256_hash = hashlib.sha256(url.encode()).hexdigest()
    hash_int = int(sha256_hash, 16)
    return hash_int % (10**10)

def calculate_entropy(s):
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    return entropy(probabilities)

def extract_features(url):
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 'is.gd', 't.co']
    special_chars = "!#$%^&*()[]{};:,/<>?\\|`~-=+"

    parsed_url = urlparse(url)

    features = {
        'url': hash_sha256_to_int(url),
        'URL_Length': len(url),
        'Shortening_Service': 1 if any(service in url for service in shortening_services) else 0,
        'Having_At_Symbol': url.count('@'),
        'Double_slash_redirecting': url.count('//'),
        'Prefix_Suffix': url.count('-'),
        'Subdomain_Count': parsed_url.netloc.count('.') - 1,
        'HTTPS_token': 1 if parsed_url.scheme == 'https' else 0,
        'Number_of_Parameters': len(parsed_url.query.split('&')),
        'Number_of_Dots': parsed_url.netloc.count('.'),
        'Length_of_Domain': len(parsed_url.netloc),
        'Number_of_Digits': sum(c.isdigit() for c in url),
        'Number_of_Underscores': url.count('_'),
        'Number_of_Special_Characters': sum(c in special_chars for c in url),
        'Number_of_Letters': sum(c.isalpha() for c in url),
        'Query_Length': len(parsed_url.query),
        'Hostname_Length': len(parsed_url.hostname) if parsed_url.hostname else 0,
        'Length_of_Top_Level_Domain': len(parsed_url.netloc.split('.')[-1]),
        'Path_Depth': len(parsed_url.path.strip('/').split('/')),
        'Entropy_of_URL': calculate_entropy(url),
        'URL_Contains_IP_Address': 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0,
        'URL_Contains_Encoded_Characters': 1 if '%' in url else 0,
    }

    return features

# Load the model and scaler
with open('../pkl/model.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)
    svm_model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    model_accuracy = model_data.get('accuracy', None)

    if model_accuracy is not None:
        model_accuracy = round(model_accuracy * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    url = None
    confidence = None
    features = {}
    if request.method == 'POST':
        url = request.form['url']
        features = extract_features(url)
        
        print("Extracted Features:", features)

        new_data = pd.DataFrame([features])
        new_data = new_data.reindex(columns=feature_columns, fill_value=0)

        print("Data Before Scaling:", new_data)

        new_data_scaled = scaler.transform(new_data)

        print("Data After Scaling:", new_data_scaled)
        
        prediction = svm_model.predict(new_data_scaled)

        print("Prediction:", prediction)

        try:
            confidence_score = svm_model.predict_proba(new_data_scaled)
            confidence = round(confidence_score.max() * 100)
        except AttributeError:
            confidence = "Model tidak mendukung penghitungan keyakinan."
        print("Confidence Score:", confidence)

        result = "Website ini terindikasi Phishing" if prediction[0] == "phishing" else "Website ini tampaknya aman (Non Phishing)"
    
    return render_template('index.html', url=url, result=result, accuracy=model_accuracy, confidence=confidence, features=features)

if __name__ == '__main__':
    app.run(debug=True)
