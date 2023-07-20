import requests

ride = { "Title": "This is a test to check how to use pickle, ipadn jason, microsoft azur and amazon AWS" }

url = 'http://localhost:5000/api/predict'
response = requests.post(url, json=ride)

# Check if the request was successful (status code 200 indicates success)
if response.status_code == 200:
    # Get the response content as a Python dictionary
    data = response.json()
    # Access the 'prediction' key in the response dictionary to get the suggested keywords
    suggested_keywords = data['prediction']
    print("Suggested mots cles:", suggested_keywords)
else:
    # If the request was not successful, print the error message
    print("Request failed with status code:", response.status_code)
