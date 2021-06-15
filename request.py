import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'R&D Spend':165300, 'Administration':137896, 'Marketing Spend':47100, 'Florida':0, 'New york':1})

print(r.json())