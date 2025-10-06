import requests

url = "https://detect.roboflow.com/fish-bdxmd/1"
params = {"api_key": "YregzS3X1k22xaPX0IMS"}
files = {"file": open("Sandbar_Shark.png", "rb")}
response = requests.post(url, params=params, files=files)
print(response.json())
