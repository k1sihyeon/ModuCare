import requests
import json
from datetime import datetime
import cv2

def printResponse(response):
    print("Status Code:", response.status_code)
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except ValueError:
        print("Response Content:", response.text)

def sendImage(file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post("http://118.219.42.214:8080/api/image/upload", files=files)
        printResponse(response)

#frame은 이미지

# image post
path = "/" + datetime.now().strftime('%Y_%m_%dT%H:%M:%S') + ".jpg"
cv2.imwrite("detected" + path, frame)
sendImage("detected" + path)