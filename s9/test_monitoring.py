import time
import requests
url = 'https://europe-west1-proven-audio-337909.cloudfunctions.net/function-iris'
payload = {'input_data': '1, 1, 1, 1'}

for _ in range(1000):
   r = requests.get(url, params=payload)
   # print(r.content)