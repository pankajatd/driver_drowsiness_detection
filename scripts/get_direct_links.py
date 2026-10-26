import urllib.request
import json
import urllib.parse

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

titles = [
    "File:Romanian Truck Driver (62829107).jpeg",
    "File:Man driving a car (50069396348).jpg",
    "File:Driver in car (15830911854).jpg",
    "File:A man driving a car.jpg",
    "File:Young woman driving a car.jpg",
    "File:Man driving car (Unsplash).jpg",
    "File:Driver behind wheel.jpg"
]

for t in titles:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(t)}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            for p in data['query']['pages'].values():
                if 'imageinfo' in p:
                    print(t, "-->", p['imageinfo'][0]['url'])
    except Exception as e:
        pass
