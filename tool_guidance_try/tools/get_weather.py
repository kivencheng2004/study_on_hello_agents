import requests

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data['current_condition'][0]
        temp = current['temp_C']
        desc = current['weatherDesc'][0]['value']

        return f"{city}的天气：{desc}, 气温{temp}°C"
    except Exception as e:
        return f"查询出错：{e}"


