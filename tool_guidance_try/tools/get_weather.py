import requests

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"
    headers = {'User-Agent': 'Mozilla/5.0'} # [新增] 模拟浏览器请求

    try:
        # 将超时时间增加到 15 秒，通常能解决大部分波动问题
        response = requests.get(url, headers=headers, timeout=15) # [修改]
        response.raise_for_status()
        data = response.json()
        current = data['current_condition'][0]
        temp = current['temp_C']
        desc = current['weatherDesc'][0]['value']

        return f"{city}的天气：{desc}, 气温{temp}°C"
    except Exception as e:
        # 这里的错误返回会传回给 LangGraph 的 Agent 节点
        return f"查询天气时遇到网络问题：{e}。请尝试直接根据城市搜索景点。"