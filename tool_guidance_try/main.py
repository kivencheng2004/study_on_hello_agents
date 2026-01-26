import os
import re
from tools.get_weather import get_weather
from tools.search_attraction import get_attraction
from openai import OpenAI

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt:str) ->str:
        print("正在调用大模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
        ]
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                stream = False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误：{e}")
            return "错误：调用语言模型服务时出错。"

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

def main():
    API_KEY = "API_KEY"
    BASE_URL = "base_url"
    MODEL_ID = "model_name"
    os.environ["TAVILY_API_KEY"] = "tavily_api_key"

    llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

    user_prompt = input("请输入您的旅行需求：")
    prompt_history = [f"用户请求：{user_prompt}"]

    print("\n" + "="*50)
    print(f"用户输入：{user_prompt}")
    print("="*50)
    for i in range(5):
        print(f"\n>>> 正在进行第 {i + 1} 轮思考...")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output,
                          re.DOTALL)
        if match:
            llm_output = match.group(1).strip()

        print(f"【AI 思考结果】:\n{llm_output}")
        prompt_history.append(llm_output)

        action_match = re.search(r"Action: (.*)", llm_output)
        if not action_match:
            obs_str = "Observation: 错误：未检测到 Action 字段，请检查回复格式。"
            print(obs_str)
            prompt_history.append(obs_str)
            continue

        action_str = action_match.group(1).strip()

        # 如果是任务结束信号
        if action_str.startswith("Finish"):
            # 提取 Finish[...] 括号里的最终答案
            finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
            final_answer = finish_match.group(1) if finish_match else action_str
            print("\n" + "★" * 20)
            print(f"任务最终完成！\n结果: {final_answer}")
            print("★" * 20)
            break

        try:
            tool_name = re.search(r"(\w+)\(", action_str).group(1)
            args_str = re.search(r"\((.*)\)", action_str).group(1)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

            if tool_name in available_tools:
                print(f"[执行] 正在调用工具 {tool_name}，参数为 {kwargs}...")
                observation = available_tools[tool_name](**kwargs)
            else:
                observation = f"错误: 未找到名为 {tool_name} 的工具。"

        except Exception as e:
            observation = f"解析指令或执行工具时出错: {e}"

        observation_str = f"Observation: {observation}"
        print(f"【工具反馈结果】:\n{observation_str}")
        prompt_history.append(observation_str)
        print("-" * 50)

if __name__ == "__main__":
    main()
