import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# 1. 加载配置（必须在导入工具前执行，确保工具能读到环境变量）
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 2. 导入你现有的工具函数
from tools.get_weather import get_weather as raw_get_weather
from tools.search_attraction import get_attraction as raw_get_attraction


# 3. 包装工具：LangGraph 通过这些装饰器识别工具用途
@tool
def get_weather(city: str):
    """查询指定城市的实时天气。"""
    return raw_get_weather(city)


@tool
def get_attraction(city: str, weather: str):
    """根据城市和天气搜索推荐的旅游景点。此工具内部已集成 Tavily 搜索。"""
    # 这里的逻辑直接复用你 search_attraction.py 里的代码
    return raw_get_attraction(city, weather)


# 定义工具集合
tools = [get_weather, get_attraction]
tool_node = ToolNode(tools)


# 4. 定义状态 (State)
class AgentState(TypedDict):
    # add_messages 会自动把新对话追加到 messages 列表中
    messages: Annotated[list[BaseMessage], add_messages]


# 5. 初始化模型并绑定工具
llm = ChatOpenAI(
    model=os.getenv("MODEL_ID", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
).bind_tools(tools)


# 6. 定义节点逻辑
def call_model(state: AgentState):
    """Agent 节点：负责思考和决策"""
    response = llm.invoke(state['messages'])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """条件路由：判断模型是想调工具还是结束对话"""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "action"
    return END


# 7. 构建图 (Graph)
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

# 设置带条件的边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "action": "action",
        END: END
    }
)

# 工具执行完后，必须回到 agent 节点让它总结结果
workflow.add_edge("action", "agent")

app = workflow.compile()


# 8. 执行入口
def main():

    user_prompt = input("请输入您的旅行需求：")

    # 初始状态
    inputs = {"messages": [HumanMessage(content=user_prompt)]}

    # 流式观察 Agent 的思考过程
    for chunk in app.stream(inputs, stream_mode="updates"):
        for node_name, values in chunk.items():
            print(f"\n[进入节点]: {node_name}")
            last_msg = values["messages"][-1]

            if last_msg.content:
                print(f"【AI 回复】: {last_msg.content}")

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for t in last_msg.tool_calls:
                    print(f"【准备调用】: {t['name']}({t['args']})")


if __name__ == "__main__":
    main()