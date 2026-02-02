import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def save_to_word(content: str, filename: str = "output.docx"):
    """将内容写入 Word 文档。
    
    Args:
        content: 要写入文档的文本内容。
        filename: 文件名，默认为 output.docx。
    """
    try:
        from docx import Document
    except ImportError:
        return "Error: 未安装 python-docx 库。请运行 `pip install python-docx`。"
    
    try:
        doc = Document()
        doc.add_paragraph(content)
        doc.save(filename)
        return f"成功将内容保存到 {filename}"
    except Exception as e:
        return f"保存文件时出错: {str(e)}"


def build_graph():
    # 1. 初始化工具
    search_tool = TavilySearchResults(max_results=2)
    # 将新定义的 save_to_word 工具加入列表
    tools = [search_tool, save_to_word]

    # 2. 初始化模型 (OpenRouter 中转)
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet").strip(),
        model_provider="openai",
        openai_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "").strip(),
    )
    llm_with_tools = llm.bind_tools(tools)

    # 3. 定义节点逻辑
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 4. 组装图
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    # --- 核心改进：添加记忆和中断 ---
    memory = MemorySaver()
    # 告诉图：在进入 "tools" 节点之前，强制中断并等待
    return graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])


def run_interactive_chat(graph):
    config = {"configurable": {"thread_id": "human_in_the_loop_demo"}}
    print("--- 助手已就绪（支持多轮审批） ---")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # 初始输入
        current_input = {"messages": [("user", user_input)]}

        while True:
            # 使用 updates 模式，只获取每个节点新产生的内容
            events = graph.stream(current_input, config, stream_mode="updates")

            for event in events:
                # event 的格式类似 {"chatbot": {"messages": [...]}}
                for node_name, value in event.items():
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        # 只有 AI 的文字回复才打印，过滤掉工具调用指令
                        if hasattr(last_msg, 'content') and last_msg.content and last_msg.type == 'ai':
                            print(f"Assistant: {last_msg.content}")

            # 关键：检查是否还有后续节点（是否被中断了）
            snapshot = graph.get_state(config)
            if not snapshot.next:
                # 没有下一个节点了，说明 AI 已经彻底回答完了，跳出内层循环等待新提问
                break

            # 如果停在了 tools 节点前
            if snapshot.next[0] == "tools":
                print(f"\n>>> [系统提示] AI 申请使用工具。")
                # 你可以从状态里把 AI 到底想搜什么给打印出来
                last_ai_msg = snapshot.values["messages"][-1]
                
                # 遍历所有工具调用请求
                if hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
                    for tool_call in last_ai_msg.tool_calls:
                        print(f">>> 工具: {tool_call['name']}")
                        print(f">>> 参数: {tool_call['args']}")

                choice = input(">>> 是否批准？(y/n): ")
                if choice.lower() == 'y':
                    current_input = None  # 继续运行不需要新输入
                    print(">>> 正在执行...")
                else:
                    print(">>> 用户拒绝，操作终止。")
                    # 这里也可以注入一条用户拒绝的消息给 AI
                    break


if __name__ == "__main__":
    my_graph = build_graph()
    run_interactive_chat(my_graph)
