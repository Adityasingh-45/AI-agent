from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)

from ai_tutor_tool import ai_tutor
from rag_ai_tool import rag_ai
from tavily_tool import tavily_ai

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.2
)

tools = [
    ai_tutor,
    rag_ai,
    tavily_ai
]

llm_with_tools = llm.bind_tools(tools)

tools_dict = {}

for tool in tools:
    tools_dict[tool.name] = tool

messages = []

print("\n===================================")
print("     MULTI TOOL AI AGENT 🤖")
print("===================================\n")

print("Available Tools:")
print("1. AI Tutor")
print("2. PDF RAG")
print("3. Tavily Web Search\n")

print("Type 'exit' to stop.\n")

while True:
    user = input("Enter your Query ---- >>>> ")

    if user.lower() == "exit":

        print("\nGood bye ")

        break

    messages.append(
        HumanMessage(
            content=user
        )
    )

    ai_message = llm_with_tools.invoke(
        messages
    )

    messages.append(
        ai_message
    )

    while ai_message.tool_calls:

        for tool_call in ai_message.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call["args"]

            tool_id = tool_call["id"]


            print(
                f"\n🔧 Using Tool: {tool_name}"
            )

            selected_tool = tools_dict[
                tool_name
            ]

            tool_result = selected_tool.invoke(
                tool_args
            )

            messages.append(

                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id
                )

            )

        ai_message = llm_with_tools.invoke(
            messages
        )

        messages.append(
            ai_message
        )

    print("\nAI -->")

    print(
        ai_message.content
    )

    print()