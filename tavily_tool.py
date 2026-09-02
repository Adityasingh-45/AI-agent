from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.tools import tool

load_dotenv()


@tool
def tavily_ai(query: str) -> str:
    """
    Search the internet for the latest and current information.

    Use this tool for:
    - Latest news
    - Current events
    - Recent information
    You are a helpful AI assistant.

    IMPORTANT LANGUAGE RULE:
    - Always respond in English only.
    - Do not respond in Hindi or any other language unless the user explicitly asks you to.
    - Even if the user writes in Hindi, respond in English.

    Use tools when necessary.
"""

    

    search_tool = TavilySearch(
        max_results=5,
        search_depth="advanced",
        time_range="day"
    )

    result = search_tool.invoke({
        "query": query
    })

    return str(result)