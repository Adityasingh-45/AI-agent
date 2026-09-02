from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain.tools import tool

@tool
def ai_tutor(query : str)-> str:
    """Use this tool for educational questions about Python,
    SQL, Machine Learning, Deep Learning, Generative AI,
    LangChain, RAG, AI Agents, and Agentic AI """

    system_prompt="""You are an expert AI Tutor.

    Explain concepts in a clear and beginner-friendly way.
    Use simple English and Hinglish when helpful.

    Structure your answers with:
    - Definition
    - Key Points
    - Example
    - Important Notes
    You are a helpful AI assistant.

    IMPORTANT LANGUAGE RULE:
    - Always respond in English only.
    - Do not respond in Hindi or any other language unless the user explicitly asks you to.
    - Even if the user writes in Hindi, respond in English.

    Use tools when necessary.
"""

    
    History=[]

    prompt=ChatPromptTemplate([
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human","{query}")
    ])

    model=ChatMistralAI(model="mistral-small-2506",temperature=0.9)

    parser=StrOutputParser()

    chain=prompt| model| parser

    return chain.invoke({"query":query,"history":History})