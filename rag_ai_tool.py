from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool


embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma(
    embedding_function=embedding_model,
    collection_name="embeddings_docu",
    persist_directory="./chromaDB"
)

@tool
def rag_ai(query:str)->str:
    """Use this tool when the user asks a question related to the provided PDF.

    Search only the provided PDF and return relevant information from it.
    Do not use external knowledge or information outside the provided PDF.

    If the answer is not available in the provided PDF,
    clearly indicate that the information was not found.

    You are a helpful AI assistant.

    IMPORTANT LANGUAGE RULE:
    - Always respond in English only.
    - Do not respond in Hindi or any other language unless the user explicitly asks you to.
    - Even if the user writes in Hindi, respond in English.

    Use tools when necessary.
"""
    

    result=vector_store.similarity_search(
        query,k=3
    )

    if not result:
        return "i can not find relevent information as you provided pdf"
    
    context=""
    for i in result:
        context += i.page_content + "\n\n"
        print("AI ---->>>>",context)
    

    system_prompt = """
    You are a PDF Question Answering Assistant.

    Your task is to answer the user's question ONLY using the information
    provided in the retrieved context from the PDF.

    STRICT RULES:

    1. Use ONLY the provided PDF context to answer the question.
    2. Do NOT use your own knowledge or any external information.
    3. Do NOT make assumptions or generate information that is not present
    in the provided context.
    4. If the provided context does not contain relevant information needed
    to answer the question, respond exactly:

    "I cannot find relevant information in the provided PDF."

    5. If relevant information is available in the context, provide the answer
    in a clear, structured, and easy-to-understand format.

    ANSWER FORMAT:

    ## Answer

    ### Explanation
    - Clearly explain the answer using only the provided PDF context.

    ### Key Points
    - Important point 1
    - Important point 2
    - Important point 3

    ### Additional Details
    - Add any other relevant information available in the PDF context.

    IMPORTANT:
    - Do not include any information that is not present in the provided context.
    - Keep the answer focused on the user's question.
    - Use headings and bullet points to make the answer easy to understand.
    - If only partial information is available, answer only with the
    information supported by the context.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ])

    model=ChatMistralAI(model="mistral-small-2506",temperature=0.1)

    parser=StrOutputParser()

    chain=prompt|model| parser

    return chain.invoke({
        "context":context,
        "question":query
    })


