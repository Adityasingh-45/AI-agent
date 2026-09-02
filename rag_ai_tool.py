from dotenv import load_dotenv

load_dotenv()


# ==========================================
# IMPORTS
# ==========================================

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool


# ==========================================
# BASE DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DB_PATH = BASE_DIR / "chromaDB"


# ==========================================
# LAZY LOADING VARIABLES
# ==========================================

embedding_model = None

vector_store = None

rag_chain = None


# ==========================================
# GET VECTOR STORE
# ==========================================

def get_vector_store():

    global embedding_model
    global vector_store


    # --------------------------------------
    # IF ALREADY LOADED
    # --------------------------------------

    if vector_store is not None:

        return vector_store


    # --------------------------------------
    # CHECK DATABASE EXISTS
    # --------------------------------------

    if not CHROMA_DB_PATH.exists():

        raise FileNotFoundError(
            f"Chroma database not found at: {CHROMA_DB_PATH}"
        )


    # --------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------

    print("Loading embedding model...")


    embedding_model = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )


    # --------------------------------------
    # LOAD CHROMA DATABASE
    # --------------------------------------

    print("Loading Chroma database...")


    vector_store = Chroma(

        embedding_function=embedding_model,

        collection_name="embeddings_docu",

        persist_directory=str(CHROMA_DB_PATH)

    )


    print("Chroma database loaded successfully.")


    return vector_store


# ==========================================
# GET RAG CHAIN
# ==========================================

def get_rag_chain():

    global rag_chain


    # --------------------------------------
    # IF ALREADY CREATED
    # --------------------------------------

    if rag_chain is not None:

        return rag_chain


    # ======================================
    # SYSTEM PROMPT
    # ======================================

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
- Important points from the PDF.

### Additional Details
- Add any other relevant information available in the PDF context.

IMPORTANT:

- Do not include information that is not present in the provided context.
- Keep the answer focused on the user's question.
- Use headings and bullet points to make the answer easy to understand.
- If only partial information is available, answer only with the
  information supported by the context.

"""


    # ======================================
    # CREATE PROMPT
    # ======================================

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            system_prompt
        ),

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


    # ======================================
    # CREATE MODEL
    # ======================================

    model = ChatMistralAI(

        model="mistral-small-2506",

        temperature=0.1

    )


    # ======================================
    # OUTPUT PARSER
    # ======================================

    parser = StrOutputParser()


    # ======================================
    # CREATE CHAIN
    # ======================================

    rag_chain = (

        prompt

        | model

        | parser

    )


    return rag_chain


# ==========================================
# RAG TOOL
# ==========================================

@tool
def rag_ai(query: str) -> str:

    """
    Use this tool when the user asks a question related to
    the provided PDF.

    Search only the provided PDF and return relevant information.

    Do not use external knowledge or information outside the PDF.

    If the answer is not available in the PDF,
    clearly indicate that the information was not found.
    """


    try:


        # ======================================
        # GET VECTOR DATABASE
        # ======================================

        vector_db = get_vector_store()


        # ======================================
        # SIMILARITY SEARCH
        # ======================================

        results = vector_db.similarity_search(

            query,

            k=3

        )


        # ======================================
        # CHECK RESULT
        # ======================================

        if not results:

            return (
                "I cannot find relevant information "
                "in the provided PDF."
            )


        # ======================================
        # CREATE CONTEXT
        # ======================================

        context = "\n\n".join(

            document.page_content

            for document in results

        )


        print(

            f"Retrieved {len(results)} documents."

        )


        # ======================================
        # GET RAG CHAIN
        # ======================================

        chain = get_rag_chain()


        # ======================================
        # RETURN ANSWER
        # ======================================

        return chain.invoke({

            "context": context,

            "question": query

        })


    except Exception as e:


        print(

            "RAG ERROR:",

            str(e)

        )


        return (

            "⚠️ Error while searching the PDF: "

            + str(e)

        )