import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from langchain_mistralai import ChatMistralAI

from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)


# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = Path(__file__).resolve().parent


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Multi Tool AI Agent"
)


# =========================
# STATIC + TEMPLATES
# =========================

app.mount(
    "/static",
    StaticFiles(
        directory=str(BASE_DIR / "static")
    ),
    name="static"
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# =========================
# CREATE LLM
# =========================

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.2
)


# =========================
# LOAD TOOLS
# =========================

from ai_tutor_tool import ai_tutor
from rag_ai_tool import rag_ai
from tavily_tool import tavily_ai


# =========================
# ADD TOOLS
# =========================

tools = [
    ai_tutor,
    rag_ai,
    tavily_ai
]


# =========================
# BIND TOOLS
# =========================

llm_with_tools = llm.bind_tools(
    tools
)


# =========================
# TOOL DICTIONARY
# =========================

tools_dict = {
    tool.name: tool
    for tool in tools
}


# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):

    message: str


# =========================
# MESSAGE HISTORY
# =========================

messages = []


# =========================
# HOME PAGE
# =========================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
async def health():

    return {
        "status": "running"
    }


# =========================
# CHAT API
# =========================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        # -------------------------
        # ADD USER MESSAGE
        # -------------------------

        messages.append(
            HumanMessage(
                content=request.message
            )
        )


        # -------------------------
        # CALL LLM
        # -------------------------

        ai_message = llm_with_tools.invoke(
            messages
        )


        # -------------------------
        # SAVE AI RESPONSE
        # -------------------------

        messages.append(
            ai_message
        )


        # -------------------------
        # TOOL CALLING LOOP
        # -------------------------

        while ai_message.tool_calls:


            # EXECUTE ALL TOOL CALLS

            for tool_call in ai_message.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call["args"]

                tool_id = tool_call["id"]


                print(
                    f"🔧 Using Tool: {tool_name}"
                )


                # GET TOOL

                selected_tool = tools_dict.get(
                    tool_name
                )


                # TOOL NOT FOUND

                if selected_tool is None:

                    tool_result = (
                        f"Tool '{tool_name}' not found."
                    )


                else:

                    # EXECUTE TOOL

                    tool_result = selected_tool.invoke(
                        tool_args
                    )


                # ADD TOOL RESULT

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    )
                )


            # -------------------------
            # CALL LLM AGAIN
            # -------------------------

            ai_message = llm_with_tools.invoke(
                messages
            )


            # SAVE AI RESPONSE

            messages.append(
                ai_message
            )


        # =========================
        # RETURN FINAL RESPONSE
        # =========================

        return {
            "response": str(
                ai_message.content
            )
        }


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return {
            "response": (
                "⚠️ Sorry, something went wrong: "
                + str(e)
            )
        }


# =========================
# RUN LOCALLY
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        )
    )