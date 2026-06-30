import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Response
from langchain_core.runnables import RunnableConfig
from src.config import settings
from src.llm_chain import ai_agent_executor, State
from src.whatsapp_service import WhatsAppService
from src.Rag_engine import initialize_rag_system, get_retriever

@asynccontextmanager
async def lifespan(app: FastAPI):
    data_path = os.path.join(settings.DATA_DIR, "Company_policy.txt")

    needs_init = True
    if os.path.exists(settings.DB_DIR):
        try:
            retriever = get_retriever()
            collection = getattr(retriever.vectorstore, "_collection", None)
            count = collection.count() if collection is not None else 0
            if count > 0:
                needs_init = False
                print(f"Found existing Vector Store with {count} chunks. Skipping initialization.")
        except Exception:
            pass  

    if needs_init:
        print("🚀 Beginning initialization of RAG system and Vector Store...")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(data_path):
            initialize_rag_system(data_path)
        else:
            print(f"⚠️ Critical Alert: Please place the data file in: {data_path}")

    yield
    print("Closing the FastAPI application. Cleanup if necessary.")

app = FastAPI(title="Enterprise AI Agent with WhatsApp & LangGraph", lifespan=lifespan)
whatsapp_service = WhatsAppService()

@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    user_number = From
    user_message = Body

    print(f"📩 رسالة واردة عبر LangGraph من {user_number}: {user_message}")

    try:
        config: RunnableConfig = {"configurable": {"thread_id": user_number}}

        state_input = State(
            input=user_message,
            chat_history=[],
            context="",
            answer=""
        )

        response = ai_agent_executor.invoke(state_input, config=config)
        ai_answer = response.get("answer", "عذراً، لم أستطع معالجة الطلب حالياً.")

        whatsapp_service.send_message(to_number=user_number, message_body=ai_answer)
        print(f"📤 the respond has been sent  {ai_answer}")

    except Exception as e:
        print(f"An Error occurred while processing the message: {e}")
        error_msg = "نواجه صعوبة في معالجة طلبك حالياً، يرجى إعادة المحاولة بعد قليل."
        whatsapp_service.send_message(to_number=user_number, message_body=error_msg)

    return Response(content="<Response></Response>", media_type="text/xml")