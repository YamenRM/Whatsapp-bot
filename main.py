import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Response
from langchain_core.runnables import RunnableConfig
from src.config import settings
from src.llm_chain import ai_agent_executor, State
from src.whatsapp_service import WhatsAppService
from src.Rag_engine import initialize_rag_system

@asynccontextmanager
async def lifespan(app: FastAPI):
    pdf_path = os.path.join(settings.DATA_DIR, "Company_policy.pdf")
    if not os.path.exists(settings.DB_DIR):
        print("🚀 بدء تهيئة نظام RAG والـ Vector Store للمرة الأولى...")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(pdf_path):
            initialize_rag_system(pdf_path)
        else:
            print(f"⚠️ تنبيه حرج: يرجى وضع ملف الـ PDF في المسار: {pdf_path}")
    else:
        print("✅ تم العثور على ChromaDB محلياً. النظام جاهز للاسترجاع.")
    yield
    print("🛑 جاري إغلاق خادم الوكيل الذكي بنجاح...")

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
        print(f"📤 تم إرسال رد LangGraph بنجاح: {ai_answer}")
        
    except Exception as e:
        print(f"❌ حدث خطأ داخلي في الـ Webhook الخاص بـ LangGraph: {e}")
        error_msg = "نواجه صعوبة في معالجة طلبك حالياً، يرجى إعادة المحاولة بعد قليل."
        whatsapp_service.send_message(to_number=user_number, message_body=error_msg)

    return Response(content="<Response></Response>", media_type="text/xml")