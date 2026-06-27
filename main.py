import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Response
from src.config import settings
from src.llm_chain import ai_agent_executor
from src.whatsapp_service import WhatsAppService
from src.Rag_engine import initialize_rag_system

@asynccontextmanager
async def lifespan(app: FastAPI):
    pdf_path = os.path.join(settings.DATA_DIR, "company_policy.pdf")
    
    if not os.path.exists(settings.DB_DIR):
        print("🚀 بدء تهيئة نظام RAG والـ Vector Store للمرة الأولى ...")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        if os.path.exists(pdf_path):
            initialize_rag_system(pdf_path)
        else:
            print(f"⚠️ تنبيه حرج: يرجى وضع ملف الـبيانات في المسار: {pdf_path} ثم أعد تشغيل الخادم لبناء الـ RAG.")
    else:
        print("✅ تم العثور على ChromaDB محلياً. النظام جاهز للاسترجاع الفوري.")
        
    yield  
    
    print("🛑 جاري إغلاق خادم الوكيل الذكي بنجاح...")


app = FastAPI(title="Enterprise AI Agent WhatsApp Bot", lifespan=lifespan)
whatsapp_service = WhatsAppService()


@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    user_number = From  
    user_message = Body 
    
    print(f"📩 رسالة واردة من {user_number}: {user_message}")
    
    try:
        response = ai_agent_executor.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": user_number}}
        )
        ai_answer = response.get("answer", "عذراً، لم أستطع معالجة الطلب حالياً.")
        
        whatsapp_service.send_message(to_number=user_number, message_body=ai_answer)
        print(f"📤 تم إرسال الرد بنجاح: {ai_answer}")
        
    except Exception as e:
        print(f"❌ حدث خطأ داخلي في الـ Webhook: {e}")
        error_msg = "نواجه صعوبة في معالجة طلبك حالياً، يرجى إعادة المحاولة بعد قليل."
        whatsapp_service.send_message(to_number=user_number, message_body=error_msg)

    return Response(content="<Response></Response>", media_type="text/xml")