from langchain_google_genai import ChatGoogleGenerativeAI
# استيراد الدوال الحديثة والمستقرة من الحزمة الكلاسيكية
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.config import settings
from .Rag_engine import get_retriever

sessions_db = {}

def get_session_history(session_id: str):
    if session_id not in sessions_db:
        sessions_db[session_id] = ChatMessageHistory()
    return sessions_db[session_id]

def build_conversational_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL, 
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.3
    )
    
    base_retriever = get_retriever()

    contextualize_q_system_prompt = (
        "بالنظر إلى تاريخ الدردشة وأحدث سؤال من المستخدم والذي قد يشير إلى سياق في تاريخ الدردشة، "
        "قم بصياغة سؤال مستقل يمكن فهمه دون الحاجة لتاريخ الدردشة. "
        "لا تقم بالإجابة على السؤال، فقط أعد صياغته إذا لزم الأمر، وإلا أعده كما هو."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever, contextualize_q_prompt
    )
    
    system_prompt = (
        "أنت مساعد ذكاء اصطناعي محترف مهني ومختص لخدمة عملاء شركة أدوية و منتجات صحية.\n"
        "استخدم الفقرات المسترجعة التالية فقط للإجابة على سؤال المستخدم بدقة عالية و بدون هلوسة.\n"
        "إذا لم تكن الإجابة موجودة في النص المسترجع، قل بلطف: 'عذراً، ليس لدي علم دقيق بهذه المعلومة حالياً، سأقوم بتحويلك للموظف المختص لمساعدتك.' دون اختراع أي تفاصيل.\n"
        "تحدث بلهجة ودية، محترفة، ومباشرة.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain


ai_agent_executor = build_conversational_rag_chain()
