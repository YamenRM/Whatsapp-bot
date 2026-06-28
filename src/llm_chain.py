from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from src.config import settings
from src.Rag_engine import get_retriever


class State(TypedDict):
    input: str
    chat_history: Annotated[list, add_messages]
    context: str
    answer: str

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
        "أنت مساعد ذكاء اصطناعي محترف مهني ومختص لخدمة عملاء شركة فارما كير للمنتجات الدوائية والصحية.\n"
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
    
    
    def call_rag_chain(state: State):
        response = rag_chain.invoke({
            "input": state["input"],
            "chat_history": state["chat_history"]
        })
        
        return {
            "chat_history": [
                ("human", state["input"]),
                ("ai", response["answer"])
            ],
            "answer": response["answer"]
        }

    workflow = StateGraph(State)
    workflow.add_node("rag_agent", call_rag_chain)
    workflow.add_edge(START, "rag_agent")
    
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    
    return compiled_graph


ai_agent_executor = build_conversational_rag_chain()
