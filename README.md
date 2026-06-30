# Arabic Healthcare & Pharmacy WhatsApp Bot (RAG)

An Arabic-language WhatsApp customer support bot designed for pharmacies and healthcare companies. Built using a **Retrieval-Augmented Generation (RAG)** pipeline, the bot answers customer inquiries regarding products, pricing, shipping, and return policies by retrieving context from internal company documents. It generates grounded, accurate responses and features a graceful fallback mechanism to hand off complex queries to human agents.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Twilio](https://img.shields.io/badge/Twilio-F22F46?style=for-the-badge&logo=twilio&logoColor=white)

## 🚀 Features
* **RAG-Powered Support:** Retrieves real-time context from company documentation to answer specific inquiries accurately in Arabic.
* **Smart Fallback:** Automatically detects when information is missing or unreliable and hands off the conversation to a human support agent.
* **Persistent Memory:** Uses LangGraph's checkpointer to maintain conversational state and context per user across unique WhatsApp threads.
* **Production-Ready API:** Built on a robust asynchronous backend capable of handling incoming webhooks.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** FastAPI
* **Orchestration & Memory:** LangGraph, LangChain
* **Vector Database:** ChromaDB
* **LLM & Embeddings:** Google Gemini Models (Chat + Embeddings)
* **Messaging Gateway:** Twilio API

### Key Highlights
* **Grounded Responses:** Employs Google Gemini models and ChromaDB to ensure answers are strictly tied to uploaded healthcare and policy documents.
* **Human-in-the-Loop:** Gracefully routes the user to a human agent if the RAG confidence score drops below a safe threshold.
* **State Management:** Keeps conversation context intact per user via LangGraph’s native checkpointing system.
