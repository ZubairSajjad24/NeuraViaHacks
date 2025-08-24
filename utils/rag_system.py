import re
import random

# Hugging Face / LangChain
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline


# ---------------------------
# Rule-based responses
# ---------------------------
RESPONSE_RULES = {
    r"symptom|sign|feel": [
        "Common early symptoms include tremors, stiffness, and balance issues.",
        "Many people experience slight tremors in their hands or fingers as an early sign."
    ],
    r"treat|medication|drug": [
        "Treatment often includes medications like Levodopa and physical therapy.",
        "Doctors may prescribe various medications to manage symptoms effectively."
    ],
    r"exercise|activity|physical": [
        "Regular exercise like walking or swimming can help maintain mobility.",
        "Physical therapy is often recommended to improve balance and coordination."
    ],
    r"diet|food|eat": [
        "A balanced diet with plenty of fiber can help manage symptoms.",
        "Some people find that certain dietary changes help with their symptoms."
    ],
    r"risk|chance|likely": [
        "Risk factors include age, family history, and exposure to certain toxins.",
        "The risk increases with age, but Parkinson's can affect people of all ages."
    ]
}

DEFAULT_RESPONSES = [
    "I'm here to help with information about neurological health.",
    "That's a good question about Parkinson's disease management.",
    "I can provide general information, but please consult a doctor for medical advice."
]


def get_response(query, risk_score=None):
    """Get a response based on the query using rule-based matching"""
    query = query.lower()
    
    # Check if any regex pattern matches the query
    for pattern, responses in RESPONSE_RULES.items():
        if re.search(pattern, query):
            return random.choice(responses)
    
    # Return a default response if nothing matched
    return random.choice(DEFAULT_RESPONSES)


# ---------------------------
# Setup RAG system
# ---------------------------
def setup_rag():
    try:
        # Load documents from your knowledge base
        loader = TextLoader("data/knowledge_base/medical_guidelines.txt")
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Build vectorstore with embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Use Flan-T5 correctly
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        return qa_chain
    
    except Exception as e:
        print(f"Error setting up RAG: {e}")
        return None
