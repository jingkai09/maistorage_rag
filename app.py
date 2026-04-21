import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup environment & credentials
# Note: Using st.secrets is safer for production, but hardcoded for this demo
KEY = "AIzaSyBcyIXuLWtSGhQnOwBphzUEa8GzGMtM1F8" 
os.environ["GOOGLE_API_KEY"] = KEY

st.set_page_config(page_title="MaiStorage RAG Engine", layout="wide")

# Internal AI Models
engine = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
vector_tool = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Track conversation and file status
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "docs_loaded" not in st.session_state:
    st.session_state.uploaded_files = [] 

# Sidebar Management
with st.sidebar:
    st.title("Admin Panel")
    st.subheader("Upload Knowledge")
    pdf_input = st.file_uploader("Drop Phison PDFs here", type="pdf")
    
    if pdf_input:
        if pdf_input.name not in st.session_state.uploaded_files:
            with st.status(f"Processing {pdf_input.name}...") as status:
                # Local handling
                path = f"local_{pdf_input.name}"
                with open(path, "wb") as f:
                    f.write(pdf_input.getbuffer())
                
                # Custom Chunking Logic
                raw_data = PyPDFLoader(path).load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
                chunks = splitter.split_documents(raw_data)
                
                # Build/Update local vector store
                if "vector_db" not in st.session_state:
                    st.session_state.vector_db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=vector_tool
                    )
                else:
                    st.session_state.vector_db.add_documents(chunks)
                
                st.session_state.uploaded_files.append(pdf_input.name)
                status.update(label="Index Updated!", state="complete")
    
    if st.session_state.uploaded_files:
        st.info(f"Connected to: {', '.join(st.session_state.uploaded_files)}")

    if st.sidebar.button("Clear App Data"):
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        if "vector_db" in st.session_state:
            del st.session_state.vector_db
        st.rerun()

# Main Chat Interface
st.title("🤖 Technical Support Agent")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "refs" in msg:
            with st.expander("Show Evidence"):
                st.caption(msg["refs"])

# Query Handling
user_input = st.chat_input("Ask about model specs, hardware, or errors...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    if "vector_db" not in st.session_state:
        st.error("Knowledge base is empty. Please upload a PDF to start.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documentation..."):
                # Search across all uploaded files
                results = st.session_state.vector_db.similarity_search(user_input, k=4)
                
                # Format references cleanly
                evidence_list = []
                for r in results:
                    file_src = os.path.basename(r.metadata.get("source", "Doc"))
                    page_num = r.metadata.get("page", 0) + 1
                    snippet = r.page_content[:200].replace("\n", " ")
                    evidence_list.append(f"**{file_src} (p.{page_num})**: ...{snippet}...")

                citation_text = "\n\n".join(evidence_list)
                
                # Building the prompt contextually
                context_block = "\n\n".join([f"[{d.metadata.get('source')} Page {d.metadata.get('page')+1}] {d.page_content}" for d in results])
                chat_mem = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                
                prompt_template = f"""You are a specialized engineer for MaiStorage. 
                Use the following context to answer precisely. If the data isn't there, say you don't know.
                
                PREVIOUS CHAT:
                {chat_mem}
                
                TECHNICAL CONTEXT:
                {context_block}
                
                USER QUESTION: {user_input}"""
                
                bot_reply = engine.invoke(prompt_template).content

            st.write(bot_reply)
            with st.expander("Show Evidence"):
                st.caption(citation_text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_reply, 
                "refs": citation_text
            })
