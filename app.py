import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
# Use your brand-new, private API Key from the new Google Project
API_KEY = "AIzaSyBcyIXuLWtSGhQnOwBphzUEa8GzGMtM1F8" 
os.environ["GOOGLE_API_KEY"] = API_KEY

st.set_page_config(page_title="MaiStorage Agentic RAG", layout="wide")

# Initialize Gemini 2.5 Flash & stable Embedding model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)

# --- 2. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = [] 

# --- 3. SIDEBAR: DYNAMIC INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    st.write("Upload technical PDFs to build the Agent's knowledge base.")
    uploaded_file = st.file_uploader("Upload Phison/MaiStorage PDF", type="pdf")
    
    if uploaded_file:
        # Check if this file has already been indexed to avoid redundant API calls
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Agent is indexing {uploaded_file.name}..."):
                # Save temp file locally for the loader
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                data = loader.load()
                
                # Recursive splitting preserves technical context better than standard splitting
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                chunks = text_splitter.split_documents(data)
                
                # Append documents to the existing Chroma collection
                if "db" not in st.session_state:
                    st.session_state.db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        collection_name="demo_collection"
                    )
                else:
                    st.session_state.db.add_documents(chunks)
                
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(f"✅ Added {uploaded_file.name}")
    
    # List active knowledge sources
    if st.session_state.uploaded_files:
        st.write("**Active Sources:**")
        for f_name in st.session_state.uploaded_files:
            st.caption(f"• {f_name}")

    if st.button("🗑️ Reset Agent Memory"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        if "db" in st.session_state:
            del st.session_state.db
        st.rerun()

# --- 4. CHAT UI ---
st.title("🤖 MaiStorage Technical Agent")
st.caption("Agentic RAG with Multi-turn Memory and Cross-Document Retrieval")

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message:
            with st.expander("📚 View Evidence"):
                st.markdown(message["references"])

# --- 5. AGENTIC LOGIC ---
query = st.chat_input("Ask a technical question...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload a technical document in the sidebar to begin.")
    else:
        with st.chat_message("assistant"):
            with st.status("Thinking (Retrieve -> Rerank -> Synthesize)..."):
                # Retrieval: Fetch the top 4 most relevant chunks
                docs = st.session_state.db.similarity_search(query, k=4)
                
                # Group references by filename and page
                sources_found = {}
                for doc in docs:
                    filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0) + 1
                    key = f"{filename} (Page {page})"
                    if key not in sources_found:
                        sources_found[key] = []
                    sources_found[key].append(doc.page_content[:250].replace("\n", " "))

                # Construct citation string for the expander
                ref_string = ""
                for source, snippets in sources_found.items():
                    ref_string += f"**{source}**\n"
                    for s in snippets:
                        ref_string += f"> ...{s}...\n\n"
                
                # Prepare context and memory for the LLM
                context = "\n\n".join([f"[Source: {d.metadata.get('source')} Page {d.metadata.get('page')+1}] {d.page_content}" for d in docs])
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:-1]])
                
                prompt = f"""You are a technical assistant for MaiStorage. 
                Answer the question based ONLY on the provided CONTEXT. 
                Use the CHAT HISTORY to understand follow-up questions.
                
                CHAT HISTORY:
                {history_str}
                
                CONTEXT:
                {context}
                
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
                answer = response.content

            # Display final answer with hidden references
            st.markdown(answer)
            with st.expander("📚 View Evidence"):
                st.markdown(ref_string)
            
            # Save assistant response and references to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer, 
                "references": ref_string
            })
