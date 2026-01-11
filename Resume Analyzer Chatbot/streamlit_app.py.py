import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import shutil
import time
import gc
import uuid

# Load environment variables
load_dotenv(find_dotenv())

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'db_path' not in st.session_state:
    st.session_state.db_path = None

def cleanup_vectorstore():
    """Properly cleanup vectorstore and release all locks"""
    try:
        # Close and delete the vectorstore object
        if st.session_state.vectorstore is not None:
            # Try to delete the collection first
            try:
                st.session_state.vectorstore.delete_collection()
            except:
                pass
            
            # Clear the reference
            st.session_state.vectorstore = None
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Wait for locks to release
        time.sleep(1)
        
        return True
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")
        return False

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    # Model settings
    st.subheader("🤖 Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, 
                           help="Lower values make output more focused and deterministic")
    top_k = st.slider("Retrieved Chunks", 1, 10, 5, 1,
                     help="Number of document chunks to retrieve")
    
    st.markdown("---")
    st.subheader("📊 Document Info")
    if st.session_state.pdf_processed:
        st.success("✅ PDF Loaded")
        st.info(f"📄 {st.session_state.current_pdf_name}")
        if st.button("🔄 Upload New PDF"):
            with st.spinner("Cleaning up..."):
                # Cleanup vectorstore properly
                cleanup_vectorstore()
                
                # Reset session state
                st.session_state.chat_history = []
                st.session_state.pdf_processed = False
                st.session_state.current_pdf_name = None
                st.session_state.db_path = None
                
                # Clean up uploads directory
                if os.path.exists("./uploads"):
                    try:
                        shutil.rmtree("./uploads")
                    except:
                        pass
            
            st.success("✅ Ready for new upload!")
            time.sleep(1)
            st.rerun()
    else:
        st.info("📄 No PDF loaded yet")

# Main content
st.title("📚 Resume Analyzer Chatbot")
st.markdown("Upload a Resume in PDF format and ask questions about its content!")

# PDF Upload Section
if not st.session_state.pdf_processed:
    st.markdown("### 📤 Upload Your Resume")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing your resume... This may take a moment."):
            try:
                # Ensure old vectorstore is cleaned up
                cleanup_vectorstore()
                
                # Create unique database path for this upload
                unique_id = str(uuid.uuid4())[:8]
                db_path = f"./chroma_db_{unique_id}"
                st.session_state.db_path = db_path
                
                # Save uploaded file temporarily
                upload_dir = Path("./uploads")
                upload_dir.mkdir(exist_ok=True)
                pdf_path = upload_dir / uploaded_file.name
                
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Load and process PDF
                pdf_loader = PyPDFLoader(str(pdf_path))
                docs = pdf_loader.load()
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(docs)
                
                # Create FRESH vector store with unique path
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )
                
                st.session_state.pdf_processed = True
                st.session_state.current_pdf_name = uploaded_file.name
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                st.info(f"📊 Created {len(chunks)} chunks from {len(docs)} pages")
                st.info(f"🗄️ Database: {db_path}")
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Chat Interface (only show if PDF is processed)
if st.session_state.pdf_processed and st.session_state.vectorstore:
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 📜 Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i+1}:** {chat['question']}")
                st.markdown(f"**A{i+1}:** {chat['answer']}")
                st.markdown("---")
    
    # Question input
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the main topics discussed in this document?",
            height=100,
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit_button = st.form_submit_button("🚀 Ask Question", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize LLM
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model="openai/gpt-oss-120b",
                    temperature=temperature
                )
                
                # Create retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": top_k}
                )
                
                # Define template
                template = """Answer the question strictly based only on the following context:
                            {context}
                            Question: {question}
                            Answer in detail:
                            """
                
                prompt = ChatPromptTemplate.from_template(template)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # Build RAG chain
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Get answer
                answer = rag_chain.invoke(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })
                
                # Display the latest answer prominently
                st.markdown("### ✨ Answer")
                st.success(answer)
                
                # Auto-scroll would happen on rerun
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    elif submit_button:
        st.warning("⚠️ Please enter a question!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built using LangChain, Groq & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)