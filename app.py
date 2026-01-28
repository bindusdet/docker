import os
import logging
import validators
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    st.warning("BeautifulSoup not installed. Some web scraping features may not work. Install with: pip install beautifulsoup4")

nest_asyncio.apply()

# LangChain + LLM imports
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# OpenAI
from langchain_openai import ChatOpenAI

# Groq
from langchain_groq import ChatGroq

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="RAG Chatbot App", page_icon="🤖", layout="wide")
st.title("🤖 Unified RAG Chatbot with LangChain, HuggingFace, OpenAI & Groq")

# Initialize session state variables
if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = False
if "config_valid" not in st.session_state:
    st.session_state.config_valid = False

def validate_api_keys(groq_key, hf_token, openai_key, langchain_key, selected_mode):
    """Validate API keys based on selected mode"""
    validation_status = {
        "valid": False,
        "missing_keys": [],
        "message": ""
    }
    
    if selected_mode in ["Chat with Websites (RAG)", "Chat with PDFs (RAG)"]:
        if not groq_key.strip():
            validation_status["missing_keys"].append("Groq API Key")
        if not hf_token.strip():
            validation_status["missing_keys"].append("HuggingFace Token")
    elif selected_mode == "General Chatbot (OpenAI)":
        if not openai_key.strip():
            validation_status["missing_keys"].append("OpenAI API Key")
    
    # LangChain API key is optional but recommended
    if langchain_key.strip():
        validation_status["has_langchain"] = True
    
    if not validation_status["missing_keys"]:
        validation_status["valid"] = True
        validation_status["message"] = "✅ All required API keys are provided!"
    else:
        missing = ", ".join(validation_status["missing_keys"])
        validation_status["message"] = f"⚠️ Missing required keys: {missing}"
    
    return validation_status

def validate_groq_api_key(api_key):
    """Validate Groq API key by making a simple test request"""
    if not api_key.strip():
        return False, "API key is empty"
    
    try:
        from langchain_groq import ChatGroq
        # Test with minimal request
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key,
            temperature=0,
            max_tokens=10
        )
        # Simple test message
        response = llm.invoke("Hi")
        return True, "API key is valid"
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg:
            return False, "Invalid or expired API key"
        elif "404" in error_msg:
            return False, "Model not found - check Groq service status"
        else:
            return False, f"API test failed: {str(e)[:100]}..."

def validate_openai_api_key(api_key):
    """Validate OpenAI API key by making a simple test request"""
    if not api_key.strip():
        return False, "API key is empty"
    
    try:
        from langchain_openai import ChatOpenAI
        # Test with minimal request
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0,
            max_tokens=10
        )
        # Simple test message
        response = llm.invoke("Hi")
        return True, "API key is valid"
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg:
            return False, "Invalid or expired API key"
        elif "429" in error_msg:
            return False, "Rate limit exceeded - API key is valid but quota reached"
        else:
            return False, f"API test failed: {str(e)[:100]}..."

def initialize_environment(groq_key, hf_token, openai_key, langchain_key):
    """Initialize environment variables"""
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["HF_TOKEN"] = hf_token
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Unified RAG Chatbot"

# Sidebar for API Keys and settings
with st.sidebar:
    st.header("🔑 API Keys & Config")
    
    # API Key inputs
    groq_api_key = st.text_input(
        "Groq API Key *", 
        value=os.environ.get("GROQ_API_KEY", ""), 
        type="password",
        help="Required for Website RAG and PDF Chat modes"
    )
    
    hf_token = st.text_input(
        "HuggingFace Token *", 
        value=os.environ.get("HF_TOKEN", ""), 
        type="password",
        help="Required for Website RAG and PDF Chat modes (embeddings)"
    )
    
    openai_api_key = st.text_input(
        "OpenAI API Key *", 
        value=os.environ.get("OPENAI_API_KEY", ""), 
        type="password",
        help="Required for OpenAI GPT models"
    )
    
    langchain_api_key = st.text_input(
        "LangChain API Key (Optional)", 
        value=os.environ.get("LANGCHAIN_API_KEY", ""), 
        type="password",
        help="Optional: For LangSmith tracing and monitoring"
    )

    st.header("⚙️ Mode & Model Settings")
    mode = st.radio(
        "Select Mode:", 
        ["Chat with Websites (RAG)", "Chat with PDFs (RAG)", "General Chatbot (OpenAI)"],
        help="Choose your preferred interaction mode"
    )

    if mode == "General Chatbot (OpenAI)":
        st.subheader("OpenAI Settings")
        llm_choice = st.selectbox("Choose OpenAI Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o","gpt-5"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 50, 2000, 500)
    
    # Validation and initialization section
    st.header("🚀 Initialize App")
    
    # Validate configuration
    validation = validate_api_keys(groq_api_key, hf_token, openai_api_key, langchain_api_key, mode)
    st.session_state.config_valid = validation["valid"]
    
    # Show validation status
    if validation["valid"]:
        st.success(validation["message"])
    else:
        st.warning(validation["message"])
    
    # Add API key test buttons
    if groq_api_key.strip() and mode in ["Chat with Websites (RAG)", "Chat with PDFs (RAG)"]:
        if st.button("🔍 Test Groq API Key", help="Verify your API key works"):
            with st.spinner("Testing Groq API key..."):
                is_valid, message = validate_groq_api_key(groq_api_key)
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    if openai_api_key.strip() and mode == "General Chatbot (OpenAI)":
        if st.button("🔍 Test OpenAI API Key", help="Verify your API key works"):
            with st.spinner("Testing OpenAI API key..."):
                is_valid, message = validate_openai_api_key(openai_api_key)
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Initialize button
    if st.button(
        "🚀 Initialize & Enter App", 
        disabled=not validation["valid"],
        help="Click to initialize the app with your configuration" if validation["valid"] else "Please provide all required API keys first"
    ):
        if validation["valid"]:
            # Initialize environment
            initialize_environment(groq_api_key, hf_token, openai_api_key, langchain_api_key)
            st.session_state.app_initialized = True
            st.success("✅ App initialized successfully!")
            st.rerun()
    
    # Show initialization status
    if st.session_state.app_initialized:
        st.success("✅ App is ready!")
        if st.button("🔄 Reset App"):
            st.session_state.app_initialized = False
            st.session_state.config_valid = False
            st.rerun()

# HuggingFace Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Main content area
if not st.session_state.app_initialized:
    # Show welcome message and instructions when not initialized
    st.info("📜 **Welcome to the Unified RAG Chatbot!**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🌐 Chat with Websites (RAG)
        - Load and chat with any website content
        - Advanced web scraping and processing
        - Conversational Q&A with retrieval
        
        **Required:** Groq API Key, HuggingFace Token
        """)
    
    with col2:
        st.markdown("""
        ### 📄 Chat with PDFs (RAG)
        - Upload and chat with PDF documents
        - Conversational memory
        - Context-aware responses
        
        **Required:** Groq API Key, HuggingFace Token
        """)
    
    with col3:
        st.markdown("""
        ### 💬 General Chatbot (OpenAI)
        - GPT-4o-mini powered conversation
        - State-of-the-art language model
        - Adjustable parameters
        
        **Required:** OpenAI API Key
        """)
    
    st.markdown("---")
    st.markdown("📝 **Instructions:**")
    st.markdown("""
    1. 🔑 **Configure API Keys:** Enter your API keys in the sidebar
    2. ⚙️ **Select Mode:** Choose your preferred interaction mode
    3. 🚀 **Initialize:** Click the 'Initialize & Enter App' button
    4. 🎉 **Start Using:** Begin using the chatbot features!
    """)
    
    # Show current configuration status
    st.markdown("### 🔍 Current Configuration Status")
    if hasattr(st.session_state, 'config_valid') and st.session_state.config_valid:
        st.success("Configuration is valid! Click 'Initialize & Enter App' in the sidebar to proceed.")
    else:
        st.warning("Please complete the configuration in the sidebar.")

else:
    # App is initialized - show main functionality
    embeddings = get_embeddings()  # Initialize embeddings only when needed
    
    # Main content based on mode selection
    if mode == "Chat with Websites (RAG)":
        st.subheader("🌐 Conversational RAG with Website Content & Chat History")

        if not groq_api_key or not hf_token:
            st.warning("Please enter your Groq API Key and HuggingFace Token in the sidebar to proceed.")
        else:
            # Session management
            session_id = st.text_input("Enter a unique Session ID", value="web_session")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("Clear Chat History"):
                    if session_id in st.session_state.chat_history:
                        st.session_state.chat_history[session_id].clear()
                        st.success(f"Chat history for session '{session_id}' cleared.")
            
            with col2:
                if st.button("Clear Vector Store", help="Clear stored website content"):
                    if f"web_vectorstore_{session_id}" in st.session_state:
                        del st.session_state[f"web_vectorstore_{session_id}"]
                        st.success("Vector store cleared. Load new content below.")
            
            # URL input and processing
            st.markdown("### 📖 Load Website Content")
            website_url = st.text_input(
                "Enter Website URL:",
                placeholder="https://example.com",
                help="Enter any website URL to load and chat with its content"
            )
            
            # URL validation and loading
            if website_url:
                if validators.url(website_url):
                    st.success("✅ Valid URL format")
                    st.info("🌐 Website detected - ready to load content")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        load_btn = st.button("🔄 Load Website Content", type="primary")
                    
                    with col2:
                        if st.button("🔍 Test URL", help="Check if URL is accessible"):
                            with st.spinner("Testing URL accessibility..."):
                                try:
                                    import requests
                                    response = requests.head(website_url, timeout=10, allow_redirects=True)
                                    if response.status_code == 200:
                                        st.success("✅ URL is accessible!")
                                    else:
                                        st.warning(f"⚠️ URL returned status code: {response.status_code}")
                                except Exception as test_error:
                                    st.error(f"❌ URL test failed: {str(test_error)}")
                    
                    # Load and process website content
                    if load_btn:
                        with st.spinner("Loading and processing website content..."):
                            try:
                                # Step 1: Load content
                                docs = None
                                st.info("📥 Step 1: Loading website content...")
                                
                                # Try primary method with UnstructuredURLLoader
                                try:
                                    headers = {
                                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                        "Accept-Language": "en-US,en;q=0.5",
                                    }
                                    loader = UnstructuredURLLoader(
                                        urls=[website_url], 
                                        ssl_verify=False, 
                                        headers=headers
                                    )
                                    docs = loader.load()
                                    st.success("✅ Content loaded successfully!")
                                except Exception as url_error:
                                    st.warning(f"⚠️ Primary method failed: {str(url_error)}")
                                    
                                    # Fallback method with requests + BeautifulSoup
                                    st.info("🔄 Trying alternative extraction method...")
                                    try:
                                        import requests
                                        from bs4 import BeautifulSoup
                                        from langchain_core.documents import Document
                                        
                                        headers = {
                                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                                        }
                                        
                                        response = requests.get(website_url, headers=headers, timeout=30)
                                        response.raise_for_status()
                                        
                                        soup = BeautifulSoup(response.content, 'html.parser')
                                        
                                        # Remove unwanted elements
                                        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                                            element.decompose()
                                        
                                        # Extract main content
                                        text = soup.get_text()
                                        lines = (line.strip() for line in text.splitlines())
                                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                                        clean_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 3)
                                        
                                        if clean_text and len(clean_text.strip()) > 100:
                                            docs = [Document(page_content=clean_text, metadata={"source": website_url, "title": soup.title.string if soup.title else "Unknown"})]
                                            st.success("✅ Content extracted using fallback method!")
                                        else:
                                            raise Exception("No meaningful content extracted")
                                            
                                    except Exception as fallback_error:
                                        st.error(f"❌ All extraction methods failed: {str(fallback_error)}")
                                        st.error("Please try a different URL or check if the website is accessible.")
                                        docs = None
                                
                                if docs and len(docs) > 0:
                                    # Step 2: Split documents
                                    st.info("📝 Step 2: Splitting content into chunks...")
                                    text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000, 
                                        chunk_overlap=200,
                                        separators=["\n\n", "\n", ".", " ", ""]
                                    )
                                    splits = text_splitter.split_documents(docs)
                                    st.success(f"✅ Content split into {len(splits)} chunks")
                                    
                                    # Step 3: Create embeddings and vector store
                                    st.info("🔗 Step 3: Creating embeddings and vector store...")
                                    vectorstore = Chroma.from_documents(
                                        documents=splits, 
                                        embedding=embeddings,
                                        collection_name=f"web_collection_{session_id}"
                                    )
                                    st.session_state[f"web_vectorstore_{session_id}"] = vectorstore
                                    st.success("✅ Vector store created successfully!")
                                    
                                    # Display content stats
                                    total_content = " ".join([doc.page_content for doc in docs])
                                    st.info(f"📊 **Content Stats**: {len(docs)} document(s), {len(splits)} chunks, {len(total_content)} characters")
                                    
                                    # Show a preview of the content
                                    with st.expander("📋 Content Preview"):
                                        st.write(f"**Title**: {docs[0].metadata.get('title', 'Unknown')}")
                                        st.write(f"**Source**: {website_url}")
                                        st.write(f"**Content Preview**:")
                                        st.write(total_content[:500] + "..." if len(total_content) > 500 else total_content)
                                    
                                    st.success("🎉 **Website content loaded and ready for questions!**")
                                    
                                else:
                                    st.error("❌ Failed to load website content. Please try a different URL.")
                                    
                            except Exception as e:
                                st.error(f"❌ **Error processing website**: {str(e)}")
                                with st.expander("🔧 Error Details"):
                                    st.exception(e)
                else:
                    st.warning("⚠️ Invalid URL format - please check your URL")
            
            # Chat interface
            st.markdown("### 💬 Chat with Website Content")
            
            # Check if we have a vector store for this session
            if f"web_vectorstore_{session_id}" in st.session_state:
                vectorstore = st.session_state[f"web_vectorstore_{session_id}"]
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # Initialize LLM
                try:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        groq_api_key=groq_api_key,
                        temperature=0.1
                    )
                except Exception as model_error:
                    st.warning("Primary model failed, using fallback...")
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        groq_api_key=groq_api_key,
                        temperature=0.1
                    )
                
                # Create RAG chain
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant for question-answering tasks about website content. Use the following pieces of retrieved context to answer the question. If you don't know the answer based on the context, just say that you don't know. Use three sentences maximum and keep the answer concise and helpful.\n\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                web_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                def get_session_history(session: str) -> BaseChatMessageHistory:
                    if session not in st.session_state.store:
                        st.session_state.store[session] = ChatMessageHistory()
                    return st.session_state.store[session]
                
                conversational_web_rag_chain = RunnableWithMessageHistory(
                    web_rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
                
                # Initialize chat history for this session
                if session_id not in st.session_state.chat_history:
                    st.session_state.chat_history[session_id] = get_session_history(session_id)
                
                # Display chat history
                if st.session_state.chat_history[session_id].messages:
                    for msg in st.session_state.chat_history[session_id].messages:
                        st.chat_message(msg.type).write(msg.content)
                
                # Chat input
                if user_input := st.chat_input("Ask a question about the website content:"):
                    st.chat_message("user").write(user_input)
                    
                    try:
                        with st.spinner("Searching and generating response..."):
                            response = conversational_web_rag_chain.invoke(
                                {"input": user_input},
                                config={"configurable": {"session_id": session_id}}
                            )
                        st.chat_message("assistant").write(response["answer"])
                        
                        # Show retrieved context in expander
                        if "context" in response and response["context"]:
                            with st.expander("📚 Retrieved Context"):
                                for i, doc in enumerate(response["context"][:2]):  # Show top 2 contexts
                                    st.write(f"**Context {i+1}:**")
                                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                    if "source" in doc.metadata:
                                        st.write(f"*Source: {doc.metadata['source']}*")
                                    st.markdown("---")
                                    
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
                        st.info("Please try rephrasing your question or check your API keys.")
            else:
                st.info("📝 **Please load website content first** using the 'Load Website Content' button above.")
                st.markdown("""
                **How to use:**
                1. 🔗 Enter a website URL above
                2. 🔄 Click "Load Website Content" to process the website
                3. 💬 Ask questions about the website content below
                
                **Features:**
                - 🧠 Intelligent content extraction and chunking
                - 🔍 Vector similarity search for relevant information
                - 💬 Conversational memory across questions
                - 📚 Source citations and context display
                """)

    elif mode == "Chat with PDFs (RAG)":
        st.subheader("📄 Conversational RAG with PDFs & Chat History")

        if not groq_api_key:
            st.warning("Please enter your Groq API Key in the sidebar to proceed.")
        else:
            session_id = st.text_input("Enter a unique Session ID", value="default_session")

            if st.button("Clear Chat History"):
                if session_id in st.session_state.chat_history:
                    st.session_state.chat_history[session_id].clear()
                    st.success(f"Chat history for session '{session_id}' cleared.")

            uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    documents = []
                    for uploaded_file in uploaded_files:
                        with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        loader = PyPDFLoader(os.path.join("/tmp", uploaded_file.name))
                        documents.extend(loader.load())

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(documents)
                    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                    retriever = vectorstore.as_retriever()

                    # Use correct Groq model with error handling
                    try:
                        llm = ChatGroq(
                            model="llama-3.1-8b-instant",
                            groq_api_key=groq_api_key,
                            temperature=0.1
                        )
                    except Exception as model_error:
                        st.warning("Primary model failed, using fallback...")
                        llm = ChatGroq(
                            model="llama-3.3-70b-versatile",
                            groq_api_key=groq_api_key,
                            temperature=0.1
                        )

                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                    def get_session_history(session: str) -> BaseChatMessageHistory:
                        if session not in st.session_state.store:
                            st.session_state.store[session] = ChatMessageHistory()
                        return st.session_state.store[session]

                    conversational_rag_chain = RunnableWithMessageHistory(
                        rag_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )
                    
                    st.session_state.chat_history[session_id] = get_session_history(session_id)

                    # Display chat history
                    if st.session_state.chat_history[session_id].messages:
                        for msg in st.session_state.chat_history[session_id].messages:
                            st.chat_message(msg.type).write(msg.content)

                    if user_input := st.chat_input("Ask a question about your PDFs:"):
                        st.chat_message("user").write(user_input)
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        st.chat_message("assistant").write(response["answer"])

    elif mode == "General Chatbot (OpenAI)":
        st.subheader("💬 General Chatbot using OpenAI GPT")

        if not openai_api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
        else:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("What is up?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        llm = ChatOpenAI(
                            model=llm_choice,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            openai_api_key=openai_api_key
                        )
                        
                        response = llm.invoke(prompt)
                        # Extract content from AIMessage response
                        response_content = response.content if hasattr(response, 'content') else str(response)
                        st.markdown(response_content)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                        
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
                        st.info("Please check your OpenAI API key and try again.")