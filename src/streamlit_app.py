
import json
import requests
import streamlit as st
from sseclient import SSEClient

# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .upload-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
    }

    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# -----------------------------
# API Functions
# -----------------------------

def check_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def upload_document(uploaded_file):
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=300
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        st.error(f"Upload failed: {str(e)}")
        return None


def stream_response(question):
    try:
        response = requests.post(
            f"{API_URL}/ask/stream",
            json={"question": question},
            stream=True,
            timeout=(10, 300)
        )

        response.raise_for_status()

        client = SSEClient(response)

        for event in client.events():

            if not event.data:
                continue
            
            if event.event == "done":
                break
            
            if event.event == "sources":
                sources = json.loads(event.data)
                continue
            
            if event.event == "token":
                token = json.loads(event.data)
                yield token

            elif event.event == "error":
                try:
                    data = json.loads(event.data)
                    yield f"\n\n⚠️ {data.get('message', 'Generation failed.')}"
                except json.JSONDecodeError:
                    yield "\n\n⚠️ Generation failed"

    except requests.RequestException as e:
        yield f"\n\n⚠️ API connection error: {str(e)}"    

    finally:
        if "response" in locals():
            response.close()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("📚 RAG Assistant")

    st.markdown("---")

    st.subheader("API Status")

    if check_api_health():
        st.success("Backend connected")
    else:
        st.error("Backend unavailable")

    st.markdown("---")

    st.subheader("Navigation")

    page = st.radio(
        "Select a section",
        ["💬 Chat", "📄 Documents"]
    )

    st.markdown("---")

    st.caption("Powered by FastAPI, Ollama and Weaviate")


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<p class="main-title">Knowledge Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Ask questions about your uploaded documents using AI.</p>',
    unsafe_allow_html=True
)


# =====================================================
# DOCUMENT UPLOAD SECTION
# =====================================================

if page == "📄 Documents":

    st.header("📄 Document Management")

    st.write(
        "Upload PDF documents to add them to your knowledge base."
    )

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files."
    )

    if uploaded_files:

        st.write(f"**Selected files:** {len(uploaded_files)}")

        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.size / 1024:.2f} KB)")

        if st.button(
            "🚀 Upload and Index Documents",
            type="primary",
            use_container_width=True
        ):

            progress = st.progress(0)
            status = st.empty()

            successful = 0

            for index, file in enumerate(uploaded_files):

                status.info(f"Processing {file.name}...")

                result = upload_document(file)

                if result:

                    successful += 1

                    st.session_state.uploaded_documents.append({
                        "filename": result.get("filename", file.name),
                        "chunks": result.get("chunks", 0)
                    })

                    st.success(
                        f"✅ {file.name} uploaded successfully! "
                        f"Chunks: {result.get('chunks', 0)}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            status.success(
                f"Completed! {successful}/{len(uploaded_files)} "
                "documents indexed."
            )

    st.markdown("---")

    st.subheader("📚 Uploaded Documents")

    if st.session_state.uploaded_documents:

        for document in st.session_state.uploaded_documents:

            with st.container(border=True):

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"📄 **{document['filename']}**")

                with col2:
                    st.metric(
                        "Chunks",
                        document["chunks"]
                    )

    else:
        st.info(
            "No documents uploaded during this session yet."
        )


# =====================================================
# CHAT SECTION
# =====================================================

elif page == "💬 Chat":

    st.header("💬 Chat with your documents")

    st.caption(
        "Ask questions and get answers based on your indexed documents."
    )

    # Clear chat button

    col1, col2 = st.columns([5, 1])

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # Display conversation history

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Chat input

    if prompt := st.chat_input(
        "Ask something about your documents..."
    ):

        # Display user message

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            full_response = ""

            with st.spinner("Thinking..."):

                for token in stream_response(prompt):

                    full_response += token

                    response_placeholder.markdown(
                        full_response + "▌"
                    )

            response_placeholder.markdown(full_response)

        # Save assistant response

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
