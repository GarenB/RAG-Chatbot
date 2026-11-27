
import json
import requests
import streamlit as st
from sseclient import SSEClient

# =====================================================
# CONFIGURATION
# =====================================================

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    [data-testid="stSidebar"] {
        min-width: 300px;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
    }

    .document-name {
        font-size: 0.9rem;
    }

    .stChatMessage {
        border-radius: 12px;
    }
    
    /* Chat history titles */
    [data-testid="stSidebar"] button {
        justify-content: flex-start !important;
        text-align: left !important;
    }

    [data-testid="stSidebar"] button p {
        text-align: left !important;
    }
    
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


# =====================================================
# API FUNCTIONS
# =====================================================

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


def delete_document(filename):
    try:
        response = requests.delete(
            f"{API_URL}/documents/{requests.utils.quote(filename, safe='')}",
            timeout=30
        )

        response.raise_for_status()

        st.session_state.uploaded_documents = [
            doc for doc in st.session_state.uploaded_documents
            if doc["filename"] != filename
        ]

        st.success(f"{filename} deleted successfully.")
        st.rerun()

    except requests.RequestException as e:
        st.error(f"Document deletion failed: {str(e)}")


def stream_response(question):
    response = None

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
                continue

            if event.event == "token":
                yield json.loads(event.data)

            elif event.event == "error":
                try:
                    data = json.loads(event.data)
                    yield (
                        "\n\n⚠️ "
                        + data.get("message", "Generation failed.")
                    )
                except json.JSONDecodeError:
                    yield "\n\n⚠️ Generation failed."

    except requests.RequestException as e:
        yield f"\n\n⚠️ API connection error: {str(e)}"

    finally:
        if response:
            response.close()


# =====================================================
# CHAT MANAGEMENT
# =====================================================

def save_current_chat():

    if not st.session_state.messages:
        return

    chat_id = st.session_state.current_chat_id

    if chat_id is None:
        return

    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            chat["messages"] = st.session_state.messages.copy()
            chat["title"] = next(
                (
                    msg["content"][:35]
                    for msg in st.session_state.messages
                    if msg["role"] == "user"
                ),
                "New conversation"
            )
            return


def new_chat():

    save_current_chat()

    st.session_state.messages = []
    st.session_state.current_chat_id = None

    st.rerun()


def open_chat(chat_id):

    save_current_chat()

    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            st.session_state.messages = chat["messages"].copy()
            st.session_state.current_chat_id = chat_id
            break

    st.rerun()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📚 RAG Assistant")

    st.caption("Your personal knowledge base")

    st.markdown("---")

    # -----------------------------
    # HISTORY
    # -----------------------------

    st.subheader("💬 Chats")

    if st.button(
        "+ New Chat",
        use_container_width=True,
        type="secondary"
    ):
        new_chat()

    if st.session_state.chat_history:

        for chat in reversed(st.session_state.chat_history):

            title = chat["title"]
            max_length = 25

            display_title = (
                title[:max_length - 3] + "..."
                if len(title) > max_length
                else title
            )

            if st.button(
                f"💬 {display_title}",
                key=f"chat_{chat['id']}",
                use_container_width=True,
                help=title
            ):
                open_chat(chat["id"])

    st.markdown("---")

    # -----------------------------
    # DOCUMENTS
    # -----------------------------

    st.subheader("📄 Documents")

    st.caption("Manage your knowledge base")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="document_uploader"
    )

    if uploaded_files:

        if st.button(
            "🚀 Upload Documents",
            use_container_width=True,
            type="primary"
        ):

            progress = st.progress(0)

            successful = 0

            for index, file in enumerate(uploaded_files):

                result = upload_document(file)

                if result:

                    successful += 1

                    filename = result.get(
                        "filename",
                        file.name
                    )

                    # Avoid duplicate entries
                    if not any(
                        doc["filename"] == filename
                        for doc in st.session_state.uploaded_documents
                    ):
                        st.session_state.uploaded_documents.append({
                            "filename": filename,
                            "chunks": result.get("chunks", 0)
                        })

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.success(
                f"{successful}/{len(uploaded_files)} documents uploaded."
            )

            st.rerun()

    st.markdown("---")

    st.markdown("**📚 Uploaded Documents**")

    if st.session_state.uploaded_documents:

        for index, document in enumerate(
            st.session_state.uploaded_documents
        ):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f"📄 {document['filename']}"
                )

                st.caption(
                    f"{document['chunks']} chunks"
                )

            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_doc_{index}",
                    help="Delete document"
                ):
                    delete_document(document["filename"])

    else:
        st.caption("No documents uploaded yet.")

    st.markdown("---")

    # -----------------------------
    # API STATUS
    # -----------------------------

    if check_api_health():
        st.success("🟢 Backend connected")
    else:
        st.error("🔴 Backend unavailable")

    st.caption("Powered by FastAPI, Ollama and Weaviate")


# =====================================================
# MAIN CHAT AREA
# =====================================================

st.markdown(
    '<p class="main-title">Knowledge Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Ask questions about your uploaded documents using AI.'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

if not st.session_state.messages:

    st.markdown("### 👋 Welcome to your Knowledge Assistant")

    st.write(
        "Ask anything about your uploaded PDF documents."
    )

    st.info(
        "Upload documents using the left sidebar to get started."
    )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# CHAT INPUT
# =====================================================

if prompt := st.chat_input(
    "Ask something about your documents..."
):

    # Create a new chat if necessary

    if st.session_state.current_chat_id is None:

        import uuid

        chat_id = str(uuid.uuid4())

        st.session_state.current_chat_id = chat_id

        st.session_state.chat_history.append({
            "id": chat_id,
            "title": prompt[:35],
            "messages": []
        })

    # Save user message

    user_message = {
        "role": "user",
        "content": prompt
    }

    st.session_state.messages.append(user_message)

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

    save_current_chat()
