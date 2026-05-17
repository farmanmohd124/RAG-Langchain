import streamlit as st    # the UI framework
import requests           # to call our FastAPI backend
import uuid               # to generate unique session IDs

# ── Configuration ──────────────────────────────────────────────────────────────

# URL of our FastAPI backend — change this if deploying to a server
API_URL = "http://localhost:8000/api"

# ── Page Setup ─────────────────────────────────────────────────────────────────

# must be the first Streamlit call — sets browser tab title and layout
st.set_page_config(
    page_title="YTRAG - RAG Agent",
    page_icon="🤖",
    layout="wide",              # use full browser width
)

st.title("🤖 YTRAG - RAG Agent")
st.caption("Ask questions about your documents. I can also search the web if needed.")

# ── Session State Init ─────────────────────────────────────────────────────────

# session_state persists values across Streamlit reruns (every interaction reruns the script)
# we initialize keys only if they don't exist yet to avoid resetting on every rerun

if "messages" not in st.session_state:
    st.session_state.messages = []       # list of {"role": "user"/"assistant", "content": "..."}

if "session_id" not in st.session_state:
    # generate a unique ID for this browser session so each user has separate memory
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # show the current session ID (useful for debugging)
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.divider()

    # ── New Chat Button ──
    if st.button("🗑️ New Chat", use_container_width=True):
        # call the backend to clear server-side conversation memory
        requests.post(f"{API_URL}/chat/clear", json={"session_id": st.session_state.session_id})

        # generate a new session ID for the fresh conversation
        st.session_state.session_id = str(uuid.uuid4())

        # clear the displayed chat history in the UI
        st.session_state.messages = []

        st.rerun()   # force Streamlit to rerun immediately so the UI clears

    st.divider()

    # ── Document Upload ──
    st.header("📄 Upload Documents")
    st.caption("Add PDFs or text files to the knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],          # only allow supported types
        help="Upload a PDF or TXT file to add it to the knowledge base.",
    )

    if uploaded_file is not None:
        # show a spinner while the file is being uploaded and indexed
        with st.spinner(f"Indexing '{uploaded_file.name}'..."):
            # send the file to the backend as a multipart form upload
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            )

        if response.status_code == 200:
            data = response.json()
            # show a green success message with chunk count
            st.success(f"✅ {data['message']} ({data['chunks_added']} chunks added)")
        else:
            # show a red error message with the server's error detail
            st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")

    st.divider()

    # ── Server Health Check ──
    st.header("🔌 Server Status")
    if st.button("Check Connection", use_container_width=True):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=3)   # 3 second timeout
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ Connected")
                st.info(f"LLM: `{data['llm_provider']}`\nEmbed: `{data['embed_model']}`")
            else:
                st.error("❌ Server returned an error")
        except requests.exceptions.ConnectionError:
            # server is not running or wrong URL
            st.error("❌ Cannot connect. Is the server running?")

# ── Chat History Display ────────────────────────────────────────────────────────

# render all previous messages from session_state
# this loop runs on every rerun to rebuild the full chat history visually
for message in st.session_state.messages:
    with st.chat_message(message["role"]):   # "user" shows user icon, "assistant" shows bot icon
        st.markdown(message["content"])       # render content as markdown (supports bold, code, etc.)

# ── Chat Input ─────────────────────────────────────────────────────────────────

# st.chat_input renders the input box pinned to the bottom of the page
# it returns the typed text when the user presses Enter, otherwise returns None
if prompt := st.chat_input("Ask a question about your documents..."):

    # immediately display the user's message in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    # save the user message to session_state so it persists on next rerun
    st.session_state.messages.append({"role": "user", "content": prompt})

    # call the backend and show a spinner while waiting for the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,     # agent may take up to 60s for complex multi-tool queries
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]

                    # display the agent's answer as markdown
                    st.markdown(answer)

                    # save assistant response to session_state
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                else:
                    # backend returned an error — show it clearly
                    error_msg = response.json().get("detail", "Unknown error from server.")
                    st.error(f"❌ {error_msg}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the server. Make sure `python server.py` is running.")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The agent is taking too long. Try a simpler question.")
