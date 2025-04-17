import streamlit as st
import os
from dotenv import load_dotenv
from main import VoiceAssistant, LLM_PROVIDER, WHISPER_MODEL_SIZE, COMPUTE_TYPE, DEVICE, NUM_WORKERS, get_ollama_models, get_lm_studio_models
import threading
import time
import queue

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Voice LLM Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ Voice LLM Assistant")
st.markdown("""
A voice-enabled LLM assistant with support for multiple providers and voice activity detection.
""")

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = ""
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # LLM Provider Selection
    provider = st.selectbox(
        "Select LLM Provider",
        ["ollama", "lm_studio", "openai", "claude", "claude_desktop", "google"],
        index=["ollama", "lm_studio", "openai", "claude", "claude_desktop", "google"].index(LLM_PROVIDER)
    )
    
    # API Key fields for remote providers
    if provider in ["openai", "claude", "google"]:
        api_key = st.text_input(
            f"{provider.upper()} API Key",
            type="password",
            value=os.getenv(f"{provider.upper()}_API_KEY", "")
        )
        os.environ[f"{provider.upper()}_API_KEY"] = api_key
    
    # Model-specific settings
    if provider == "ollama":
        ollama_models = get_ollama_models()
        model_name = st.selectbox(
            "Ollama Model",
            ollama_models,
            index=0 if ollama_models else None,
            help="Select a model from your local Ollama installation"
        )
    elif provider == "lm_studio":
        lm_studio_models = get_lm_studio_models()
        if lm_studio_models:
            model = st.selectbox(
                "LM Studio Model",
                lm_studio_models,
                index=0,
                help="Select a model from your local LM Studio installation"
            )
        else:
            st.warning("No models found in LM Studio. Please make sure LM Studio is running and has models loaded.")
            model = st.text_input(
                "LM Studio Model Name",
                value="",
                help="Enter the name of your local model (e.g., 'mistral-7b-instruct-v0.2.Q4_K_M.gguf')"
            )
    elif provider == "openai":
        model = st.selectbox(
            "OpenAI Model",
            [
                "gpt-4o-mini",
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-4-32k",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ],
            index=0
        )
    elif provider == "claude":
        model = st.selectbox(
            "Claude Model",
            [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-2.0"
            ],
            index=0
        )
    elif provider == "google":
        model = st.selectbox(
            "Google Model",
            [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro",
                "gemini-1.0-ultra"
            ],
            index=0
        )
    
    # Speech Recognition Settings
    st.subheader("Speech Recognition")
    whisper_model = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=["tiny", "base", "small", "medium", "large"].index(WHISPER_MODEL_SIZE)
    )
    
    compute_type = st.selectbox(
        "Compute Type",
        ["float32", "float16", "int8"],
        index=["float32", "float16", "int8"].index(COMPUTE_TYPE)
    )
    
    device = st.selectbox(
        "Device",
        ["cpu", "cuda"],
        index=["cpu", "cuda"].index(DEVICE)
    )
    
    num_workers = st.slider(
        "Number of Workers",
        min_value=1,
        max_value=8,
        value=NUM_WORKERS
    )
    
    # Voice Activity Detection Settings
    st.subheader("Voice Activity Detection")
    vad_mode = st.slider(
        "VAD Mode (0-3, higher is more aggressive)",
        min_value=0,
        max_value=3,
        value=3
    )
    
    silence_threshold = st.slider(
        "Silence Threshold (frames)",
        min_value=5,
        max_value=30,
        value=10
    )

    # Add system message configuration
    st.subheader("Assistant Behavior")
    system_message = st.text_area(
        "System Message",
        value="You are a helpful, friendly voice assistant. Keep your responses concise and conversational since they will be spoken aloud.",
        help="This message helps guide the assistant's behavior and response style."
    )

# Main content area
st.header("Voice Assistant")

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Assistant"):
        if not st.session_state.is_running:
            try:
                # Create configuration dictionary
                config = {
                    'provider': provider,
                    'whisper_model': whisper_model,
                    'compute_type': compute_type,
                    'device': device,
                    'num_workers': num_workers,
                    'vad_mode': vad_mode,
                    'silence_threshold': silence_threshold,
                    'system_message': system_message
                }
                
                if provider == "ollama":
                    config['model_name'] = model_name
                elif provider == "lm_studio":
                    config['model'] = model
                elif provider in ["openai", "claude", "google"]:
                    config['model'] = model
                    config['api_key'] = os.getenv(f"{provider.upper()}_API_KEY")
                
                # Create and store the assistant
                assistant = VoiceAssistant(config=config)
                st.session_state.assistant = assistant
                st.session_state.is_running = True
                
                # Start the assistant in a separate thread
                def run_assistant():
                    try:
                        assistant.run()
                    except Exception as e:
                        st.session_state.message_queue.put(("error", str(e)))
                
                thread = threading.Thread(target=run_assistant)
                thread.daemon = True
                thread.start()
                
                st.success("Assistant started!")
            except Exception as e:
                st.error(f"Error starting assistant: {str(e)}")
                st.session_state.debug_info = str(e)
        else:
            st.warning("Assistant is already running!")

with col2:
    if st.button("Stop Assistant"):
        if st.session_state.is_running and st.session_state.assistant:
            st.session_state.assistant.running = False
            st.session_state.is_running = False
            st.success("Assistant stopped!")
        else:
            st.warning("Assistant is not running!")

# Process any messages from the assistant thread
while not st.session_state.message_queue.empty():
    msg_type, content = st.session_state.message_queue.get()
    if msg_type == "error":
        st.error(f"Assistant error: {content}")
        st.session_state.debug_info = content

# Debug information
with st.expander("Debug Information"):
    st.text(st.session_state.debug_info)

# Conversation history
st.subheader("Conversation History")
if st.session_state.assistant and st.session_state.assistant.local_llm:
    for message in st.session_state.assistant.local_llm.history:
        if message["role"] == "user":
            st.write(f"👤 You: {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"🤖 Assistant: {message['content']}")

# Status information
st.sidebar.subheader("Status")
if st.session_state.is_running:
    st.sidebar.success("Assistant is running")
else:
    st.sidebar.info("Assistant is stopped")

# Add a refresh button to update the conversation history
if st.button("Refresh Conversation"):
    st.experimental_rerun() 