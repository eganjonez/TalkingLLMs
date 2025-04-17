# Voice LLM Assistant

A voice-enabled LLM assistant with a Streamlit interface for easy model selection and configuration.

## Features

- Voice Activity Detection for natural conversation
- Support for multiple LLM providers:
  - Ollama (local models)
  - LM Studio (local models)
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Claude Desktop (local Claude on Mac)
- Streamlit UI for easy configuration
- Text-to-speech capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-llm-assistant.git
cd voice-llm-assistant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
```

## Usage

### Running with Streamlit UI

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Configure your settings in the sidebar:
   - Select your preferred LLM provider
   - Configure speech recognition settings
   - Adjust voice activity detection parameters

3. Click "Start Assistant" to begin the conversation

### Running Directly

You can also run the assistant directly without the Streamlit interface:

```bash
python main.py
```

## Configuration

### LLM Providers

- **Ollama**: Local models running through Ollama
- **LM Studio**: Local models running through LM Studio
- **OpenAI**: GPT models (requires API key)
- **Claude**: Anthropic's Claude models (requires API key)
- **Claude Desktop**: Local Claude on Mac

### Speech Recognition

- Whisper model size: tiny, base, small, medium, large
- Compute type: float32, float16, int8
- Device: CPU or CUDA
- Number of workers: 1-8

### Voice Activity Detection

- VAD Mode: 0-3 (higher is more aggressive)
- Silence Threshold: 5-30 frames

## Requirements

- Python 3.8+
- GPU recommended for better performance
- Microphone for voice input
- Speakers for voice output

## License

MIT License 