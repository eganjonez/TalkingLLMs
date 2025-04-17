"""
Advanced Voice LLM Assistant with VAD - A complete voice interface for local LLMs

This enhanced version includes:
- Voice Activity Detection for better real-time conversation
- Streaming processing when possible
- Enhanced error handling and user experience
- Support for multiple LLM providers
- Streamlit interface for configuration

Requirements:
- Python 3.8+
- GPU recommended for better performance
"""

import os
import time
import queue
import threading
import numpy as np
import torch
import wave
import webrtcvad
import pyaudio
import sounddevice as sd
import requests
from faster_whisper import WhisperModel
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# ============= Configuration =============
# Speech Recognition settings
WHISPER_MODEL_SIZE = "base"  # Options: "tiny", "base", "small", "medium", "large"
COMPUTE_TYPE = "float32"  # Use float32 for CPU compatibility
DEVICE = "cpu"  # Force CPU usage
NUM_WORKERS = 2  # Reduce workers for CPU-only system

# Text-to-Speech settings
USE_CUSTOM_VOICE = False  # Set to True if you want to use a custom voice
CUSTOM_VOICE_PATH = "path/to/your/voice_sample.wav"  # Only used if USE_CUSTOM_VOICE is True
SCOTTISH_SPEAKER_ID = "p229"  # Scottish speaker ID from VCTK dataset

# LLM Provider settings
LLM_PROVIDER = "ollama"  # Options: "claude", "openai", "lm_studio", "ollama", "claude_desktop"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")  # Get API key from environment
CLAUDE_MODEL = "claude-3-opus-20240229"  # Default Claude model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key from environment
OPENAI_MODEL = "gpt-4-turbo-preview"  # Default OpenAI model

# Local LLM settings
OLLAMA_API_URL = "http://localhost:11434"  # Base Ollama API URL
MODEL_NAME = "llama3.1:8b"  # Using llama3.1:8b model
LM_STUDIO_URL = "http://localhost:1234/v1"  # Default LM Studio API URL

# Audio recording settings
RATE = 16000
CHUNK = 480  # Must be 10, 20, or 30ms for WebRTC VAD (480 = 30ms at 16kHz)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_MODE = 3  # 0-3, 0 is the least aggressive, 3 is the most aggressive
PADDING_MS = 300  # Padding milliseconds
SILENCE_THRESHOLD = 10  # Number of consecutive silent frames to stop recording


# ============= Speech Recognition Module with VAD =============
class EnhancedSpeechRecognizer:
    def __init__(self, model_size=WHISPER_MODEL_SIZE, compute_type=COMPUTE_TYPE):
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=compute_type,
            num_workers=NUM_WORKERS,
            download_root="./models"  # Cache models locally
        )
        self.vad = webrtcvad.Vad(VAD_MODE)

    def record_audio_with_vad(self):
        """Record audio from microphone with Voice Activity Detection"""
        print("Listening... (speak to start, silence to end, or Ctrl+C to cancel)")

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        voiced_frames = []
        silent_frames = 0
        recording_started = False

        try:
            while True:
                frame = stream.read(CHUNK)
                frames.append(frame)

                # Check if the frame has voice activity
                is_speech = self.vad.is_speech(frame, RATE)

                if is_speech:
                    voiced_frames.append(frame)
                    silent_frames = 0
                    if not recording_started:
                        print("Voice detected! Recording...")
                        recording_started = True
                elif recording_started:
                    # Add frames during silence for a smoother recording
                    voiced_frames.append(frame)
                    silent_frames += 1

                    # Stop recording after consecutive silent frames
                    if silent_frames > SILENCE_THRESHOLD:
                        print("Silence detected. Processing...")
                        break

                # Auto-stop if the recording gets too long
                if len(voiced_frames) > 3000:  # ~90 seconds at 30ms per frame
                    print("Maximum recording time reached. Processing...")
                    break

        except KeyboardInterrupt:
            print("Recording canceled.")
            return None
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        # If no voice was detected, return None
        if not recording_started or len(voiced_frames) < 10:
            print("No voice detected.")
            return None

        # Save as WAV file
        filename = "temp_recording.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(voiced_frames))
        wf.close()

        return filename

    def transcribe(self, audio_file):
        """Transcribe audio file using Whisper"""
        if audio_file is None:
            return ""

        segments, info = self.model.transcribe(audio_file, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()


# ============= LLM Module =============
class BaseLLM:
    def __init__(self, system_message=None):
        self.history = [
            {"role": "system",
             "content": system_message or "You are a helpful, friendly voice assistant. Keep your responses concise and conversational since they will be spoken aloud."}
        ]

    def get_response(self, user_input):
        raise NotImplementedError("Subclasses must implement get_response")

    def reset_conversation(self):
        self.history = [
            {"role": "system",
             "content": "You are a helpful, friendly voice assistant. Keep your responses concise and conversational since they will be spoken aloud."}
        ]

class ClaudeLLM(BaseLLM):
    def __init__(self, api_key=CLAUDE_API_KEY, model=CLAUDE_MODEL, system_message=None):
        super().__init__(system_message)
        if not api_key:
            raise ValueError("Claude API key is required")
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        print(f"Using Claude with model: {model}")

    def get_response(self, user_input):
        """Get response from Claude"""
        if not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        try:
            # Format messages for Claude
            messages = []
            for msg in self.history[1:]:  # Skip system message for now
                if msg["role"] != "system":  # Claude handles system message separately
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Prepare the request
            headers = {
                "anthropic-version": "2023-06-01",
                "x-api-key": self.api_key,
                "content-type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": messages,
                "system": self.history[0]["content"],  # Add system message separately
                "max_tokens": 1000
            }

            # Make request to Claude
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # Get the response content
            response_data = response.json()
            assistant_message = response_data["content"][0]["text"]
            
            # Print and add to history
            print(f"Assistant: {assistant_message}")
            
            # Add messages to history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message

        except Exception as e:
            error_msg = f"Error communicating with Claude: {str(e)}"
            print(error_msg)
            return error_msg

class LMStudioLLM(BaseLLM):
    def __init__(self, model=None, system_message=None):
        super().__init__(system_message)
        self.api_url = "http://localhost:1234/v1"
        self.model = model
        print(f"Using LM Studio with model: {model}")

    def get_response(self, user_input):
        """Get response from LM Studio"""
        if not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        try:
            # Format messages for LM Studio
            messages = []
            for msg in self.history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Prepare the request
            headers = {
                "Content-Type": "application/json"
            }

            data = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }

            # Make request to LM Studio
            response = requests.post(f"{self.api_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            
            # Get the response content
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"].strip()
            
            # Print and add to history
            print(f"Assistant: {assistant_message}")
            
            # Add messages to history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message

        except Exception as e:
            error_msg = f"Error communicating with LM Studio: {str(e)}"
            print(error_msg)
            return error_msg

class OllamaLLM(BaseLLM):
    def __init__(self, model_name=MODEL_NAME, system_message=None):
        super().__init__(system_message)
        print(f"Using Ollama with model: {model_name}")
        self.model_name = model_name
        self.api_url = f"{OLLAMA_API_URL}/api/generate"

    def get_response(self, user_input):
        """Get response from LLM using Ollama"""
        if not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        try:
            # Format the prompt with conversation history
            prompt = ""
            for msg in self.history:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"{content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            # Add the current user input
            prompt += f"User: {user_input}\nAssistant:"

            # Prepare the request
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            # Make request to Ollama
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            
            # Get the response content
            response_data = response.json()
            assistant_message = response_data.get("response", "").strip()
            
            # Print and add to history
            print(f"Assistant: {assistant_message}")
            
            # Add messages to history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message

        except Exception as e:
            error_msg = f"Error communicating with Ollama: {str(e)}"
            print(error_msg)
            return error_msg

class OpenAILLM(BaseLLM):
    def __init__(self, api_key=OPENAI_API_KEY, model=OPENAI_MODEL, system_message=None):
        super().__init__(system_message)
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        print(f"Using OpenAI with model: {model}")

    def get_response(self, user_input):
        """Get response from OpenAI"""
        if not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        try:
            # Format messages for OpenAI
            messages = []
            for msg in self.history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }

            # Make request to OpenAI
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # Get the response content
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"].strip()
            
            # Print and add to history
            print(f"Assistant: {assistant_message}")
            
            # Add messages to history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message

        except Exception as e:
            error_msg = f"Error communicating with OpenAI: {str(e)}"
            print(error_msg)
            return error_msg

class ClaudeDesktopLLM(BaseLLM):
    def __init__(self, system_message=None):
        super().__init__(system_message)
        print("Using Claude Desktop (local Claude on Mac)")
        # Add implementation for Claude Desktop

    def get_response(self, user_input):
        """Get response from Claude Desktop"""
        # Add implementation for Claude Desktop
        pass

class GoogleLLM(BaseLLM):
    def __init__(self, api_key=None, model="gemini-1.5-pro", system_message=None):
        super().__init__(system_message)
        if not api_key:
            raise ValueError("Google API key is required")
        self.api_key = api_key
        self.model = model
        print(f"Using Google with model: {model}")

    def get_response(self, user_input):
        """Get response from Google's Gemini"""
        if not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Format messages for Gemini
            messages = []
            for msg in self.history:
                if msg["role"] == "system":
                    messages.append({"role": "user", "parts": [msg["content"]]})
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    messages.append({"role": role, "parts": [msg["content"]]})
            
            # Add current user message
            messages.append({"role": "user", "parts": [user_input]})

            # Create the model
            model = genai.GenerativeModel(self.model)
            
            # Start a chat
            chat = model.start_chat(history=messages[:-1])
            
            # Get response
            response = chat.send_message(user_input)
            assistant_message = response.text
            
            # Print and add to history
            print(f"Assistant: {assistant_message}")
            
            # Add messages to history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message

        except Exception as e:
            error_msg = f"Error communicating with Google: {str(e)}"
            print(error_msg)
            return error_msg


# ============= Text-to-Speech Module =============
class StreamingTextToSpeech:
    def __init__(self, use_custom_voice=USE_CUSTOM_VOICE, custom_voice_path=CUSTOM_VOICE_PATH):
        # Initialize TTS
        print("Loading TTS model...")
        try:
            # Using VCTK model for Scottish voice support
            self.tts = TTS("tts_models/en/vctk/vits")
            self.scottish_speaker = SCOTTISH_SPEAKER_ID
            self.use_custom_voice = use_custom_voice
            self.custom_voice_path = custom_voice_path
            self.is_playing = False
            print("TTS model loaded successfully!")
        except Exception as e:
            print(f"Error loading TTS model: {str(e)}")
            print("Speech synthesis will be disabled.")
            self.tts = None

    def speak(self, text):
        """Convert text to speech"""
        if not text.strip() or self.tts is None:
            return

        output_file = "response.wav"

        try:
            # Generate speech with Scottish voice
            self.tts.tts_to_file(
                text=text,
                file_path=output_file,
                speaker=self.scottish_speaker  # Specify Scottish speaker
            )

            # Play the audio
            self.play_audio(output_file)

            # Clean up the file
            try:
                os.remove(output_file)
            except:
                pass

        except Exception as e:
            print(f"TTS Error: {str(e)}")
            print("Falling back to text output only.")

    def play_audio(self, file_path):
        """Play audio file"""
        try:
            print(f"Attempting to play audio file: {file_path}")
            import soundfile as sf
            data, fs = sf.read(file_path)
            print(f"Audio file loaded. Sample rate: {fs}Hz, Shape: {data.shape}")
            
            # Ensure data is in the correct format for sounddevice
            if data.dtype != np.float32:
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                else:
                    data = data.astype(np.float32)
            print(f"Audio data converted to float32. Range: [{np.min(data)}, {np.max(data)}]")
            
            # Ensure data is in the correct range [-1, 1]
            if np.max(np.abs(data)) > 1.0:
                data = data / np.max(np.abs(data))
            
            # Ensure data is 2D if it's stereo
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            print(f"Final audio shape: {data.shape}")
            
            # Play the audio
            print("Starting audio playback...")
            self.is_playing = True
            sd.play(data, fs)
            
            # Wait for playback to complete or until interrupted
            while self.is_playing and sd.get_stream().active:
                time.sleep(0.1)
            
            print("Audio playback completed.")
            
        except Exception as e:
            print(f"Audio playback error: {str(e)}")
            print("Falling back to text output only.")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
        finally:
            self.is_playing = False
            # Clean up the file
            try:
                os.remove(file_path)
            except:
                pass

    def stop(self):
        """Stop current audio playback"""
        if self.is_playing:
            print("Stopping audio playback...")
            sd.stop()
            self.is_playing = False


# ============= Main Application =============
class VoiceAssistant:
    def __init__(self, config=None):
        print("Initializing Voice LLM Assistant...")
        
        # Apply configuration if provided
        if config:
            print(f"Applying configuration: {config}")
            global WHISPER_MODEL_SIZE, COMPUTE_TYPE, DEVICE, NUM_WORKERS, VAD_MODE, SILENCE_THRESHOLD, LLM_PROVIDER, MODEL_NAME, LM_STUDIO_URL
            
            # Update global variables
            WHISPER_MODEL_SIZE = config.get('whisper_model', WHISPER_MODEL_SIZE)
            COMPUTE_TYPE = config.get('compute_type', COMPUTE_TYPE)
            DEVICE = config.get('device', DEVICE)
            NUM_WORKERS = config.get('num_workers', NUM_WORKERS)
            VAD_MODE = config.get('vad_mode', VAD_MODE)
            SILENCE_THRESHOLD = config.get('silence_threshold', SILENCE_THRESHOLD)
            
            # Update LLM provider settings
            LLM_PROVIDER = config.get('provider', LLM_PROVIDER)
            if LLM_PROVIDER == "ollama":
                MODEL_NAME = config.get('model_name', MODEL_NAME)
            elif LLM_PROVIDER == "lm_studio":
                LM_STUDIO_URL = "http://localhost:1234/v1"  # Default URL
                MODEL_NAME = config.get('model', MODEL_NAME)
            elif LLM_PROVIDER in ["openai", "claude", "google"]:
                if LLM_PROVIDER == "openai":
                    global OPENAI_MODEL
                    OPENAI_MODEL = config.get('model', OPENAI_MODEL)
                elif LLM_PROVIDER == "claude":
                    global CLAUDE_MODEL
                    CLAUDE_MODEL = config.get('model', CLAUDE_MODEL)
                elif LLM_PROVIDER == "google":
                    global GOOGLE_MODEL
                    GOOGLE_MODEL = config.get('model', "gemini-1.5-pro")

        # Check for GPU
        if torch.cuda.is_available():
            print("GPU detected! Using CUDA for faster processing.")
        else:
            print("No GPU detected. Processing might be slower.")

        self.speech_recognizer = EnhancedSpeechRecognizer()
        
        # Initialize LLM based on provider
        print(f"Initializing LLM provider: {LLM_PROVIDER}")
        if LLM_PROVIDER == "claude":
            self.local_llm = ClaudeLLM(api_key=config.get('api_key') if config else None, system_message=config.get('system_message') if config else None)
        elif LLM_PROVIDER == "openai":
            self.local_llm = OpenAILLM(api_key=config.get('api_key') if config else None, system_message=config.get('system_message') if config else None)
        elif LLM_PROVIDER == "lm_studio":
            self.local_llm = LMStudioLLM(model=config.get('model') if config else None, system_message=config.get('system_message') if config else None)
        elif LLM_PROVIDER == "claude_desktop":
            self.local_llm = ClaudeDesktopLLM(system_message=config.get('system_message') if config else None)
        elif LLM_PROVIDER == "google":
            self.local_llm = GoogleLLM(api_key=config.get('api_key') if config else None, system_message=config.get('system_message') if config else None)
        else:  # Default to Ollama
            self.local_llm = OllamaLLM(system_message=config.get('system_message') if config else None)
            
        self.text_to_speech = StreamingTextToSpeech()

        # Define command handlers
        self.commands = {
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
            "goodbye": self.cmd_exit,
            "reset": self.cmd_reset,
            "clear": self.cmd_reset,
            "stop": self.cmd_stop,
            "halt": self.cmd_stop,
            "pause": self.cmd_stop
        }

        self.running = True
        print("Voice Assistant initialized successfully!")

    def cmd_exit(self):
        """Handle exit command"""
        self.text_to_speech.speak("Goodbye! Have a great day.")
        self.running = False
        return True

    def cmd_reset(self):
        """Handle reset command"""
        self.local_llm.reset_conversation()
        self.text_to_speech.speak("Conversation history has been reset.")
        return True

    def cmd_stop(self):
        """Handle stop command"""
        self.text_to_speech.stop()
        return True

    def run(self):
        """Main conversation loop"""
        print("\nAll models loaded! Starting conversation...")
        self.text_to_speech.speak("Hello! I'm your voice assistant. How can I help you today?")

        while self.running:
            try:
                # Record and transcribe user's speech
                print("Listening for speech...")
                audio_file = self.speech_recognizer.record_audio_with_vad()
                if not audio_file:
                    print("No audio file recorded, continuing...")
                    continue
                    
                print("Transcribing speech...")
                user_text = self.speech_recognizer.transcribe(audio_file)

                if not user_text:
                    print("No text transcribed, continuing...")
                    continue

                print(f"You said: {user_text}")

                # Check for commands
                command_executed = False
                for command, handler in self.commands.items():
                    if command in user_text.lower():
                        print(f"Executing command: {command}")
                        command_executed = handler()
                        break

                if command_executed:
                    continue

                # Get response from LLM
                print("Getting response from LLM...")
                response = self.local_llm.get_response(user_text)
                print(f"LLM Response: {response}")

                # Convert response to speech
                print("Converting response to speech...")
                self.text_to_speech.speak(response)

                # Clean up temp file
                if audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)

            except KeyboardInterrupt:
                print("\nExiting Voice LLM Assistant...")
                break
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")
                self.text_to_speech.speak("I encountered an error. Let's try again.")
                raise  # Re-raise the exception to be caught by the Streamlit thread

    def get_ollama_models(self):
        """Fetch available models from Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
        return ["llama3.1:8b"]  # Default fallback

    def get_lm_studio_models(self):
        """Fetch available models from LM Studio"""
        try:
            response = requests.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
        except Exception as e:
            print(f"Error fetching LM Studio models: {e}")
        return []  # Return empty list if no models found


def get_ollama_models():
    """Fetch available models from Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
    return ["llama3.1:8b"]  # Default fallback

def get_lm_studio_models():
    """Fetch available models from LM Studio"""
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            return [model["id"] for model in models]
    except Exception as e:
        print(f"Error fetching LM Studio models: {e}")
    return []  # Return empty list if no models found


if __name__ == "__main__":
    # If running directly (not through Streamlit), start the assistant
    assistant = VoiceAssistant()
    assistant.run()