import random
import numpy as np
import torch

# Try to import multilingual TTS only
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    MULTILINGUAL_AVAILABLE = True
    print("🌍 Multilingual TTS support detected")
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    print("❌ Multilingual TTS not available. Please install latest chatterbox package.")

# Try to import NEMO ASR for Parakeet
try:
    import nemo.collections.asr as nemo_asr
    PARAKEET_AVAILABLE = True
    print("🎤 Parakeet ASR support detected")
except (ImportError, AttributeError) as e:
    PARAKEET_AVAILABLE = False
    print("❌ Parakeet ASR not available. Will auto-install when needed.")
    print("   Install manually with: pip install nemo_toolkit[asr]")

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("🧠 Gemini AI support detected")
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Gemini AI not available. Install with: pip install google-generativeai")

import gradio as gr
import os
import subprocess
import sys
import warnings
import re
import json
import io
from datetime import datetime
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import tempfile
import shutil
# Add new imports for advanced audio processing
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft, ifft, fftfreq
import librosa
import soundfile as sf
import base64
import requests
from pathlib import Path
from urllib.parse import urlparse
import threading
from typing import List, Dict, Optional
import pandas as pd
import math
import json

# Windows signal compatibility fix
import signal
import platform
if platform.system() == "Windows":
    # Add missing signals for Windows compatibility
    if not hasattr(signal, 'SIGKILL'):
        signal.SIGKILL = 9
    if not hasattr(signal, 'SIGTERM'):
        signal.SIGTERM = 15
    if not hasattr(signal, 'SIGHUP'):
        signal.SIGHUP = 1

# Suppress the specific LoRACompatibleLinear deprecation warning
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*deprecated.*", category=FutureWarning)

# Suppress torch CUDA sdp_kernel deprecation warning
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*deprecated.*", category=FutureWarning)

# Suppress LlamaModel attention implementation warning
warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)

# Suppress past_key_values tuple deprecation warning
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*deprecated.*", category=UserWarning)

# Suppress additional transformers warnings
warnings.filterwarnings("ignore", message=".*LlamaModel.*LlamaSdpaAttention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*We detected that you are passing.*past_key_values.*", category=UserWarning)

# Suppress Gradio audio conversion warning
warnings.filterwarnings("ignore", message=".*Trying to convert audio automatically.*", category=UserWarning)

# More aggressive warning suppression for transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# Suppress all warnings containing these key phrases
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*LlamaModel.*")
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*")

# Suppress torch/contextlib warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*contextlib.*")

# Suppress torch.load warnings related to TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*weights_only.*argument.*not explicitly passed.*")
warnings.filterwarnings("ignore", message=".*forcing weights_only=False.*")

# Suppress checkpoint manager warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*checkpoint_manager.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*perth.*")

# Suppress chatterbox TTS model warnings
warnings.filterwarnings("ignore", message=".*Detected.*repetition of token.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*forcing EOS token.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=".*chatterbox.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*alignment_stream_analyzer.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MULTILINGUAL_MODEL = None

# Supported languages for multilingual model
SUPPORTED_LANGUAGES = {
    'ar': 'Arabic',
    'da': 'Danish', 
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'fi': 'Finnish',
    'fr': 'French',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ms': 'Malay',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'tr': 'Turkish',
    'zh': 'Chinese'
}

# --- Video Dubbing System Classes ---

# Global dubbing system instances
PARAKEET_MODEL = None
API_MANAGER = None
GEMINI_TRANSLATOR = None
AUDIO_PROCESSOR = None

# Gemini model options for fallback - FREE MODELS FIRST
GEMINI_MODELS = [
    "gemini-1.5-flash", "gemini-1.5-flash-002", "gemini-1.5-flash-001",
    "gemini-1.5-pro", "gemini-1.5-pro-002", "gemini-1.5-pro-001", 
    "gemini-2.0-flash-exp", "gemini-2.0-flash-001", "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-flash-preview-04-17", 
    "gemini-2.5-flash-lite-preview-06-17", "gemini-2.0-pro"
]

# --- Enhanced Audio Processing Functions ---

def get_audio_duration(file_path):
    """Get the duration of an audio file using ffprobe"""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except (subprocess.SubprocessError, ValueError):
        return None

def split_audio_file(file_path, chunk_duration=600, progress=None):
    """Split audio into chunks of specified duration (in seconds)"""
    temp_dir = tempfile.mkdtemp()
    duration = get_audio_duration(file_path)
    if not duration:
        return None, 0
    
    num_chunks = math.ceil(duration / chunk_duration)
    chunk_files = []
    
    for i in range(num_chunks):
        if progress is not None:
            progress(i/num_chunks * 0.2, desc=f"Splitting audio ({i+1}/{num_chunks})...")
        
        start_time = i * chunk_duration
        output_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        
        cmd = ['ffmpeg', '-i', file_path, '-ss', str(start_time), '-t', str(chunk_duration),
               '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_file, '-y']
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk_files.append(output_file)
        except subprocess.CalledProcessError:
            continue
    
    return chunk_files, duration

def enhanced_extract_audio_from_video(video_path, progress=None):
    """Enhanced audio extraction with progress tracking"""
    if progress is None:
        progress = lambda x, desc=None: None
    
    progress(0.1, desc="Extracting audio from video...")
    
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_path = temp_audio.name
    temp_audio.close()
    
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
           '-ar', '16000', '-ac', '1', audio_path, '-y']
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress(0.2, desc="Audio extraction complete")
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

class ParakeetTranscriber:
    """NVIDIA Parakeet TDT model for transcription with timestamps"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        
    def auto_install_parakeet(self):
        """Auto-install Parakeet if not available"""
        try:
            print("🔧 Auto-installing Parakeet ASR...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "nemo_toolkit[asr]>=1.20.0", "--quiet"
            ])
            print("✅ Parakeet ASR installed successfully!")
            
            # Try to import again
            global PARAKEET_AVAILABLE
            try:
                import nemo.collections.asr as nemo_asr
                PARAKEET_AVAILABLE = True
                print("🎤 Parakeet ASR now available")
                return True
            except ImportError:
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Parakeet ASR: {e}")
            return False
    
    def load_model(self, progress_callback=None):
        """Load NVIDIA Parakeet TDT model with auto-install"""
        # Check if Parakeet is available, if not try to install
        global PARAKEET_AVAILABLE
        if not PARAKEET_AVAILABLE:
            if progress_callback:
                progress_callback("🔧 Installing Parakeet ASR...")
            
            if not self.auto_install_parakeet():
                raise Exception("❌ Failed to install Parakeet ASR! Please install manually: pip install nemo_toolkit[asr]")
            
            # Update global flag after successful installation
            PARAKEET_AVAILABLE = True
        
        try:
            # Dynamic import after potential installation
            try:
                import nemo.collections.asr as nemo_asr
                print("🎤 NEMO ASR imported successfully")
            except ImportError as e:
                raise Exception(f"❌ Failed to import NEMO ASR after installation: {e}")
            
            if progress_callback:
                progress_callback("🎤 Loading Parakeet TDT model...")
            
            print("🎤 Loading ASR model...")
            
            # Use model cache directory similar to chatterbox models
            model_cache_dir = os.path.join("models", "parakeet")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v2"
            )
            print("✅ Parakeet TDT model loaded successfully!")
            
            if progress_callback:
                progress_callback("✅ Parakeet model ready")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ CRITICAL: Failed to load Parakeet model: {e}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            raise Exception(error_msg)
    
    def unload_model(self):
        """Unload model to free memory for Chatterbox"""
        if self.model is not None:
            print("🗑️ Unloading Parakeet model to free memory...")
            del self.model
            self.model = None
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Parakeet model unloaded")
    
    def transcribe_with_timestamps(self, audio_path: str, progress=None, is_music=False) -> List[Dict]:
        """Transcribe audio with segment-level timestamps using Parakeet TDT ONLY"""
        if not PARAKEET_AVAILABLE or self.model is None:
            raise Exception("❌ Parakeet TDT model is REQUIRED! No fallback available.")
        
        try:
            # Check if audio is long and needs splitting
            duration = get_audio_duration(audio_path)
            long_audio_threshold = 600  # 10 minutes
            
            print(f"🎵 Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            if duration and duration > long_audio_threshold:
                print(f"📦 Audio is long ({duration/60:.1f} min), splitting into chunks...")
                return self._process_long_audio(audio_path, is_music, progress)
            else:
                print("🎵 Processing audio as single chunk...")
                return self._process_audio_chunk(audio_path, is_music, progress, 0, 1.0)
                
        except Exception as e:
            raise Exception(f"❌ CRITICAL Parakeet transcription error: {e}")
    
    def _process_long_audio(self, audio_path, is_music, progress):
        """Process long audio by splitting into chunks"""
        if progress:
            progress(0.1, desc="Analyzing audio file...")
        
        chunk_files, total_duration = split_audio_file(audio_path, 600, progress)
        if not chunk_files:
            return []
        
        all_segments = []
        
        for i, chunk_file in enumerate(chunk_files):
            chunk_start_time = i * 600
            progress_start = 0.2 + (i / len(chunk_files)) * 0.8
            
            if progress:
                progress(progress_start, desc=f"Processing chunk {i+1}/{len(chunk_files)}...")
            
            chunk_segments = self._process_audio_chunk(
                chunk_file, is_music, progress, chunk_start_time, 0.8/len(chunk_files)
            )
            all_segments.extend(chunk_segments)
            
            # Clean up chunk file
            try:
                os.unlink(chunk_file)
            except:
                pass
        
        return all_segments
    
    def _process_audio_chunk(self, audio_path, is_music, progress, time_offset=0, progress_scale=1.0):
        """Process a single audio chunk"""
        if progress:
            progress(0.3 * progress_scale, desc="Transcribing audio...")
        
        try:
            output = self.model.transcribe([audio_path], timestamps=True)
            segments = []
            
            print(f"🔍 Debug: Transcription output type: {type(output)}")
            print(f"🔍 Debug: Output length: {len(output) if output else 'None'}")

            transcriptions = None
            # FIX: Add a check for the 'list' type returned by the model
            if isinstance(output, list) and len(output) > 0:
                transcriptions = output
            # Keep the original check for tuple as a fallback
            elif isinstance(output, tuple) and len(output) >= 2:
                transcriptions = output[0]
            else:
                print("🔍 Debug: Output is not in a recognized list or tuple format.")

            # Process the transcriptions if they were successfully extracted
            if transcriptions:
                first_result = transcriptions[0]
                print(f"🔍 Debug: First result type: {type(first_result)}")
                
                if hasattr(first_result, 'timestep'):
                    timestep_data = first_result.timestep
                    
                    if 'segment' in timestep_data:
                        segment_timestamps = timestep_data['segment']
                        print(f"🔍 Debug: Found {len(segment_timestamps)} segment timestamps")
                        
                        for i, stamp in enumerate(segment_timestamps):
                            segment_text = stamp['segment']
                            start_time = stamp['start'] + time_offset
                            end_time = stamp['end'] + time_offset
                            
                            if is_music:
                                end_time += 0.3
                                min_duration = 0.5
                                if end_time - start_time < min_duration:
                                    end_time = start_time + min_duration

                            segments.append({
                                "text": segment_text,
                                "start": start_time,
                                "end": end_time,
                                "duration": end_time - start_time
                            })
                elif hasattr(first_result, 'text'):
                    print(f"🔍 Debug: Using text without timestamps: '{first_result.text}'")
                    segments.append({
                        "text": first_result.text,
                        "start": 0.0 + time_offset,
                        "end": 10.0 + time_offset,
                        "duration": 10.0
                    })
            
            print(f"🔍 Debug: Returning {len(segments)} segments")
            return segments
            
        except Exception as e:
            print(f"❌ Chunk transcription error: {e}")
            return []
    


class APIManager:
    """Manage multiple API keys with round-robin and failure handling"""
    
    def __init__(self):
        self.apis = []
        self.current_api_index = 0
        
    def add_api(self, api_config: Dict):
        """Add API configuration"""
        self.apis.append(api_config)
        print(f"✅ Added API: {api_config.get('name', 'Unnamed')}")
        
    def get_next_api(self):
        """Get next available API (round-robin)"""
        if not self.apis:
            return None
        api = self.apis[self.current_api_index]
        self.current_api_index = (self.current_api_index + 1) % len(self.apis)
        return api
        
    def remove_failed_api(self, api_config):
        """Remove failed API from rotation"""
        if api_config in self.apis:
            self.apis.remove(api_config)
            print(f"⚠️ Removed failed API: {api_config.get('name', 'Unnamed')}")
            
    def get_api_count(self):
        """Get number of available APIs"""
        return len(self.apis)

class GeminiTranslator:
    """Gemini-based translation with custom prompts and multi-language support"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        
    def translate_segments(self, segments: List[Dict], target_languages: List[str], 
                         custom_prompt: str = "", progress_callback=None) -> Dict[str, List[Dict]]:
        """Translate segments using smart chunking (20k-30k characters per chunk)"""
        if not GEMINI_AVAILABLE:
            raise Exception("Gemini AI not available. Install with: pip install google-generativeai")
            
        translations = {}
        
        for language in target_languages:
            print(f"🌍 Translating to {SUPPORTED_LANGUAGES.get(language, language)}...")
            if progress_callback:
                progress_callback(f"🌍 Starting translation to {SUPPORTED_LANGUAGES.get(language, language)}")
            
            # Create smart chunks
            chunks = self._create_smart_chunks(segments)
            print(f"📦 Created {len(chunks)} smart chunks for translation")
            
            translations[language] = []
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    if progress_callback:
                        progress_callback(f"🌍 Translating chunk {chunk_idx + 1}/{len(chunks)} to {language}")
                    
                    # Translate entire chunk at once
                    translated_chunk = self._translate_chunk(
                        chunk, 
                        target_language=SUPPORTED_LANGUAGES.get(language, language),
                        custom_prompt=custom_prompt
                    )
                    
                    translations[language].extend(translated_chunk)
                    print(f"✅ Chunk {chunk_idx + 1}/{len(chunks)} translated to {language}")
                    
                except Exception as e:
                    print(f"❌ Translation error for chunk {chunk_idx}: {e}")
                    # Keep original text if translation fails
                    for segment in chunk:
                        translations[language].append({
                            "text": segment['text'],
                            "start": segment['start'],
                            "end": segment['end'],
                            "duration": segment['duration'],
                            "segment_id": segment.get('segment_id', 0)
                        })
        
        return translations
    
    def _create_smart_chunks(self, segments: List[Dict], max_chars=25000) -> List[List[Dict]]:
        """Create smart chunks of segments based on character count"""
        chunks = []
        current_chunk = []
        current_chars = 0
        
        for i, segment in enumerate(segments):
            segment['segment_id'] = i  # Add segment ID
            segment_chars = len(segment['text'])
            
            # If adding this segment would exceed limit, start new chunk
            if current_chars + segment_chars > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_chars = segment_chars
            else:
                current_chunk.append(segment)
                current_chars += segment_chars
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _translate_chunk(self, chunk: List[Dict], target_language: str, custom_prompt: str = "") -> List[Dict]:
        """Translate a chunk of segments as a batch"""
        # Create JSON structure for batch translation
        chunk_data = {
            "segments": [
                {
                    "id": seg['segment_id'],
                    "text": seg['text'],
                    "start": seg['start'],
                    "end": seg['end'],
                    "duration": seg['duration']
                }
                for seg in chunk
            ]
        }
        
        # Create translation prompt
        prompt = f"""Translate the following JSON segments to {target_language}.
Maintain the exact JSON structure and only translate the "text" field.
Keep all timing information unchanged.
{f"Style instructions: {custom_prompt}" if custom_prompt else ""}

Input JSON:
{json.dumps(chunk_data, indent=2)}

Return ONLY the translated JSON with the same structure:"""

        # Translate using API with model fallback
        translated_json = self._translate_text_batch(prompt, target_language)
        
        try:
            # FIX: Use a more robust regex to find the JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', translated_json, re.DOTALL)
            
            if not json_match:
                # If no JSON is found at all, raise an error
                raise ValueError("No valid JSON object found in the Gemini API response.")
            
            json_string = json_match.group(0)
            
            # Now, try to parse the extracted JSON string
            translated_data = json.loads(json_string)
            
            result = []
            for seg_data in translated_data.get('segments', []):
                result.append({
                    "text": seg_data['text'],
                    "start": seg_data['start'],
                    "end": seg_data['end'],
                    "duration": seg_data['duration'],
                    "segment_id": seg_data['id']
                })
            
            if not result:
                raise ValueError("JSON was parsed, but no segments were found.")

            print(f"✅ Successfully translated and parsed {len(result)} segments")
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            # If any part of the process fails, raise an exception to stop the dubbing.
            error_message = f"❌ CRITICAL: Failed to parse translation from Gemini. Error: {e}"
            print(error_message)
            print(f"   Raw response from API was: {translated_json[:500]}...")
            # This will stop the process and display the error in the UI
            raise Exception(error_message)
        
    def _translate_text_batch(self, prompt: str, target_language: str) -> str:
        """Translate batch of segments using NEW Gemini API"""
        last_error = None
        
        # Try different APIs in round-robin
        for api_attempt in range(self.api_manager.get_api_count()):
            api_config = self.api_manager.get_next_api()
            if not api_config:
                break
            
            # Try different models with this API using NEW API format
            for model_name in GEMINI_MODELS:
                try:
                    # Use the new Gemini API format from gemini.md
                    from google import genai as new_genai
                    
                    client = new_genai.Client(api_key=api_config['api_key'])
                    
                    print(f"🤖 Using {model_name} with API {api_config.get('name', 'Unnamed')}")
                    
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    
                    return response.text.strip()
                    
                except Exception as e:
                    # Fallback to old API format
                    try:
                        genai.configure(api_key=api_config['api_key'])
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        return response.text.strip()
                    except Exception as e2:
                        last_error = e2
                        print(f"❌ Model {model_name} failed: {e2}")
                        continue
            
            print(f"⚠️ All models failed with API {api_config.get('name', 'Unnamed')}")
        
        raise Exception(f"All APIs and models failed. Last error: {last_error}")
    
    def _translate_text(self, text: str, target_language: str, custom_prompt: str = "") -> str:
        """Translate single text segment with model fallback"""
        api_config = self.api_manager.get_next_api()
        if not api_config:
            raise Exception("No available APIs")
        
        # Try different models in order of preference
        for model_name in GEMINI_MODELS:
            try:
                genai.configure(api_key=api_config['api_key'])
                model = genai.GenerativeModel(model_name)
                
                # Build translation prompt
                base_prompt = f"""Translate the following text to {target_language}.
Keep the translation natural and contextually appropriate.
Maintain the same tone and style as the original.
Only return the translated text, nothing else."""

                if custom_prompt.strip():
                    base_prompt += f"\n\nAdditional style instructions: {custom_prompt}"
                    
                base_prompt += f"\n\nText: {text}"
                
                response = model.generate_content(base_prompt)
                return response.text.strip()
                
            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                continue
        
        # If all models failed with this API, try next API
        self.api_manager.remove_failed_api(api_config)
        if self.api_manager.get_api_count() > 0:
            return self._translate_text(text, target_language, custom_prompt)
        else:
            raise Exception("All APIs and models failed")

class AudioProcessor:
    """Audio processing for speed adjustment and segment combination"""
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video using FFmpeg"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio_path = temp_audio.name
            temp_audio.close()
            
            cmd = [
                'ffmpeg', '-i', video_path, '-vn',
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                audio_path, '-y'
            ]
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Extracted audio to: {audio_path}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            return None
        except Exception as e:
            print(f"❌ Audio extraction error: {e}")
            return None
    
    def adjust_audio_speed(self, audio_path: str, target_duration: float) -> str:
        """Adjust audio speed to match target duration with high quality processing"""
        try:
            # Load audio at high quality
            audio, sr = librosa.load(audio_path, sr=None)  # Keep original sample rate
            current_duration = len(audio) / sr
            
            print(f"🔧 Adjusting audio: {current_duration:.2f}s → {target_duration:.2f}s")
            
            # Skip adjustment if durations are very close (within 0.1s)
            if abs(current_duration - target_duration) < 0.1:
                print("⏭️ Duration close enough, skipping adjustment")
                return audio_path
            
            # Remove silence from beginning and end
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # Calculate speed ratio
            speed_ratio = current_duration / target_duration
            
            # Limit speed ratio to reasonable bounds (0.5x to 2.0x)
            speed_ratio = max(0.5, min(2.0, speed_ratio))
            
            print(f"⚡ Speed ratio: {speed_ratio:.2f}x")
            
            # High-quality speed adjustment using phase vocoder
            if speed_ratio != 1.0:
                adjusted_audio = librosa.effects.time_stretch(audio_trimmed, rate=speed_ratio)
            else:
                adjusted_audio = audio_trimmed
            
            # Normalize audio to prevent clipping
            adjusted_audio = librosa.util.normalize(adjusted_audio)
            
            # Apply gentle fade in/out to prevent clicks
            fade_samples = int(0.01 * sr)  # 10ms fade
            if len(adjusted_audio) > 2 * fade_samples:
                # Fade in
                adjusted_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out
                adjusted_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Save adjusted audio at high quality
            output_path = audio_path.replace('.wav', '_adjusted.wav')
            sf.write(output_path, adjusted_audio, sr, subtype='PCM_24')  # 24-bit quality
            
            # Verify the adjustment
            final_duration = len(adjusted_audio) / sr
            print(f"✅ Speed adjusted: {final_duration:.2f}s (target: {target_duration:.2f}s)")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Speed adjustment error: {e}")
            import traceback
            traceback.print_exc()
            return audio_path
    
    def combine_audio_segments(self, audio_segments: List[str], output_path: str) -> str:
        """Combine multiple audio segments into single file"""
        try:
            combined_audio = []
            sr = None
            
            for segment_path in audio_segments:
                audio, current_sr = librosa.load(segment_path)
                if sr is None:
                    sr = current_sr
                combined_audio.extend(audio)
            
            sf.write(output_path, combined_audio, sr)
            print(f"✅ Combined audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Audio combination error: {e}")
            return None

def combine_video_with_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """Combine video with new audio track"""
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,  # Input video
            '-i', audio_path,  # Input audio
            '-c:v', 'copy',    # Copy video stream
            '-c:a', 'aac',     # Encode audio as AAC
            '-map', '0:v:0',   # Map video from first input
            '-map', '1:a:0',   # Map audio from second input
            '-shortest',       # End when shortest stream ends
            output_path,
            '-y'              # Overwrite if exists
        ]
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Combined video saved to: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        return None
    except Exception as e:
        print(f"❌ Video combination error: {e}")
        return None

def initialize_dubbing_system():
    """Initialize global dubbing system components"""
    global PARAKEET_MODEL, API_MANAGER, GEMINI_TRANSLATOR, AUDIO_PROCESSOR
    
    if API_MANAGER is None:
        API_MANAGER = APIManager()
    
    if PARAKEET_MODEL is None:
        PARAKEET_MODEL = ParakeetTranscriber()
    
    if GEMINI_TRANSLATOR is None:
        GEMINI_TRANSLATOR = GeminiTranslator(API_MANAGER)
    
    if AUDIO_PROCESSOR is None:
        AUDIO_PROCESSOR = AudioProcessor()

# --- Dubbing System Functions ---

def add_gemini_api(api_name: str, api_key: str):
    """Add a new Gemini API key to the manager"""
    initialize_dubbing_system()
    
    if not api_key.strip():
        return "❌ Please enter a valid API key", gr.update()
    
    api_config = {
        'name': api_name or f"API_{len(API_MANAGER.apis) + 1}",
        'api_key': api_key.strip(),
        'type': 'gemini'
    }
    
    API_MANAGER.add_api(api_config)
    
    # Return updated status and clear the input
    status = f"✅ Added API: {api_config['name']}\nTotal APIs: {API_MANAGER.get_api_count()}"
    return status, gr.update(value="")

def complete_video_dubbing_workflow_with_realtime_updates(video_file, target_languages, custom_prompt="", reference_audio=None):
    """Complete video dubbing workflow with real-time progress updates"""
    initialize_dubbing_system()
    
    # Initialize all progress states
    progress_states = {
        "transcription_status": "🎬 Starting video processing...",
        "parakeet_status": "🤖 Preparing Parakeet model...",
        "transcript_html": "<div style='padding: 20px; text-align: center; color: #666;'>Processing...</div>",
        "translation_status": "⏳ Waiting for transcription...",
        "chunk_progress": "📊 No chunks created yet",
        "translation_results": "<div style='padding: 20px; text-align: center; color: #666;'>Waiting...</div>",
        "tts_status": "⏳ Waiting for translation...",
        "audio_processing": "⏳ Speed adjustment pending...",
        "tts_results": "<div style='padding: 20px; text-align: center; color: #666;'>Waiting...</div>",
        "video_assembly": "⏳ Waiting for audio generation...",
        "final_output": "📋 No files generated yet",
        "output_files": "<div style='padding: 20px; text-align: center; color: #666;'>Waiting...</div>",
        "main_status": "🚀 Starting dubbing process...",
        "final_video": None
    }
    
    # Validation
    if not video_file:
        progress_states["main_status"] = "❌ Please upload a video file"
        progress_states["transcription_status"] = "❌ No video file provided"
        return tuple(progress_states.values())
    
    if not target_languages:
        progress_states["main_status"] = "❌ Please select at least one target language"
        progress_states["translation_status"] = "❌ No target languages selected"
        return tuple(progress_states.values())
    
    if API_MANAGER.get_api_count() == 0:
        progress_states["main_status"] = "❌ Please add at least one Gemini API key"
        progress_states["translation_status"] = "❌ No API keys configured"
        return tuple(progress_states.values())
    
    try:
        # Step 1: Extract audio from video
        progress_states["transcription_status"] = "🎬 Extracting audio from video..."
        print("🎬 Extracting audio from video...")
        
        audio_path = enhanced_extract_audio_from_video(video_file)
        if not audio_path:
            progress_states["transcription_status"] = "❌ Failed to extract audio from video"
            progress_states["main_status"] = "❌ Audio extraction failed"
            return tuple(progress_states.values())
        
        progress_states["transcription_status"] = "✅ Audio extracted successfully"
        
        # Step 2: Load Parakeet model if not loaded
        progress_states["parakeet_status"] = "🤖 Loading Parakeet TDT model..."
        
        if PARAKEET_MODEL.model is None:
            def parakeet_progress(msg):
                progress_states["parakeet_status"] = msg
            
            PARAKEET_MODEL.load_model(progress_callback=parakeet_progress)
        
        progress_states["parakeet_status"] = "✅ Parakeet model ready"
        
        # Step 3: Transcribe with timestamps
        progress_states["transcription_status"] = "🎤 Transcribing audio with Parakeet TDT..."
        
        segments = PARAKEET_MODEL.transcribe_with_timestamps(audio_path, None, False)
        if not segments:
            progress_states["transcription_status"] = "❌ Failed to transcribe audio"
            progress_states["main_status"] = "❌ Transcription failed"
            return tuple(progress_states.values())
        
        progress_states["transcription_status"] = f"✅ Transcribed {len(segments)} segments"
        
        # Create transcript HTML display
        transcript_html = create_transcript_html_table(segments)
        progress_states["transcript_html"] = transcript_html
        
        progress_states["main_status"] = "✅ Transcription complete! Starting translation..."
        
        # Step 4: Translate to target languages using smart chunking
        progress_states["translation_status"] = "🌍 Starting smart chunked translation..."
        print("🌍 Starting smart chunked translation...")
        
        def translation_progress_callback(message):
            progress_states["translation_status"] = message
            progress_states["chunk_progress"] = message
            print(f"📊 {message}")
        
        translations = GEMINI_TRANSLATOR.translate_segments(
            segments, 
            target_languages, 
            custom_prompt,
            progress_callback=translation_progress_callback
        )
        
        # Debug: Check what translations were actually stored
        print("🔍 Debug: Translation results:")
        for lang, lang_segments in translations.items():
            print(f"   Language {lang}: {len(lang_segments)} segments")
            if lang_segments:
                print(f"   First segment: '{lang_segments[0]['text'][:100]}...'")
                print(f"   Last segment: '{lang_segments[-1]['text'][:100]}...'")
        print("🔍 End translation debug")
        
        # Create translation results HTML
        translation_html = "<div style='font-family: monospace;'>"
        for language in target_languages:
            lang_name = SUPPORTED_LANGUAGES.get(language, language)
            lang_translations = translations.get(language, [])
            
            translation_html += f"<h4>🌍 {lang_name} ({len(lang_translations)} segments)</h4>"
            translation_html += "<div style='max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 5px 0;'>"
            
            for i, trans_seg in enumerate(lang_translations[:10]):  # Show first 10
                original_seg = segments[i] if i < len(segments) else {"text": "N/A"}
                translation_html += f"""
                <div style='margin: 5px 0; padding: 5px; background: #f9f9f9;'>
                    <strong>Segment {i + 1}:</strong> {trans_seg['start']:.1f}s - {trans_seg['end']:.1f}s<br>
                    <span style='color: #666;'>Original:</span> {original_seg.get('text', 'N/A')[:100]}...<br>
                    <span style='color: #0066cc;'>Translated:</span> {trans_seg['text'][:100]}...
                </div>
                """
            
            if len(lang_translations) > 10:
                translation_html += f"<div style='text-align: center; color: #666;'>... and {len(lang_translations) - 10} more segments</div>"
            
            translation_html += "</div>"
        
        translation_html += "</div>"
        progress_states["translation_results"] = translation_html
        progress_states["translation_status"] = f"✅ Translation complete for {len(target_languages)} languages"
        
        # Step 5: Generate TTS for each language
        progress_states["tts_status"] = "🗣️ Starting TTS generation with Chatterbox..."
        progress_states["audio_processing"] = "🔧 Preparing audio processing..."
        print("🗣️ Generating TTS for translated segments...")
        
        dubbed_videos = {}
        tts_html = "<div style='font-family: monospace;'>"
        
        for lang_idx, language in enumerate(target_languages):
            lang_name = SUPPORTED_LANGUAGES.get(language, language)
            progress_states["tts_status"] = f"🎭 Generating audio for {lang_name}..."
            
            print(f"🎭 Processing {language}...")
            
            # Generate TTS for each segment
            audio_segments = []
            lang_tts_html = f"<h4>🎙️ {lang_name} TTS Progress</h4>"
            lang_tts_html += "<div style='max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 5px 0;'>"
            
            for seg_idx, segment in enumerate(translations[language]):
                try:
                    progress_states["tts_status"] = f"🗣️ Generating TTS segment {seg_idx + 1}/{len(translations[language])} for {lang_name}"
                    progress_states["audio_processing"] = f"🔧 Processing segment {seg_idx + 1}: {segment['text'][:50]}..."
                    
                    print(f"🔍 Debug TTS Input - Language: {language}")
                    print(f"🔍 Debug TTS Input - Text: '{segment['text'][:100]}...'")
                    print(f"🔍 Debug TTS Input - Duration: {segment['duration']}")
                    
                    tts_audio = generate_tts_for_segment_enhanced(
                        segment['text'], 
                        language, 
                        segment['duration'],
                        reference_audio=reference_audio
                    )
                    
                    if tts_audio:
                        audio_segments.append(tts_audio)
                        lang_tts_html += f"""
                        <div style='margin: 3px 0; padding: 3px; background: #e8f5e8;'>
                            ✅ Segment {seg_idx + 1}: {segment['text'][:60]}... ({segment['duration']:.2f}s)
                        </div>
                        """
                        print(f"✅ TTS generated for segment {seg_idx + 1}")
                    else:
                        lang_tts_html += f"""
                        <div style='margin: 3px 0; padding: 3px; background: #ffe8e8;'>
                            ❌ Segment {seg_idx + 1}: TTS Failed
                        </div>
                        """
                        
                except Exception as e:
                    print(f"❌ TTS error for segment {seg_idx}: {e}")
                    lang_tts_html += f"""
                    <div style='margin: 3px 0; padding: 3px; background: #ffe8e8;'>
                        ❌ Segment {seg_idx + 1}: Error - {str(e)[:50]}...
                    </div>
                    """
            
            lang_tts_html += "</div>"
            tts_html += lang_tts_html
            
            if audio_segments:
                progress_states["audio_processing"] = f"🔧 Combining {len(audio_segments)} audio segments for {lang_name}..."
                
                # Combine audio segments
                combined_audio_path = f"temp_dubbed_{language}.wav"
                final_audio = AUDIO_PROCESSOR.combine_audio_segments(
                    audio_segments, combined_audio_path
                )
                
                progress_states["video_assembly"] = f"🎬 Assembling final video for {lang_name}..."
                
                # Combine with original video
                cache_dir = os.path.join("cache", "dubbed_videos")
                os.makedirs(cache_dir, exist_ok=True)
                output_video_path = os.path.join(cache_dir, f"dubbed_{language}_{os.path.basename(video_file)}")
                final_video = combine_video_with_audio(
                    video_file, final_audio, output_video_path
                )
                
                if final_video:
                    dubbed_videos[language] = final_video
        
        tts_html += "</div>"
        progress_states["tts_results"] = tts_html
        progress_states["tts_status"] = f"✅ TTS generation complete for {len(target_languages)} languages"
        progress_states["audio_processing"] = "✅ All audio processing complete"
        
        # Unload Parakeet model to free memory for Chatterbox
        PARAKEET_MODEL.unload_model()
        progress_states["parakeet_status"] = "💤 Model unloaded (memory freed)"
        
        # Final assembly status
        progress_states["video_assembly"] = f"✅ Video assembly complete for {len(dubbed_videos)} languages"
        
        # Create output files HTML
        output_html = "<div style='font-family: monospace;'>"
        if dubbed_videos:
            output_html += f"<h4>🎬 Generated Videos ({len(dubbed_videos)} files)</h4>"
            for lang, video_path in dubbed_videos.items():
                lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
                output_html += f"""
                <div style='margin: 5px 0; padding: 10px; background: #e8f5e8; border-radius: 5px;'>
                    <strong>🌍 {lang_name}</strong><br>
                    📁 File: {os.path.basename(video_path)}<br>
                    📏 Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB
                </div>
                """
        else:
            output_html += "<div style='color: #cc0000;'>❌ No videos were generated</div>"
        
        output_html += "</div>"
        progress_states["output_files"] = output_html
        progress_states["final_output"] = f"✅ {len(dubbed_videos)} videos generated successfully"
        
        # Final status
        progress_states["main_status"] = f"🎉 Dubbing Complete! Generated {len(dubbed_videos)} videos in {len(target_languages)} languages"
        
        # Return first video for preview
        first_video = list(dubbed_videos.values())[0] if dubbed_videos else None
        progress_states["final_video"] = first_video
        
        return tuple(progress_states.values())
        
    except Exception as e:
        error_msg = f"❌ Dubbing error: {str(e)}"
        print(error_msg)
        
        # Update all progress states with error
        progress_states["main_status"] = error_msg
        progress_states["transcription_status"] = "❌ Process failed"
        progress_states["parakeet_status"] = "❌ Error occurred"
        progress_states["translation_status"] = "❌ Process failed"
        progress_states["tts_status"] = "❌ Process failed"
        progress_states["video_assembly"] = "❌ Process failed"
        
        # Try to unload model if it was loaded
        try:
            PARAKEET_MODEL.unload_model()
        except:
            pass
        
        return tuple(progress_states.values())

def process_video_for_dubbing(video_file, target_languages, custom_prompt=""):
    """Main function to process video for dubbing"""
    initialize_dubbing_system()
    
    if not video_file:
        return None, "❌ Please upload a video file"
    
    if not target_languages:
        return None, "❌ Please select at least one target language"
    
    if API_MANAGER.get_api_count() == 0:
        return None, "❌ Please add at least one Gemini API key"
    
    try:
        # Step 1: Extract audio from video
        print("🎬 Extracting audio from video...")
        audio_path = AUDIO_PROCESSOR.extract_audio_from_video(video_file)
        if not audio_path:
            return None, "❌ Failed to extract audio from video"
        
        # Step 2: Transcribe with timestamps
        print("🎤 Transcribing audio with timestamps...")
        segments = PARAKEET_MODEL.transcribe_with_timestamps(audio_path)
        if not segments:
            return None, "❌ Failed to transcribe audio. Check if Parakeet model is available."
        
        # Step 3: Translate to target languages
        print("🌍 Translating segments...")
        translations = GEMINI_TRANSLATOR.translate_segments(
            segments, target_languages, custom_prompt
        )
        
        # Step 4: Generate TTS for each language
        print("🗣️ Generating TTS for translated segments...")
        dubbed_videos = {}
        
        for language in target_languages:
            print(f"🎭 Processing {language}...")
            
            # Generate TTS for each segment
            audio_segments = []
            for segment in translations[language]:
                # Generate TTS using chatterbox
                tts_audio = generate_tts_for_segment(
                    segment['text'], 
                    language, 
                    segment['duration']
                )
                if tts_audio:
                    audio_segments.append(tts_audio)
            
            if audio_segments:
                # Combine audio segments
                combined_audio_path = f"temp_dubbed_{language}.wav"
                final_audio = AUDIO_PROCESSOR.combine_audio_segments(
                    audio_segments, combined_audio_path
                )
                
                # Combine with original video
                output_video_path = f"dubbed_{language}_{os.path.basename(video_file)}"
                final_video = combine_video_with_audio(
                    video_file, final_audio, output_video_path
                )
                
                if final_video:
                    dubbed_videos[language] = final_video
        
        # Create result summary
        result_info = f"""✅ Dubbing Complete!

📊 Processing Summary:
• Original segments: {len(segments)}
• Languages processed: {len(target_languages)}
• Videos created: {len(dubbed_videos)}

🎬 Generated Videos:
"""
        for lang, video_path in dubbed_videos.items():
            result_info += f"• {SUPPORTED_LANGUAGES.get(lang, lang)}: {video_path}\n"
        
        # Return the first video for preview (or could return all)
        first_video = list(dubbed_videos.values())[0] if dubbed_videos else None
        
        return first_video, result_info
        
    except Exception as e:
        error_msg = f"❌ Dubbing error: {str(e)}"
        print(error_msg)
        return None, error_msg

def generate_tts_for_segment_enhanced(text: str, language: str, target_duration: float, reference_audio=None):
    """Enhanced TTS generation with speed adjustment and reference audio support"""
    try:
        global MULTILINGUAL_MODEL
        if MULTILINGUAL_MODEL is None:
            print("🔄 Loading multilingual TTS model...")
            MULTILINGUAL_MODEL = get_or_load_model()
        
        if MULTILINGUAL_MODEL is None:
            print("❌ No TTS model available - please download multilingual models first")
            return None
        
        print(f"🗣️ Generating TTS: '{text[:50]}...' in {language}")
        
        # Generate TTS (multilingual model doesn't support reference audio)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if reference_audio and os.path.exists(reference_audio):
                print(f"🎤 Reference audio provided but not supported by multilingual model: {reference_audio}")
                print("🔄 Using default voice for multilingual TTS")
            
            wav = MULTILINGUAL_MODEL.generate(
                text,
                language_id=language,
                temperature=0.8,
                cfg_weight=0.5,
            )
        
        # Convert to numpy array
        if hasattr(wav, 'squeeze'):
            audio_array = wav.squeeze(0).numpy()
        else:
            audio_array = wav
        
        # Save to temporary file
        temp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(temp_path, audio_array, MULTILINGUAL_MODEL.sr)
        
        # Adjust speed to match target duration
        adjusted_path = AUDIO_PROCESSOR.adjust_audio_speed(temp_path, target_duration)
        
        print(f"✅ TTS generated and adjusted for {target_duration:.2f}s duration")
        return adjusted_path
        
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_tts_for_segment(text: str, language: str, target_duration: float):
    """Generate TTS audio for a single segment"""
    try:
        # Get the current multilingual model
        global MULTILINGUAL_MODEL
        if MULTILINGUAL_MODEL is None:
            MULTILINGUAL_MODEL = get_or_load_model()
        
        if MULTILINGUAL_MODEL is None:
            print("❌ No TTS model available")
            return None
        
        # Generate TTS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            wav = MULTILINGUAL_MODEL.generate(
                text,
                language_id=language,
                temperature=0.8,
                cfg_weight=0.5,
            )
        
        # Convert to numpy array
        audio_array = wav.squeeze(0).numpy()
        
        # Save to temporary file
        temp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(temp_path, audio_array, MULTILINGUAL_MODEL.sr)
        
        # Adjust speed to match target duration
        adjusted_path = AUDIO_PROCESSOR.adjust_audio_speed(temp_path, target_duration)
        
        return adjusted_path
        
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        return None

def get_api_status():
    """Get current API manager status"""
    initialize_dubbing_system()
    
    if API_MANAGER.get_api_count() == 0:
        return "❌ No API keys added. Please add at least one Gemini API key."
    
    status = f"✅ {API_MANAGER.get_api_count()} API key(s) available\n"
    status += "Ready for translation and dubbing!"
    
    return status

def create_transcript_html_table(segments):
    """Create an HTML table for transcript display"""
    if not segments:
        return "No segments found"
    
    html = """
    <style>
    .transcript-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-family: monospace;
    }
    .transcript-table th, .transcript-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .transcript-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .transcript-table tr:hover {
        background-color: #f5f5f5;
    }
    .time-cell {
        font-weight: bold;
        color: #0066cc;
    }
    </style>
    <table class="transcript-table">
    <tr>
        <th>Start</th>
        <th>End</th>
        <th>Duration</th>
        <th>Text</th>
    </tr>
    """
    
    for segment in segments:
        html += f"""
        <tr>
            <td class="time-cell">{segment['start']:.2f}s</td>
            <td class="time-cell">{segment['end']:.2f}s</td>
            <td class="time-cell">{segment.get('duration', segment['end'] - segment['start']):.2f}s</td>
            <td>{segment['text']}</td>
        </tr>
        """
    
    html += "</table>"
    return html

# Multilingual model download configuration
MULTILINGUAL_MODEL_FILES = {
    'Cangjie5_TC': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/Cangjie5_TC.json',
    'conds': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt',
    'mtl_tokenizer': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/mtl_tokenizer.json',
    's3gen': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.pt',
    't3_23lang': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_23lang.safetensors',
    've': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.pt'
}

# Model download directory
MODEL_DOWNLOAD_DIR = "models/multilingual"
download_status = {"status": "ready", "progress": 0, "current_file": "", "total_files": 0}

def ensure_model_download_dir():
    """Ensure the model download directory exists."""
    if not os.path.exists(MODEL_DOWNLOAD_DIR):
        os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
        print(f"📁 Created model download directory: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")

def check_multilingual_models_exist():
    """Check if multilingual model files already exist."""
    ensure_model_download_dir()
    missing_files = []
    existing_files = []
    
    for model_name, url in MULTILINGUAL_MODEL_FILES.items():
        # Extract the correct filename with extension from the URL
        filename_with_ext = url.split('/')[-1]
        model_path = os.path.join(MODEL_DOWNLOAD_DIR, filename_with_ext)
        if os.path.exists(model_path):
            existing_files.append(model_name)
        else:
            missing_files.append(model_name)
    
    return existing_files, missing_files

def download_file_with_progress(url, filepath, filename):
    """Download a file with progress tracking."""
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        download_status["progress"] = progress
        
        print(f"✅ Downloaded {filename} successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def download_multilingual_models():
    """Download all multilingual model files."""
    global download_status
    
    try:
        ensure_model_download_dir()
        
        # Check what needs to be downloaded
        existing_files, missing_files = check_multilingual_models_exist()
        
        if not missing_files:
            download_status["status"] = "complete"
            download_status["progress"] = 100
            return "✅ All multilingual model files already exist!"
        
        download_status["status"] = "downloading"
        download_status["total_files"] = len(missing_files)
        
        print(f"🌍 Starting download of {len(missing_files)} multilingual model files...")
        
        for i, model_name in enumerate(missing_files):
            download_status["current_file"] = model_name
            download_status["progress"] = (i / len(missing_files)) * 100
            
            url = MULTILINGUAL_MODEL_FILES[model_name]
            # Extract the correct filename with extension from the URL
            filename_with_ext = url.split('/')[-1]
            filepath = os.path.join(MODEL_DOWNLOAD_DIR, filename_with_ext)
            
            success = download_file_with_progress(url, filepath, filename_with_ext)
            
            if not success:
                download_status["status"] = "error"
                return f"❌ Failed to download {model_name}"
        
        download_status["status"] = "complete"
        download_status["progress"] = 100
        download_status["current_file"] = ""
        
        print("🎉 All multilingual model files downloaded successfully!")
        return "🎉 Multilingual models downloaded successfully! You can now enable multilingual mode."
        
    except Exception as e:
        download_status["status"] = "error"
        error_msg = f"❌ Download error: {str(e)}"
        print(error_msg)
        return error_msg

def download_models_async():
    """Download models in a separate thread to avoid blocking the UI."""
    threading.Thread(target=download_multilingual_models, daemon=True).start()

def get_download_status():
    """Get current download status for UI updates."""
    status = download_status["status"]
    progress = download_status.get("progress", 0)
    current_file = download_status.get("current_file", "")
    total_files = download_status.get("total_files", 0)
    
    if status == "ready":
        return "📋 Ready to download multilingual models"
    elif status == "downloading":
        file_info = f" ({current_file})" if current_file else ""
        return f"📥 Downloading... {progress:.1f}%{file_info}"
    elif status == "complete":
        return "✅ Download complete! Multilingual models ready."
    elif status == "error":
        return "❌ Download failed. Check console for details."
    else:
        return f"Status: {status}"

def check_model_files_status():
    """Check and return the status of model files."""
    existing_files, missing_files = check_multilingual_models_exist()
    
    if not missing_files:
        status_text = f"✅ All multilingual model files present ({len(existing_files)} files)\n"
        status_text += f"📁 Location: {os.path.abspath(MODEL_DOWNLOAD_DIR)}\n"
        status_text += f"Files: {', '.join(existing_files)}"
        return status_text
    else:
        status_text = f"⚠️ Missing {len(missing_files)} model files\n"
        if existing_files:
            status_text += f"✅ Found: {', '.join(existing_files)}\n"
        status_text += f"❌ Missing: {', '.join(missing_files)}\n"
        status_text += "Click 'Download Models' to get the missing files."
        return status_text

def load_model_manually():
    """Manually load the multilingual model into memory."""
    global MULTILINGUAL_MODEL
    
    try:
        if MULTILINGUAL_MODEL is not None:
            return "✅ Model already loaded in memory!", True
        
        print("🚀 Manually loading multilingual model...")
        MULTILINGUAL_MODEL = get_or_load_model()
        
        if MULTILINGUAL_MODEL is not None:
            return "✅ Model loaded successfully! Ready for speech generation.", True
        else:
            return "❌ Failed to load model. Please check your model files.", False
            
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return error_msg, False

def check_model_loaded_status():
    """Check if the model is loaded and return status."""
    global MULTILINGUAL_MODEL
    
    if MULTILINGUAL_MODEL is not None:
        return "✅ Model loaded in memory - Ready for generation!"
    else:
        existing_files, missing_files = check_multilingual_models_exist()
        if not missing_files:
            return "📁 Models downloaded but not loaded - Click 'Load Model' to use them"
        else:
            return "📥 Models not downloaded - Use download section to get them"

def should_show_load_button():
    """Check if the load model button should be visible."""
    global MULTILINGUAL_MODEL
    
    if MULTILINGUAL_MODEL is not None:
        return False  # Hide button if model is already loaded
    
    existing_files, missing_files = check_multilingual_models_exist()
    return len(missing_files) == 0  # Show button if all models are downloaded

# --- Voice Presets System ---
PRESETS_FILE = "voice_presets.json"
PRESETS_AUDIO_DIR = "saved_voices"

def ensure_presets_dir():
    """Ensure the presets audio directory exists."""
    if not os.path.exists(PRESETS_AUDIO_DIR):
        os.makedirs(PRESETS_AUDIO_DIR)
        print(f"📁 Created presets directory: {os.path.abspath(PRESETS_AUDIO_DIR)}")

def copy_reference_audio(ref_audio_path, preset_name):
    """Copy reference audio to presets directory."""
    if not ref_audio_path or not os.path.exists(ref_audio_path):
        return None
    
    try:
        ensure_presets_dir()
        
        # Get file extension
        _, ext = os.path.splitext(ref_audio_path)
        if not ext:
            ext = '.wav'  # Default extension
        
        # Create unique filename for this preset
        safe_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        audio_filename = f"{safe_name}_voice{ext}"
        audio_path = os.path.join(PRESETS_AUDIO_DIR, audio_filename)
        
        # Copy the file
        shutil.copy2(ref_audio_path, audio_path)
        
        print(f"🎤 Copied reference audio to: {os.path.abspath(audio_path)}")
        return audio_path
        
    except Exception as e:
        print(f"❌ Error copying reference audio: {e}")
        return None

def load_voice_presets():
    """Load voice presets from JSON file."""
    try:
        preset_path = os.path.abspath(PRESETS_FILE)
        print(f"🔍 Looking for presets file at: {preset_path}")
        
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, 'r') as f:
                presets = json.load(f)
                print(f"✅ Loaded {len(presets)} voice presets from file")
                
                # Verify audio files still exist
                for name, preset in presets.items():
                    audio_path = preset.get('ref_audio_path', '')
                    if audio_path and os.path.exists(audio_path):
                        print(f"  🎤 Preset '{name}' has valid audio file")
                    elif audio_path:
                        print(f"  ⚠️ Preset '{name}' audio file missing: {audio_path}")
                
                return presets
        else:
            print("📝 No presets file found, starting with empty presets")
    except Exception as e:
        print(f"❌ Error loading presets: {e}")
    return {}

def save_voice_presets(presets):
    """Save voice presets to JSON file."""
    try:
        preset_path = os.path.abspath(PRESETS_FILE)
        print(f"💾 Saving voice presets to: {preset_path}")
        
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        
        print(f"✅ Successfully saved {len(presets)} voice presets")
        return True
    except Exception as e:
        print(f"❌ Error saving presets: {e}")
        return False

def save_voice_preset(name, settings):
    """Save a new voice preset with reference audio."""
    presets = load_voice_presets()
    
    # Copy the reference audio file if provided
    ref_audio_path = settings.get('ref_audio', '')
    saved_audio_path = None
    
    if ref_audio_path:
        saved_audio_path = copy_reference_audio(ref_audio_path, name)
        if not saved_audio_path:
            print(f"⚠️ Warning: Could not save reference audio for preset '{name}'")
    
    presets[name] = {
        'exaggeration': settings['exaggeration'],
        'temperature': settings['temperature'],
        'cfg_weight': settings['cfg_weight'],
        'chunk_size': settings['chunk_size'],
        'language': settings.get('language', 'en'),  # Save language setting
        'ref_audio_path': saved_audio_path or '',  # Path to saved audio file
        'original_ref_audio': ref_audio_path or '',  # Original path for reference
        'created': datetime.now().isoformat()
    }
    
    success = save_voice_presets(presets)
    if success:
        if saved_audio_path:
            print(f"🎭 Saved voice preset '{name}' with custom voice audio")
        else:
            print(f"🎭 Saved voice preset '{name}' (parameters only, no custom voice)")
    return success

def load_voice_preset(name):
    """Load a voice preset by name."""
    presets = load_voice_presets()
    preset = presets.get(name, None)
    if preset:
        audio_path = preset.get('ref_audio_path', '')
        if audio_path and os.path.exists(audio_path):
            print(f"🎭 Loaded voice preset '{name}' with custom voice: {audio_path}")
        else:
            print(f"🎭 Loaded voice preset '{name}' (parameters only)")
    return preset

def delete_voice_preset(name):
    """Delete a voice preset and its audio file."""
    presets = load_voice_presets()
    if name in presets:
        preset = presets[name]
        
        # Delete associated audio file
        audio_path = preset.get('ref_audio_path', '')
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"🗑️ Deleted audio file: {audio_path}")
            except Exception as e:
                print(f"⚠️ Could not delete audio file: {e}")
        
        del presets[name]
        success = save_voice_presets(presets)
        if success:
            print(f"🗑️ Deleted voice preset '{name}'")
        return success
    return False

def get_preset_names():
    """Get list of all preset names."""
    presets = load_voice_presets()
    names = list(presets.keys())
    print(f"📋 Available voice presets: {names}")
    return names

# --- Audio Effects ---
def apply_reverb(audio, sr, room_size=0.3, damping=0.5, wet_level=0.3):
    """Apply more noticeable reverb effect to audio."""
    try:
        # Create multiple delayed versions for richer reverb
        reverb_audio = audio.copy()
        
        # Early reflections (multiple short delays)
        delays = [0.01, 0.02, 0.03, 0.05, 0.08]  # Multiple delay times in seconds
        gains = [0.6, 0.4, 0.3, 0.2, 0.15]      # Corresponding gains
        
        for delay_time, gain in zip(delays, gains):
            delay_samples = int(sr * delay_time)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain * (1 - damping)
                reverb_audio += delayed * wet_level
        
        # Late reverberation (longer decay)
        late_delay = int(sr * 0.1)  # 100ms
        if late_delay < len(audio):
            late_reverb = np.zeros_like(audio)
            late_reverb[late_delay:] = audio[:-late_delay] * 0.3 * (1 - damping)
            reverb_audio += late_reverb * wet_level * room_size
        
        return np.clip(reverb_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Reverb error: {e}")
        return audio

def apply_echo(audio, sr, delay=0.3, decay=0.5):
    """Apply echo effect to audio."""
    try:
        delay_samples = int(sr * delay)
        if delay_samples < len(audio):
            echo_audio = audio.copy()
            echo_audio[delay_samples:] += audio[:-delay_samples] * decay
            return np.clip(echo_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Echo error: {e}")
    return audio

def apply_pitch_shift(audio, sr, semitones):
    """Apply simple pitch shift (speed change method)."""
    try:
        if semitones == 0:
            return audio
        
        # Simple pitch shift by resampling (changes speed too)
        factor = 2 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio

# --- Advanced Audio Processing ---

def test_equalizer_functionality():
    """Test function to verify equalizer is working correctly."""
    try:
        # Create test signal
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create test audio with multiple frequencies
        test_audio = (
            0.3 * np.sin(2 * np.pi * 100 * t) +    # Bass
            0.3 * np.sin(2 * np.pi * 1000 * t) +   # Mid
            0.3 * np.sin(2 * np.pi * 5000 * t)     # High
        )
        
        # Test EQ settings (boost bass, cut mid, boost high)
        eq_bands = {
            'sub_bass': 0,
            'bass': 6,
            'low_mid': 0,
            'mid': -6,
            'high_mid': 0,
            'presence': 6,
            'brilliance': 0
        }
        
        # Apply equalizer
        processed = apply_equalizer(test_audio, sr, eq_bands)
        
        if processed is not None and len(processed) == len(test_audio):
            print("✅ Equalizer test passed - processing working correctly")
            return True
        else:
            print("❌ Equalizer test failed - output issue")
            return False
            
    except Exception as e:
        print(f"❌ Equalizer test failed with error: {e}")
        return False

def apply_noise_reduction(audio, sr, noise_factor=0.02, spectral_floor=0.002):
    """Apply spectral subtraction noise reduction to clean up audio."""
    try:
        print("🧹 Applying noise reduction...")
        
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Estimate noise profile from first 0.5 seconds (assumed to be quieter)
        noise_frame_count = min(int(0.5 * sr / 512), magnitude.shape[1] // 4)
        noise_profile = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
        
        # Apply spectral subtraction with over-subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.001  # Floor factor
        
        # Subtract noise profile
        clean_magnitude = magnitude - alpha * noise_profile
        
        # Apply spectral floor to prevent artifacts
        spectral_floor_level = beta * magnitude
        clean_magnitude = np.maximum(clean_magnitude, spectral_floor_level)
        
        # Reconstruct signal
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=512)
        
        # Ensure same length as input
        if len(clean_audio) < len(audio):
            clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))
        else:
            clean_audio = clean_audio[:len(audio)]
        
        return np.clip(clean_audio, -1.0, 1.0)
        
    except Exception as e:
        print(f"Noise reduction error: {e}")
        return audio

def apply_equalizer(audio, sr, eq_bands):
    """Apply multi-band equalizer to audio."""
    try:
        if not any(gain != 0 for gain in eq_bands.values()):
            return audio  # No EQ applied
            
        print("🎛️ Applying equalizer...")
        
        # Define frequency bands (Hz) with better ranges
        band_ranges = {
            'sub_bass': (20, 80),
            'bass': (80, 250), 
            'low_mid': (250, 800),
            'mid': (800, 2500),
            'high_mid': (2500, 5000),
            'presence': (5000, 10000),
            'brilliance': (10000, 20000)
        }
        
        # Start with original audio
        processed_audio = audio.copy().astype(np.float64)
        
        for band_name, gain_db in eq_bands.items():
            if gain_db == 0 or band_name not in band_ranges:
                continue
                
            low_freq, high_freq = band_ranges[band_name]
            
            # Ensure frequencies are within Nyquist limit
            nyquist = sr / 2
            low_freq = min(low_freq, nyquist * 0.95)
            high_freq = min(high_freq, nyquist * 0.95)
            
            if low_freq >= high_freq:
                continue
            
            # Design bandpass filter with lower order to reduce artifacts
            low_norm = max(low_freq / nyquist, 0.001)  # Avoid zero frequency
            high_norm = min(high_freq / nyquist, 0.999)  # Avoid Nyquist frequency
            
            try:
                # Use second-order filter to reduce artifacts
                if band_name == 'sub_bass':
                    # Low-pass filter for sub bass
                    b, a = butter(2, high_norm, btype='low')
                elif band_name == 'brilliance':
                    # High-pass filter for brilliance
                    b, a = butter(2, low_norm, btype='high')
                else:
                    # Bandpass filter for mid bands
                    b, a = butter(2, [low_norm, high_norm], btype='band')
                
                # Filter the audio
                band_audio = filtfilt(b, a, audio.astype(np.float64))
                
                # Convert dB gain to linear gain
                gain_linear = 10 ** (gain_db / 20.0)
                
                # Apply gain correctly: multiply band by gain, then blend with original
                if gain_db > 0:
                    # Boost: add the boosted band energy
                    boost_amount = (gain_linear - 1.0)
                    processed_audio += band_audio * boost_amount
                else:
                    # Cut: reduce the band energy in the processed audio
                    cut_amount = (1.0 - gain_linear)
                    processed_audio -= band_audio * cut_amount
                
                print(f"  Applied {gain_db:+.1f}dB to {band_name} ({low_freq}-{high_freq}Hz)")
                
            except Exception as band_error:
                print(f"EQ band {band_name} error: {band_error}")
                continue
        
        # Normalize to prevent clipping while preserving dynamics
        max_val = np.abs(processed_audio).max()
        if max_val > 0.95:
            processed_audio = processed_audio * (0.95 / max_val)
        
        return processed_audio.astype(np.float32)
        
    except Exception as e:
        print(f"Equalizer error: {e}")
        return audio

def apply_spatial_audio(audio, sr, azimuth=0, elevation=0, distance=1.0):
    """Apply 3D spatial audio positioning using simple HRTF-like processing."""
    try:
        if azimuth == 0 and elevation == 0 and distance == 1.0:
            return audio  # No spatial processing needed
            
        print(f"🎧 Applying 3D spatial audio (az: {azimuth}°, el: {elevation}°, dist: {distance})")
        
        # Convert to stereo if mono
        if len(audio.shape) == 1:
            # Simple stereo panning based on azimuth
            azimuth_rad = np.radians(azimuth)
            
            # Calculate left/right gains using equal-power panning
            left_gain = np.cos((azimuth_rad + np.pi/2) / 2)
            right_gain = np.sin((azimuth_rad + np.pi/2) / 2)
            
            # Apply distance attenuation
            distance_gain = 1.0 / max(distance, 0.1)
            left_gain *= distance_gain
            right_gain *= distance_gain
            
            # Apply elevation effect (simple high-frequency filtering)
            processed_audio = audio.copy()
            if elevation != 0:
                # High-pass filter for upward elevation, low-pass for downward
                elevation_factor = elevation / 90.0  # Normalize to -1 to 1
                if elevation_factor > 0:
                    # Upward - enhance high frequencies
                    cutoff = 2000 + elevation_factor * 3000
                    b, a = butter(2, cutoff / (sr/2), btype='high')
                    high_freq = filtfilt(b, a, audio)
                    processed_audio = audio + high_freq * elevation_factor * 0.3
                else:
                    # Downward - enhance low frequencies  
                    cutoff = 1000 + abs(elevation_factor) * 2000
                    b, a = butter(2, cutoff / (sr/2), btype='low')
                    low_freq = filtfilt(b, a, audio)
                    processed_audio = audio + low_freq * abs(elevation_factor) * 0.3
            
            # Create stereo output
            stereo_audio = np.column_stack([
                processed_audio * left_gain,
                processed_audio * right_gain
            ])
            
            return np.clip(stereo_audio, -1.0, 1.0)
        
        return audio  # Return original if already stereo or other format
        
    except Exception as e:
        print(f"Spatial audio error: {e}")
        return audio

def mix_with_background(speech_audio, sr, background_path, bg_volume=0.3, speech_volume=1.0, fade_in=1.0, fade_out=1.0):
    """Mix generated speech with background music/ambience."""
    try:
        if not background_path or not os.path.exists(background_path):
            return speech_audio
            
        print(f"🎵 Mixing with background audio: {os.path.basename(background_path)}")
        
        # Load background audio
        bg_audio, bg_sr = librosa.load(background_path, sr=sr)
        
        # Ensure speech is 1D
        if len(speech_audio.shape) > 1:
            speech_audio = np.mean(speech_audio, axis=1)
            
        speech_length = len(speech_audio)
        bg_length = len(bg_audio)
        
        # Handle different background audio lengths
        if bg_length < speech_length:
            # Loop background audio if it's shorter
            repeat_count = int(np.ceil(speech_length / bg_length))
            bg_audio = np.tile(bg_audio, repeat_count)[:speech_length]
        else:
            # Trim background audio if it's longer
            bg_audio = bg_audio[:speech_length]
        
        # Apply volume adjustments
        speech_mixed = speech_audio * speech_volume
        bg_mixed = bg_audio * bg_volume
        
        # Apply fades to background
        fade_in_samples = int(fade_in * sr)
        fade_out_samples = int(fade_out * sr)
        
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            bg_mixed[:fade_in_samples] *= fade_in_curve
            
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            bg_mixed[-fade_out_samples:] *= fade_out_curve
        
        # Mix the audio
        mixed_audio = speech_mixed + bg_mixed
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed_audio).max()
        if max_val > 0.95:
            mixed_audio = mixed_audio / max_val * 0.95
        
        return mixed_audio
        
    except Exception as e:
        print(f"Background mixing error: {e}")
        return speech_audio

def apply_audio_effects(audio, sr, effects_settings):
    """Apply selected audio effects to the generated audio."""
    processed_audio = audio.copy()
    
    print(f"🎵 Starting audio effects processing...")
    print(f"   Input audio: shape={audio.shape}, max={np.max(np.abs(audio)):.4f}, dtype={audio.dtype}")
    
    # Basic effects (existing)
    if effects_settings.get('enable_reverb', False):
        processed_audio = apply_reverb(
            processed_audio, sr,
            room_size=effects_settings.get('reverb_room', 0.3),
            damping=effects_settings.get('reverb_damping', 0.5),
            wet_level=effects_settings.get('reverb_wet', 0.3)
        )
        print(f"   After reverb: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_echo', False):
        processed_audio = apply_echo(
            processed_audio, sr,
            delay=effects_settings.get('echo_delay', 0.3),
            decay=effects_settings.get('echo_decay', 0.5)
        )
        print(f"   After echo: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_pitch', False):
        processed_audio = apply_pitch_shift(
            processed_audio, sr,
            semitones=effects_settings.get('pitch_semitones', 0)
        )
        print(f"   After pitch: max={np.max(np.abs(processed_audio)):.4f}")
    
    # Advanced effects (new)
    if effects_settings.get('enable_noise_reduction', False):
        processed_audio = apply_noise_reduction(processed_audio, sr)
        print(f"   After noise reduction: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_equalizer', False):
        eq_bands = {
            'sub_bass': effects_settings.get('eq_sub_bass', 0),
            'bass': effects_settings.get('eq_bass', 0),
            'low_mid': effects_settings.get('eq_low_mid', 0),
            'mid': effects_settings.get('eq_mid', 0),
            'high_mid': effects_settings.get('eq_high_mid', 0),
            'presence': effects_settings.get('eq_presence', 0),
            'brilliance': effects_settings.get('eq_brilliance', 0)
        }
        print(f"   EQ settings: {eq_bands}")
        print(f"   Before EQ: max={np.max(np.abs(processed_audio)):.4f}")
        processed_audio = apply_equalizer(processed_audio, sr, eq_bands)
        print(f"   After EQ: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_spatial', False):
        processed_audio = apply_spatial_audio(
            processed_audio, sr,
            azimuth=effects_settings.get('spatial_azimuth', 0),
            elevation=effects_settings.get('spatial_elevation', 0),
            distance=effects_settings.get('spatial_distance', 1.0)
        )
        print(f"   After spatial: max={np.max(np.abs(processed_audio)):.4f}")
    
    # Background mixing (applied last)
    if effects_settings.get('enable_background', False):
        processed_audio = mix_with_background(
            processed_audio, sr,
            background_path=effects_settings.get('background_path', ''),
            bg_volume=effects_settings.get('bg_volume', 0.3),
            speech_volume=effects_settings.get('speech_volume', 1.0),
            fade_in=effects_settings.get('bg_fade_in', 1.0),
            fade_out=effects_settings.get('bg_fade_out', 1.0)
        )
        print(f"   After background: max={np.max(np.abs(processed_audio)):.4f}")
    
    print(f"🎵 Audio effects processing complete. Final max: {np.max(np.abs(processed_audio)):.4f}")
    return processed_audio

# --- Export Functions ---
EXPORT_DIR = "exports"

def ensure_export_dir():
    """Ensure the export directory exists."""
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        print(f"📁 Created export directory: {os.path.abspath(EXPORT_DIR)}")

def export_audio(audio_data, sr, format_type="wav", quality="high"):
    """Export audio in different formats and qualities to export folder."""
    try:
        ensure_export_dir()
        
        # Normalize audio to prevent clipping and fuzzy sound
        audio_normalized = np.copy(audio_data)
        
        # Check if audio needs normalization
        max_val = np.abs(audio_normalized).max()
        if max_val > 0:
            # Normalize to 85% of full scale to prevent clipping
            audio_normalized = audio_normalized / max_val * 0.85
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"chatterbox_export_{timestamp}"
        
        if format_type == "wav":
            filename = f"{base_filename}.wav"
            filepath = os.path.join(EXPORT_DIR, filename)
            
            # Implement real quality differences with supported data types
            if quality == "high":
                # High: 16-bit, original sample rate (best quality)
                export_sr = sr
                audio_int = (audio_normalized * 32767).astype(np.int16)
                print(f"🎵 WAV High Quality: 16-bit, {export_sr} Hz")
                
            elif quality == "medium":
                # Medium: 16-bit, half sample rate (smaller file, good quality)
                from scipy import signal
                export_sr = sr // 2
                # Resample to lower sample rate
                num_samples = int(len(audio_normalized) * export_sr / sr)
                audio_resampled = signal.resample(audio_normalized, num_samples)
                audio_int = (audio_resampled * 32767).astype(np.int16)
                print(f"🎵 WAV Medium Quality: 16-bit, {export_sr} Hz (resampled)")
                
            else:  # low
                # Low: 16-bit, quarter sample rate, reduced bit depth simulation
                from scipy import signal
                export_sr = sr // 4
                # Resample to much lower sample rate
                num_samples = int(len(audio_normalized) * export_sr / sr)
                audio_resampled = signal.resample(audio_normalized, num_samples)
                # Simulate lower bit depth by quantizing to fewer levels
                audio_quantized = np.round(audio_resampled * 4096) / 4096  # 12-bit simulation
                audio_int = (audio_quantized * 32767).astype(np.int16)
                print(f"🎵 WAV Low Quality: 16-bit (12-bit simulation), {export_sr} Hz (resampled)")
            
            wavfile.write(filepath, export_sr, audio_int)
            print(f"✅ WAV exported: {filepath}")
            return filepath
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        return None

def handle_export(audio_data, export_quality):
    """Handle audio export and return status message."""
    if audio_data is None:
        return "❌ No audio to export. Generate audio first!"
    
    try:
        # Extract sample rate and audio array from the tuple
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, audio_array = audio_data
        else:
            return "❌ Invalid audio data format"
        
        print(f"🎵 Exporting audio: WAV format, {export_quality} quality")
        print(f"📊 Audio stats: {len(audio_array)} samples, {sr} Hz, max level: {np.abs(audio_array).max():.3f}")
        
        # Export the audio (only WAV now)
        export_path = export_audio(audio_array, sr, "wav", export_quality)
        
        if export_path:
            relative_path = os.path.relpath(export_path)
            file_size = os.path.getsize(export_path) / 1024 / 1024  # Size in MB
            
            # Show quality info
            if export_quality == "high":
                quality_info = " (16-bit, full sample rate - best quality)"
            elif export_quality == "medium":
                quality_info = " (16-bit, half sample rate - balanced)"
            else:
                quality_info = " (16-bit, quarter sample rate - smallest file)"
            
            return f"✅ Audio exported successfully!\n📁 Saved to: {relative_path}\n📊 File size: {file_size:.1f} MB{quality_info}"
        else:
            return "❌ Export failed"
            
    except Exception as e:
        return f"❌ Export error: {str(e)}"

def clear_hf_credentials():
    """Clear any cached Hugging Face credentials that might cause 401 errors."""
    try:
        # Clear environment variables
        os.environ.pop('HF_TOKEN', None)
        os.environ.pop('HUGGINGFACE_HUB_TOKEN', None)
        
        # Try to logout using CLI
        subprocess.run([sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli', 'logout'], 
                      capture_output=True, check=False)
        print("🔧 Cleared Hugging Face credentials")
        return True
    except Exception as e:
        print(f"⚠️ Could not clear HF credentials: {e}")
        return False

def get_or_load_model():
    """Loads the multilingual ChatterBox model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MULTILINGUAL_MODEL
    
    # Check if multilingual TTS is available
    if not MULTILINGUAL_AVAILABLE:
        raise RuntimeError("❌ Multilingual TTS not available. Please install latest chatterbox package.")
    
    # Load multilingual model
    if MULTILINGUAL_MODEL is None:
        print("🌍 Multilingual model not loaded, initializing...")
        try:
            # Check if we have local downloaded models
            existing_files, missing_files = check_multilingual_models_exist()
            if not missing_files:
                print(f"📁 Using local multilingual models from: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")
                # Load from local directory if all files exist
                MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_local(MODEL_DOWNLOAD_DIR, device=DEVICE)
            else:
                print("🌐 Loading multilingual model from Hugging Face...")
                print("💡 Tip: Download models locally using the download section for faster loading")
                MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
            
            if hasattr(MULTILINGUAL_MODEL, 'to') and str(MULTILINGUAL_MODEL.device) != DEVICE:
                MULTILINGUAL_MODEL.to(DEVICE)
            print(f"🌍 Multilingual model loaded successfully. Internal device: {getattr(MULTILINGUAL_MODEL, 'device', 'N/A')}")
        except Exception as e:
            error_str = str(e)
            # Check if it's a 401 authentication error
            if "401" in error_str and "Unauthorized" in error_str:
                print("🔧 Detected 401 authentication error. Clearing credentials and retrying...")
                clear_hf_credentials()
                try:
                    # Retry loading the model
                    existing_files, missing_files = check_multilingual_models_exist()
                    if not missing_files:
                        print(f"📁 Retrying with local multilingual models from: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")
                        MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_local(MODEL_DOWNLOAD_DIR, device=DEVICE)
                    else:
                        print("🌐 Retrying multilingual model from Hugging Face...")
                        print("💡 Tip: Download models locally using the download section for faster loading")
                        MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
                    
                    if hasattr(MULTILINGUAL_MODEL, 'to') and str(MULTILINGUAL_MODEL.device) != DEVICE:
                        MULTILINGUAL_MODEL.to(DEVICE)
                    print(f"🌍 Multilingual model loaded successfully after clearing credentials. Internal device: {getattr(MULTILINGUAL_MODEL, 'device', 'N/A')}")
                except Exception as retry_error:
                    print(f"❌ Error loading multilingual model after retry: {retry_error}")
                    raise
            else:
                print(f"❌ Error loading multilingual model: {e}")
                raise
    
    return MULTILINGUAL_MODEL

# Skip model loading at startup - models will be loaded on-demand
print("🚀 App ready - multilingual models will be loaded when needed")
print("💡 Use the download section to get multilingual models for 23-language support")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def split_text_into_chunks(text: str, max_chunk_length: int = 300) -> list[str]:
    """
    Splits text into chunks that respect sentence boundaries and word limits.
    
    Args:
        text: The input text to split
        max_chunk_length: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_length:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 2 > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by commas or phrases
                if len(sentence) > max_chunk_length:
                    # Split by commas or natural breaks
                    parts = re.split(r'[,;]+', sentence)
                    for part in parts:
                        part = part.strip()
                        if len(current_chunk) + len(part) + 2 > max_chunk_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (", " if current_chunk else "") + part
                else:
                    current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_or_load_model():
    """Get the current multilingual model or load it if not available"""
    global MULTILINGUAL_MODEL
    
    if MULTILINGUAL_MODEL is None:
        try:
            print("🌍 Loading multilingual model on-demand...")
            
            # Check if local models exist
            existing_files, missing_files = check_multilingual_models_exist()
            
            if not missing_files:
                print(f"📁 Using local multilingual models from: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")
                MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_local(MODEL_DOWNLOAD_DIR, device=DEVICE)
            else:
                print("🌐 Loading multilingual model from Hugging Face...")
                MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
            
            if hasattr(MULTILINGUAL_MODEL, 'to') and str(MULTILINGUAL_MODEL.device) != DEVICE:
                MULTILINGUAL_MODEL.to(DEVICE)
            
            print("✅ Multilingual model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading multilingual model: {e}")
            return None
    
    return MULTILINGUAL_MODEL

def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size_input: int,
    # Language selection
    language_id_input: str = "en",
    # Basic audio effects parameters
    enable_reverb: bool = False,
    reverb_room: float = 0.3,
    reverb_damping: float = 0.5,
    reverb_wet: float = 0.3,
    enable_echo: bool = False,
    echo_delay: float = 0.3,
    echo_decay: float = 0.5,
    enable_pitch: bool = False,
    pitch_semitones: float = 0,
    # Advanced audio effects parameters
    enable_noise_reduction: bool = False,
    enable_equalizer: bool = False,
    eq_sub_bass: float = 0,
    eq_bass: float = 0,
    eq_low_mid: float = 0,
    eq_mid: float = 0,
    eq_high_mid: float = 0,
    eq_presence: float = 0,
    eq_brilliance: float = 0,
    enable_spatial: bool = False,
    spatial_azimuth: float = 0,
    spatial_elevation: float = 0,
    spatial_distance: float = 1.0,
    enable_background: bool = False,
    background_path: str = "",
    bg_volume: float = 0.3,
    speech_volume: float = 1.0,
    bg_fade_in: float = 1.0,
    bg_fade_out: float = 1.0,
    # Conversation mode parameters
    conversation_mode: bool = False,
    conversation_script: str = "",
    conversation_pause: float = 0.8,
    speaker_transition_pause: float = 0.3,
    # Speaker settings (will be passed as JSON string)
    speaker_settings_json: str = "{}",
) -> tuple[tuple[int, np.ndarray], tuple[int, np.ndarray], str]:
    """
    Generates TTS audio using the multilingual ChatterBox model.
    Returns: (audio_output, waveform_data, waveform_info)
    """
    # Load the multilingual model
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("Multilingual TTS model is not loaded.")
    
    # Show which language is being used
    language_name = SUPPORTED_LANGUAGES.get(language_id_input, f"Unknown ({language_id_input})")
    print(f"🌍 Using multilingual model for {language_name} generation")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    # Prepare effects settings
    effects_settings = {
        # Basic effects
        'enable_reverb': enable_reverb,
        'reverb_room': reverb_room,
        'reverb_damping': reverb_damping,
        'reverb_wet': reverb_wet,
        'enable_echo': enable_echo,
        'echo_delay': echo_delay,
        'echo_decay': echo_decay,
        'enable_pitch': enable_pitch,
        'pitch_semitones': pitch_semitones,
        # Advanced effects
        'enable_noise_reduction': enable_noise_reduction,
        'enable_equalizer': enable_equalizer,
        'eq_sub_bass': eq_sub_bass,
        'eq_bass': eq_bass,
        'eq_low_mid': eq_low_mid,
        'eq_mid': eq_mid,
        'eq_high_mid': eq_high_mid,
        'eq_presence': eq_presence,
        'eq_brilliance': eq_brilliance,
        'enable_spatial': enable_spatial,
        'spatial_azimuth': spatial_azimuth,
        'spatial_elevation': spatial_elevation,
        'spatial_distance': spatial_distance,
        'enable_background': enable_background,
        'background_path': background_path,
        'bg_volume': bg_volume,
        'speech_volume': speech_volume,
        'bg_fade_in': bg_fade_in,
        'bg_fade_out': bg_fade_out,
    }

    # Check if conversation mode is enabled
    if conversation_mode and conversation_script.strip():
        print("🎭 Conversation mode activated")
        
        # Parse speaker settings from JSON
        try:
            import json
            speaker_settings = json.loads(speaker_settings_json) if speaker_settings_json else {}
        except:
            speaker_settings = {}
        
        # Generate conversation
        audio_result, info_or_error = generate_conversation_audio(
            conversation_script,
            speaker_settings,
            conversation_pause_duration=conversation_pause,
            speaker_transition_pause=speaker_transition_pause,
            effects_settings=effects_settings if any(effects_settings[key] for key in ['enable_reverb', 'enable_echo', 'enable_pitch', 'enable_noise_reduction', 'enable_equalizer', 'enable_spatial', 'enable_background']) else None,
            use_multilingual=True,
            language_id=language_id_input,
            current_model=current_model
        )
        
        if audio_result is None:
            # Error occurred
            waveform_info = info_or_error
            return (current_model.sr, np.zeros(1000)), (current_model.sr, np.zeros(1000)), waveform_info
        else:
            # Success
            sr, final_audio = audio_result
            waveform_info = format_conversation_info(info_or_error)
            return (sr, final_audio), (sr, final_audio), waveform_info
        
    else:
        # Original single voice mode
        # Split text into manageable chunks
        text_chunks = split_text_into_chunks(text_input, max_chunk_length=chunk_size_input)
        
        if len(text_chunks) == 1:
            print(f"Generating audio for text: '{text_input[:50]}...'")
        else:
            print(f"Generating audio in {len(text_chunks)} chunks for text: '{text_input[:50]}...'")
        
        audio_chunks = []
        
        # Temporarily suppress ALL warnings during generation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i, chunk in enumerate(text_chunks):
                if len(text_chunks) > 1:
                    print(f"Processing chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...'")
                
                # Generate audio with multilingual parameters
                wav = current_model.generate(
                    chunk,
                    audio_prompt_path=audio_prompt_path_input,
                    language_id=language_id_input,
                    exaggeration=exaggeration_input,
                    temperature=temperature_input,
                    cfg_weight=cfgw_input,
                )
                
                audio_chunks.append(wav.squeeze(0).numpy())
        
        # Concatenate all audio chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            # Add small silence between chunks for natural flow
            silence_samples = int(current_model.sr * 0.05)  # 0.05 second silence
            silence = np.zeros(silence_samples)
            
            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:  # Don't add silence after the last chunk
                    concatenated_chunks.append(silence)
            
            final_audio = np.concatenate(concatenated_chunks)
        
        # Apply audio effects (both basic and advanced)
        if any(effects_settings[key] for key in ['enable_reverb', 'enable_echo', 'enable_pitch', 'enable_noise_reduction', 'enable_equalizer', 'enable_spatial', 'enable_background']):
            print("Applying audio effects...")
            final_audio = apply_audio_effects(final_audio, current_model.sr, effects_settings)
        
        # Create audio output tuple
        audio_output = (current_model.sr, final_audio)
        
        # Generate waveform analysis
        print("🔍 Performing waveform analysis...")
        _, stats = create_waveform_visualization(final_audio, current_model.sr)
        waveform_info = format_waveform_info(stats)
        
        print("Audio generation and analysis complete.")
        return audio_output, audio_output, waveform_info

# Voice preset management functions for Gradio
def save_current_preset(preset_name, exaggeration, temperature, cfg_weight, chunk_size, ref_audio):
    """Save current settings as a preset including the reference audio."""
    if not preset_name.strip():
        return "❌ Please enter a preset name", gr.update()
    
    settings = {
        'exaggeration': exaggeration,
        'temperature': temperature,
        'cfg_weight': cfg_weight,
        'chunk_size': chunk_size,
        'ref_audio': ref_audio or ''
    }
    
    if save_voice_preset(preset_name.strip(), settings):
        updated_choices = get_preset_names()
        if ref_audio:
            return f"✅ Voice preset '{preset_name}' saved successfully with custom voice!", gr.update(choices=updated_choices, value=None)
        else:
            return f"✅ Preset '{preset_name}' saved (no custom voice audio)", gr.update(choices=updated_choices, value=None)
    else:
        return "❌ Failed to save preset", gr.update()

def load_selected_preset(preset_name):
    """Load selected preset and return its settings including reference audio."""
    if not preset_name:
        return "Please select a preset", None, None, None, None, None
    
    preset = load_voice_preset(preset_name)
    if preset:
        # Use the saved audio path, not the original
        ref_audio_path = preset.get('ref_audio_path', '')
        
        return (
            f"✅ Loaded voice preset '{preset_name}'" + (" with custom voice" if ref_audio_path else ""),
            preset['exaggeration'],
            preset['temperature'], 
            preset['cfg_weight'],
            preset['chunk_size'],
            ref_audio_path if ref_audio_path and os.path.exists(ref_audio_path) else None
        )
    else:
        return "❌ Failed to load preset", None, None, None, None, None

def delete_selected_preset(preset_name):
    """Delete selected preset and its audio file."""
    if not preset_name:
        return "Please select a preset to delete", gr.update()
    
    if delete_voice_preset(preset_name):
        updated_choices = get_preset_names()
        return f"✅ Voice preset '{preset_name}' deleted (including audio file)", gr.update(choices=updated_choices, value=None)
    else:
        return "❌ Failed to delete preset", gr.update()

def refresh_preset_dropdown():
    """Refresh the preset dropdown with current presets."""
    choices = get_preset_names()
    return gr.update(choices=choices, value=None)

# Standard Gradio theme - no custom CSS needed
def get_custom_css():
    """Using Gradio's default theme styling."""
    return ""

def create_waveform_visualization(audio_data, sr=22050):
    """Create enhanced waveform visualization data."""
    if audio_data is None:
        return None, "No audio data available"
    
    try:
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Calculate waveform statistics
        duration = len(audio_data) / sr
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))
        peak_db = 20 * np.log10(max_amplitude + 1e-10)
        rms_db = 20 * np.log10(rms_level + 1e-10)
        
        # Detect zero crossings for pitch estimation
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        estimated_pitch = zero_crossings / (2 * duration)
        
        # Create frequency analysis
        fft_data = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/sr)
        magnitude = np.abs(fft_data)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_freq = abs(freqs[dominant_freq_idx])
        
        stats = {
            "duration": f"{duration:.2f}s",
            "sample_rate": f"{sr} Hz",
            "samples": f"{len(audio_data):,}",
            "max_amplitude": f"{max_amplitude:.4f}",
            "peak_level": f"{peak_db:.1f} dB",
            "rms_level": f"{rms_db:.1f} dB",
            "estimated_pitch": f"{estimated_pitch:.1f} Hz",
            "dominant_freq": f"{dominant_freq:.1f} Hz",
            "dynamic_range": f"{peak_db - rms_db:.1f} dB"
        }
        
        return audio_data, stats
        
    except Exception as e:
        return None, f"Error analyzing audio: {str(e)}"

def format_waveform_info(stats):
    """Format waveform statistics for display."""
    if isinstance(stats, str):
        return stats
    
    info_text = f"""
🎵 Audio Analysis:
• Duration: {stats['duration']} | Sample Rate: {stats['sample_rate']} | Samples: {stats['samples']}
• Peak Level: {stats['peak_level']} | RMS Level: {stats['rms_level']} | Dynamic Range: {stats['dynamic_range']}  
• Estimated Pitch: {stats['estimated_pitch']} | Dominant Frequency: {stats['dominant_freq']}
• Max Amplitude: {stats['max_amplitude']}
    """.strip()
    
    return info_text

def analyze_audio_waveform(audio_data):
    """Analyze waveform data and return formatted information."""
    if audio_data is None:
        return "No audio data available for analysis. Generate audio first."
    
    try:
        # Extract sample rate and audio array from the tuple
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, audio_array = audio_data
        else:
            return "Invalid audio data format for analysis."
        
        print(f"🔍 Analyzing audio: {len(audio_array)} samples at {sr} Hz")
        
        # Perform waveform analysis
        _, stats = create_waveform_visualization(audio_array, sr)
        
        return format_waveform_info(stats)
        
    except Exception as e:
        return f"Error analyzing waveform: {str(e)}"

# Initialize CSS (none needed for standard theme)
initial_css = get_custom_css()

# --- Voice Conversation System ---
def parse_conversation_script(script_text):
    """Parse conversation script in Speaker: Text format."""
    try:
        lines = script_text.strip().split('\n')
        conversation = []
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains speaker designation (Speaker: Text format)
            if ':' in line and not line.startswith(' '):
                # Save previous speaker's text if exists
                if current_speaker and current_text:
                    conversation.append({
                        'speaker': current_speaker,
                        'text': current_text.strip()
                    })
                
                # Parse new speaker line
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_speaker = parts[0].strip()
                    current_text = parts[1].strip()
                else:
                    # Invalid format, treat as continuation
                    current_text += " " + line
            else:
                # Continuation of previous speaker's text
                current_text += " " + line
        
        # Add the last speaker's text
        if current_speaker and current_text:
            conversation.append({
                'speaker': current_speaker,
                'text': current_text.strip()
            })
        
        return conversation, None
        
    except Exception as e:
        return [], f"Error parsing conversation: {str(e)}"

def get_speaker_names_from_script(script_text):
    """Extract unique speaker names from conversation script."""
    conversation, error = parse_conversation_script(script_text)
    if error:
        return []
    
    speakers = list(set([item['speaker'] for item in conversation]))
    return sorted(speakers)

def generate_conversation_audio(
    conversation_script,
    speaker_settings,
    conversation_pause_duration=0.8,
    speaker_transition_pause=0.3,
    effects_settings=None,
    use_multilingual=False,
    language_id="en",
    current_model=None
):
    """Generate a complete conversation with multiple voices."""
    try:
        print("🎭 Starting conversation generation...")
        
        # Parse the conversation script
        conversation, parse_error = parse_conversation_script(conversation_script)
        if parse_error:
            return None, f"❌ Script parsing error: {parse_error}"
        
        if not conversation:
            return None, "❌ No valid conversation found in script"
        
        print(f"📝 Parsed {len(conversation)} conversation lines")
        
        # Use the passed model
        if current_model is None:
            return None, "❌ TTS model not available"
        
        conversation_audio_chunks = []
        conversation_info = []
        
        # Generate audio for each conversation line
        for i, line in enumerate(conversation):
            speaker = line['speaker']
            text = line['text']
            
            print(f"🗣️ Generating line {i+1}/{len(conversation)}: {speaker}")
            
            # Get speaker settings
            if speaker not in speaker_settings:
                return None, f"❌ No settings found for speaker '{speaker}'"
            
            settings = speaker_settings[speaker]
            
            # Suppress warnings during generation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Generate audio for this line
                try:
                    # Generate with multilingual parameters using speaker's language
                    speaker_language = settings.get('language', language_id)  # Use speaker's language or fallback to global
                    wav = current_model.generate(
                        text,
                        audio_prompt_path=settings.get('ref_audio', ''),
                        language_id=speaker_language,
                        exaggeration=settings.get('exaggeration', 0.5),
                        temperature=settings.get('temperature', 0.8),
                        cfg_weight=settings.get('cfg_weight', 0.5),
                    )
                    
                    line_audio = wav.squeeze(0).numpy()
                    
                    # Apply individual speaker effects if specified
                    if effects_settings:
                        line_audio = apply_audio_effects(line_audio, current_model.sr, effects_settings)
                    
                    conversation_audio_chunks.append(line_audio)
                    conversation_info.append({
                        'speaker': speaker,
                        'text': text[:50] + ('...' if len(text) > 50 else ''),
                        'duration': len(line_audio) / current_model.sr,
                        'samples': len(line_audio),
                        'language': speaker_language
                    })
                    
                except Exception as gen_error:
                    return None, f"❌ Error generating audio for {speaker}: {str(gen_error)}"
        
        # Combine all audio with proper timing
        print("🎵 Combining conversation audio with proper timing...")
        
        # Calculate pause durations
        conversation_pause_samples = int(current_model.sr * conversation_pause_duration)
        transition_pause_samples = int(current_model.sr * speaker_transition_pause)
        
        final_audio_parts = []
        previous_speaker = None
        
        for i, (audio_chunk, info) in enumerate(zip(conversation_audio_chunks, conversation_info)):
            current_speaker = info['speaker']
            
            # Add audio chunk
            final_audio_parts.append(audio_chunk)
            
            # Add pause after each line (except the last one)
            if i < len(conversation_audio_chunks) - 1:
                next_speaker = conversation_info[i + 1]['speaker']
                
                # Different pause duration based on speaker change
                if current_speaker != next_speaker:
                    # Speaker transition - longer pause
                    pause_samples = conversation_pause_samples
                else:
                    # Same speaker continuing - shorter pause
                    pause_samples = transition_pause_samples
                
                pause_audio = np.zeros(pause_samples)
                final_audio_parts.append(pause_audio)
        
        # Concatenate all parts
        final_conversation_audio = np.concatenate(final_audio_parts)
        
        # Create conversation summary
        total_duration = len(final_conversation_audio) / current_model.sr
        unique_speakers = len(set([info['speaker'] for info in conversation_info]))
        
        # Collect language information
        languages_used = list(set([info['language'] for info in conversation_info]))
        
        summary = {
            'total_lines': len(conversation),
            'unique_speakers': unique_speakers,
            'total_duration': total_duration,
            'speakers': list(set([info['speaker'] for info in conversation_info])),
            'languages_used': languages_used,
            'conversation_info': conversation_info
        }
        
        print(f"✅ Conversation generated: {len(conversation)} lines, {unique_speakers} speakers, {total_duration:.1f}s")
        
        return (current_model.sr, final_conversation_audio), summary
        
    except Exception as e:
        return None, f"❌ Conversation generation error: {str(e)}"

def format_conversation_info(summary):
    """Format conversation summary for display."""
    if isinstance(summary, str):
        return summary
    
    try:
        # Format languages used
        languages_used = summary.get('languages_used', [])
        language_names = [SUPPORTED_LANGUAGES.get(lang, lang) for lang in languages_used]
        
        info_text = f"""
🎭 Conversation Summary:
• Total Lines: {summary['total_lines']} | Speakers: {summary['unique_speakers']} | Duration: {summary['total_duration']:.1f}s
• Speakers: {', '.join(summary['speakers'])}
• Languages: {', '.join(language_names)} ({len(languages_used)} language{'s' if len(languages_used) != 1 else ''})

📝 Line Breakdown:"""
        
        for i, line_info in enumerate(summary['conversation_info'], 1):
            speaker = line_info['speaker']
            text_preview = line_info['text']
            duration = line_info['duration']
            language = line_info.get('language', 'en')
            language_name = SUPPORTED_LANGUAGES.get(language, language)
            info_text += f"\n{i:2d}. {speaker} ({language_name}): \"{text_preview}\" ({duration:.1f}s)"
        
        return info_text.strip()
        
    except Exception as e:
        return f"Error formatting conversation info: {str(e)}"

def update_speaker_settings_from_presets(speakers_text, current_settings_json):
    """Update speaker settings by loading from available presets."""
    try:
        current_settings = json.loads(current_settings_json) if current_settings_json else {}
        available_presets = load_voice_presets()
        
        # Get speaker names from detected speakers text
        speakers = []
        if "Found" in speakers_text and "speakers:" in speakers_text:
            lines = speakers_text.split('\n')[1:]  # Skip first line
            speakers = [line.replace('• ', '').strip() for line in lines if line.strip()]
        
        # Try to match speakers with available presets
        for speaker in speakers:
            if speaker in available_presets:
                preset = available_presets[speaker]
                current_settings[speaker] = {
                    'ref_audio': preset.get('ref_audio_path', ''),
                    'exaggeration': preset.get('exaggeration', 0.5),
                    'temperature': preset.get('temperature', 0.8),
                    'cfg_weight': preset.get('cfg_weight', 0.5)
                }
                print(f"🎭 Auto-loaded preset for speaker '{speaker}'")
        
        return json.dumps(current_settings)
        
    except Exception as e:
        print(f"Error updating speaker settings: {e}")
        return current_settings_json

def setup_speaker_audio_components(script_text):
    """Set up audio components for detected speakers and return visibility updates."""
    speakers = get_speaker_names_from_script(script_text)
    
    # Maximum 5 speakers supported in UI
    audio_components_visibility = [False] * 5
    audio_labels = ["🎤 Speaker Voice"] * 5
    lang_labels = ["🌍 Speaker Language"] * 5
    speaker_controls_visible = False
    
    if speakers:
        speaker_controls_visible = True
        for i, speaker in enumerate(speakers[:5]):  # Limit to 5 speakers
            audio_components_visibility[i] = True
            audio_labels[i] = f"🎤 {speaker}'s Voice"
            lang_labels[i] = f"🌍 {speaker}'s Language"
    
    # Create updates for all audio components
    updates = []
    for i in range(5):
        updates.append(gr.update(
            visible=audio_components_visibility[i],
            label=audio_labels[i],
            value=None  # Clear previous uploads
        ))
    
    # Create updates for all language dropdowns
    for i in range(5):
        updates.append(gr.update(
            visible=audio_components_visibility[i],
            label=lang_labels[i],
            value="en"  # Default to English
        ))
    
    # Add visibility update for the speaker controls row
    updates.append(gr.update(visible=speaker_controls_visible))
    
    # Return speakers list and settings JSON
    updates.append(speakers)
    
    # Initialize speaker settings
    speaker_settings = {}
    available_presets = load_voice_presets()
    
    for speaker in speakers:
        if speaker in available_presets:
            preset = available_presets[speaker]
            speaker_settings[speaker] = {
                'ref_audio': preset.get('ref_audio_path', ''),
                'exaggeration': preset.get('exaggeration', 0.5),
                'temperature': preset.get('temperature', 0.8),
                'cfg_weight': preset.get('cfg_weight', 0.5),
                'language': preset.get('language', 'en')
            }
        else:
            speaker_settings[speaker] = {
                'ref_audio': '',
                'exaggeration': 0.5,
                'temperature': 0.8,
                'cfg_weight': 0.5,
                'language': 'en'
            }
    
    updates.append(json.dumps(speaker_settings))
    
    return updates

def update_speaker_audio_settings(speakers_list, audio1, audio2, audio3, audio4, audio5, current_settings_json):
    """Update speaker settings JSON with uploaded audio files."""
    try:
        current_settings = json.loads(current_settings_json) if current_settings_json else {}
        audio_files = [audio1, audio2, audio3, audio4, audio5]
        
        # Update settings with uploaded audio
        for i, speaker in enumerate(speakers_list[:5]):
            if speaker in current_settings:
                if i < len(audio_files) and audio_files[i]:
                    current_settings[speaker]['ref_audio'] = audio_files[i]
                    print(f"🎤 Updated audio for {speaker}: {audio_files[i]}")
        
        return json.dumps(current_settings)
        
    except Exception as e:
        print(f"Error updating speaker audio settings: {e}")
        return current_settings_json

def update_speaker_language_settings(speakers_list, lang1, lang2, lang3, lang4, lang5, current_settings_json):
    """Update speaker settings JSON with language selections."""
    try:
        current_settings = json.loads(current_settings_json) if current_settings_json else {}
        languages = [lang1, lang2, lang3, lang4, lang5]
        
        # Update settings with selected languages
        for i, speaker in enumerate(speakers_list[:5]):
            if speaker in current_settings:
                if i < len(languages) and languages[i]:
                    current_settings[speaker]['language'] = languages[i]
                    print(f"🌍 Updated language for {speaker}: {languages[i]}")
        
        return json.dumps(current_settings)
        
    except Exception as e:
        print(f"Error updating speaker language settings: {e}")
        return current_settings_json

with gr.Blocks(title="🌍 Chatterbox TTS Pro - Multilingual") as demo:
    # Header
    if MULTILINGUAL_AVAILABLE:
        multilingual_status = "🌍 Multilingual Ready"
        status_message = "**Supports 23 languages**: Arabic, Chinese, French, German, Spanish, and many more! Download models below to get started."
    else:
        multilingual_status = "⚠️ No Models Available"
        status_message = "**Please install chatterbox-tts**: `pip install chatterbox-tts` or download models below."
    
    gr.Markdown(
        f"""
        # 🎭 Chatterbox TTS Pro {multilingual_status}
        **Advanced Multilingual Text-to-Speech with Voice Presets, Audio Effects & Export Options**
        
        Generate high-quality speech from text with reference audio styling, save your favorite voice presets, apply professional audio effects, and export in multiple formats!
        {status_message}
        
        **🚀 Getting Started**: Models are loaded on-demand. Use the download section below to get multilingual models for 23-language support.
        """
    )
    
    # Initialize presets on app startup
    initial_presets = get_preset_names()
    print(f"🚀 App starting with {len(initial_presets)} presets available")
    
    # Model Download Section - closed by default to reduce UI clutter
    with gr.Accordion("📥 Download Multilingual Models", open=False):
        gr.Markdown("""
        ### 🌍 Multilingual Model Download Manager
        **Manual Download Required**: Download the required model files for 23-language support. These models enable text-to-speech in Arabic, Chinese, French, German, Spanish, and many more languages.
        
        **No Auto-Download**: Models are only downloaded when you click the download button below.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                model_status_display = gr.Textbox(
                    label="📋 Model Files Status",
                    value=check_model_files_status(),
                    interactive=False,
                    lines=4
                )
                
                download_progress_display = gr.Textbox(
                    label="📊 Download Progress",
                    value=get_download_status(),
                    interactive=False,
                    lines=2
                )

                model_loading_status = gr.Textbox(
                    label="🚀 Model Loading Status",
                    value=check_model_loaded_status(),
                    interactive=False,
                    lines=2
                )
            
                with gr.Column(scale=1):
                    with gr.Group():
                        check_models_btn = gr.Button(
                            "🔍 Check Model Files",
                            variant="secondary",
                            size="sm"
                        )

                        download_models_btn = gr.Button(
                            "📥 Download Multilingual Models",
                            variant="primary",
                            size="lg"
                        )

                        load_model_btn = gr.Button(
                            "🚀 Load Model into Memory",
                            variant="secondary",
                            size="lg",
                            visible=False
                        )

                        refresh_status_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary",
                            size="sm"
                        )
                
                gr.Markdown("""
                **Model Files:**
                - `Cangjie5_TC` - Chinese tokenizer
                - `conds` - Conditional embeddings
                - `mtl_tokenizer` - Multilingual tokenizer
                - `s3gen` - Speech generator
                - `t3_23lang` - Text-to-speech model
                - `ve` - Voice encoder
                
                **Total size:** ~2-4 GB
                """)
    

    
    with gr.Row():
        with gr.Column(scale=2):
            # Main text input
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="📝 Text to synthesize (any length supported)",
                max_lines=10,
                placeholder="Enter your text here..."
            )
            
            # Language selection
            with gr.Group():
                gr.Markdown("### 🌍 Language Selection")
                with gr.Row():
                    language_dropdown = gr.Dropdown(
                        choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                        value="en",
                        label="🗣️ Target Language",
                        info="Select the language for text-to-speech generation"
                    )
                
                # Show language availability info
                if MULTILINGUAL_AVAILABLE:
                    gr.Markdown("*✅ Multilingual support available - Download models above to use 23 languages*")
                else:
                    gr.Markdown("*❌ No TTS models available - Please install chatterbox-tts or download models*")
                
                gr.Markdown("*💡 Models are loaded on-demand when you generate speech*")
            
            # Reference audio
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="🎤 Reference Audio File (Optional)",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
            )
            
            # Voice Conversation Mode Section
            with gr.Accordion("🎭 Voice Conversation Mode", open=False):
                gr.Markdown("### 🗣️ Multi-Voice Conversation Generator")
                gr.Markdown("*Generate conversations between multiple speakers with different voices*")
                
                conversation_mode = gr.Checkbox(
                    label="🎭 Enable Conversation Mode",
                    value=False,
                    info="Switch to conversation mode to generate multi-speaker dialogues"
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        conversation_script = gr.Textbox(
                            label="📝 Conversation Script",
                            placeholder="""Enter conversation in this format:

Alice: Hello there! How are you doing today?
Bob: I'm doing great, thanks for asking! How about you?
Alice: I'm wonderful! I just got back from vacation.
Bob: That sounds amazing! Where did you go?
Alice: I went to Japan. It was absolutely incredible!""",
                            lines=8,
                            info="Format: 'SpeakerName: Text' - Each line should start with speaker name followed by colon"
                        )
                        
                        # Conversation timing controls
                        with gr.Row():
                            conversation_pause = gr.Slider(
                                0.2, 2.0, step=0.1,
                                label="🔇 Speaker Change Pause (s)",
                                value=0.8,
                                info="Pause duration when speakers change"
                            )
                            speaker_transition_pause = gr.Slider(
                                0.1, 1.0, step=0.1,
                                label="⏸️ Same Speaker Pause (s)",
                                value=0.3,
                                info="Pause when same speaker continues"
                            )
                    
                    with gr.Column(scale=1):
                        # Speaker detection and management
                        detected_speakers = gr.Textbox(
                            label="🔍 Detected Speakers",
                            interactive=False,
                            lines=3,
                            info="Speakers found in your script will appear here"
                        )
                        
                        parse_script_btn = gr.Button(
                            "🔍 Analyze Script",
                            size="sm",
                            variant="secondary"
                        )
                        
                        conversation_help = gr.Markdown("""
                        **📋 Script Format Guide:**
                        - Each line: `SpeakerName: Dialogue text`
                        - Speaker names are case-sensitive
                        - Use consistent speaker names
                        - Multi-line dialogue will be joined
                        
                        **🎭 Example:**
                        ```
                        Alice: Hello Bob!
                        Bob: Hi Alice, how's it going?
                        Alice: Great! I wanted to tell you about my trip.
                        ```
                        """)

                # Dynamic speaker management section
                with gr.Group():
                    gr.Markdown("### 🎤 Speaker Voice Configuration")
                    gr.Markdown("*Configure voice settings for each speaker in your conversation*")
                    
                    # Speaker configuration will be dynamically generated
                    speaker_config_area = gr.HTML(
                        value="<p style='text-align: center; color: #666; padding: 20px;'>📝 Enter a conversation script above and click 'Analyze Script' to configure speaker voices</p>"
                    )
                    
                    # Dynamic speaker controls container
                    with gr.Row(visible=False) as speaker_controls_row:
                        with gr.Column():
                            # These will be dynamically created based on detected speakers
                            speaker_audio_1 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 1 Voice",
                                visible=False
                            )
                            speaker_audio_2 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 2 Voice",
                                visible=False
                            )
                            speaker_audio_3 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 3 Voice",
                                visible=False
                            )
                            speaker_audio_4 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 4 Voice",
                                visible=False
                            )
                            speaker_audio_5 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 5 Voice",
                                visible=False
                            )
                        
                        with gr.Column():
                            # Language selection for each speaker
                            speaker_lang_1 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 1 Language",
                                visible=False
                            )
                            speaker_lang_2 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 2 Language",
                                visible=False
                            )
                            speaker_lang_3 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 3 Language",
                                visible=False
                            )
                            speaker_lang_4 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 4 Language",
                                visible=False
                            )
                            speaker_lang_5 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 5 Language",
                                visible=False
                            )
                    
                    # Hidden components to store speaker configurations
                    speaker_settings_json = gr.Textbox(
                        value="{}",
                        visible=False,
                        label="Speaker Settings JSON"
                    )
                    
                    # Store current speakers for reference
                    current_speakers = gr.State([])
                    
                    # Dynamic speaker controls (will be created programmatically)
                    dynamic_speaker_controls = gr.State({})
                
                # Conversation generation controls
                with gr.Row():
                    generate_conversation_btn = gr.Button(
                        "🎭 Generate Conversation",
                        variant="primary",
                        size="lg"
                    )
                    
                    clear_conversation_btn = gr.Button(
                        "🗑️ Clear Script",
                        variant="secondary",
                        size="sm"
                    )

            # Voice Presets Section
            with gr.Group():
                gr.Markdown("### 🎭 Voice Presets")
                gr.Markdown("*Save your complete voice setup including the reference audio file*")
                
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=initial_presets,
                        label="Select Voice Preset",
                        value=None,
                        interactive=True
                    )
                    preset_name_input = gr.Textbox(
                        label="New Voice Preset Name",
                        placeholder="Enter preset name...",
                        scale=1
                    )
                
                with gr.Row():
                    load_preset_btn = gr.Button("📥 Load Voice", size="sm")
                    save_preset_btn = gr.Button("💾 Save Voice", size="sm", variant="secondary")
                    delete_preset_btn = gr.Button("🗑️ Delete Voice", size="sm", variant="stop")
                    refresh_btn = gr.Button("🔄 Refresh List", size="sm", variant="secondary")
                
                preset_status = gr.Textbox(label="Status", interactive=False, visible=True)
                
                # Show current preset file locations
                with gr.Accordion("📁 File Locations", open=False):
                    preset_path_info = gr.Textbox(
                        label="Presets config saved to",
                        value=os.path.abspath(PRESETS_FILE),
                        interactive=False
                    )
                    audio_path_info = gr.Textbox(
                        label="Voice audio files saved to",
                        value=os.path.abspath(PRESETS_AUDIO_DIR),
                        interactive=False
                    )

            # Main controls
            with gr.Row():
                exaggeration = gr.Slider(
                    0.25, 2, step=.05, 
                    label="🎭 Exaggeration (Neutral = 0.5)", 
                    value=.5,
                    info="Higher values = more dramatic speech"
                )
                cfg_weight = gr.Slider(
                    0.2, 1, step=.05, 
                    label="⚡ CFG/Pace", 
                    value=0.5,
                    info="Controls generation speed vs quality"
                )

            with gr.Accordion("🔧 Advanced Settings", open=False):
                with gr.Row():
                    chunk_size = gr.Slider(
                        100, 400, step=25, 
                        label="📄 Chunk size (characters per chunk)", 
                        value=300,
                        info="Smaller = more consistent, larger = fewer seams"
                    )
                    temp = gr.Slider(
                        0.05, 5, step=.05, 
                        label="🌡️ Temperature", 
                        value=.8,
                        info="Higher = more creative/varied"
                    )
                    seed_num = gr.Number(
                        value=0, 
                        label="🎲 Random seed (0 for random)",
                        info="Use same seed for reproducible results"
                    )

            # Audio Effects Section
            with gr.Accordion("🎵 Audio Effects & Processing", open=False):
                gr.Markdown("### Professional audio effects and advanced processing")
                
                # Basic Effects Tab
                with gr.Tab("🎭 Basic Effects"):
                    with gr.Row():
                        with gr.Column():
                            enable_reverb = gr.Checkbox(label="🏛️ Enable Reverb", value=False)
                            reverb_room = gr.Slider(0.1, 1.0, step=0.1, label="Room Size", value=0.3, visible=True)
                            reverb_damping = gr.Slider(0.1, 1.0, step=0.1, label="Damping", value=0.5, visible=True)
                            reverb_wet = gr.Slider(0.1, 0.8, step=0.1, label="Reverb Amount", value=0.3, visible=True)
                        
                        with gr.Column():
                            enable_echo = gr.Checkbox(label="🔊 Enable Echo", value=False)
                            echo_delay = gr.Slider(0.1, 1.0, step=0.1, label="Echo Delay (s)", value=0.3, visible=True)
                            echo_decay = gr.Slider(0.1, 0.9, step=0.1, label="Echo Decay", value=0.5, visible=True)
                        
                        with gr.Column():
                            enable_pitch = gr.Checkbox(label="🎼 Enable Pitch Shift", value=False)
                            pitch_semitones = gr.Slider(-12, 12, step=1, label="Pitch (semitones)", value=0, visible=True)

                # Advanced Processing Tab
                with gr.Tab("🔧 Advanced Processing"):
                    with gr.Row():
                        with gr.Column():
                            # Noise Reduction
                            enable_noise_reduction = gr.Checkbox(label="🧹 Enable Noise Reduction", value=False)
                            gr.Markdown("*Automatically clean up reference audio*")
                        
                        with gr.Column():
                            # Audio Equalizer
                            enable_equalizer = gr.Checkbox(label="🎛️ Enable Equalizer", value=False)
                            gr.Markdown("*Fine-tune frequency bands*")
                    
                    # Equalizer Controls (shown when enabled)
                    with gr.Group():
                        gr.Markdown("#### 🎛️ 7-Band Equalizer (dB)")
                        with gr.Row():
                            eq_sub_bass = gr.Slider(-12, 12, step=1, label="Sub Bass\n(20-60 Hz)", value=0)
                            eq_bass = gr.Slider(-12, 12, step=1, label="Bass\n(60-200 Hz)", value=0)
                            eq_low_mid = gr.Slider(-12, 12, step=1, label="Low Mid\n(200-500 Hz)", value=0)
                            eq_mid = gr.Slider(-12, 12, step=1, label="Mid\n(500-2k Hz)", value=0)
                        with gr.Row():
                            eq_high_mid = gr.Slider(-12, 12, step=1, label="High Mid\n(2k-4k Hz)", value=0)
                            eq_presence = gr.Slider(-12, 12, step=1, label="Presence\n(4k-8k Hz)", value=0)
                            eq_brilliance = gr.Slider(-12, 12, step=1, label="Brilliance\n(8k-20k Hz)", value=0)

                # 3D Spatial Audio Tab
                with gr.Tab("🎧 3D Spatial Audio"):
                    enable_spatial = gr.Checkbox(label="🎧 Enable 3D Spatial Positioning", value=False)
                    gr.Markdown("*Position voices in 3D space for immersive experiences*")
                    
                    with gr.Row():
                        with gr.Column():
                            spatial_azimuth = gr.Slider(
                                -180, 180, step=5, 
                                label="🧭 Azimuth (degrees)", 
                                value=0,
                                info="Left-Right positioning (-180° to 180°)"
                            )
                            spatial_elevation = gr.Slider(
                                -90, 90, step=5, 
                                label="📐 Elevation (degrees)", 
                                value=0,
                                info="Up-Down positioning (-90° to 90°)"
                            )
                        with gr.Column():
                            spatial_distance = gr.Slider(
                                0.1, 5.0, step=0.1, 
                                label="📏 Distance", 
                                value=1.0,
                                info="Distance from listener (0.1 = close, 5.0 = far)"
                            )
                            gr.Markdown("""
                            **Quick Presets:**
                            - Center: Az=0°, El=0°, Dist=1.0
                            - Left: Az=-90°, El=0°, Dist=1.0  
                            - Right: Az=90°, El=0°, Dist=1.0
                            - Above: Az=0°, El=45°, Dist=1.0
                            - Distant: Az=0°, El=0°, Dist=3.0
                            """)

                # Background Music Mixer Tab
                with gr.Tab("🎵 Background Music"):
                    enable_background = gr.Checkbox(label="🎵 Enable Background Music/Ambience", value=False)
                    gr.Markdown("*Blend generated speech with background audio*")
                    
                    with gr.Row():
                        with gr.Column():
                            background_path = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="🎼 Background Audio File"
                            )
                            gr.Markdown("*Upload music, ambience, or sound effects*")
                            
                        with gr.Column():
                            bg_volume = gr.Slider(
                                0.0, 1.0, step=0.05, 
                                label="🔊 Background Volume", 
                                value=0.3,
                                info="Volume of background audio"
                            )
                            speech_volume = gr.Slider(
                                0.0, 2.0, step=0.05, 
                                label="🗣️ Speech Volume", 
                                value=1.0,
                                info="Volume of generated speech"
                            )
                    
                    with gr.Row():
                        bg_fade_in = gr.Slider(
                            0.0, 5.0, step=0.1, 
                            label="📈 Fade In (seconds)", 
                            value=1.0,
                            info="Background fade-in duration"
                        )
                        bg_fade_out = gr.Slider(
                            0.0, 5.0, step=0.1, 
                            label="📉 Fade Out (seconds)", 
                            value=1.0,
                            info="Background fade-out duration"
                        )
                    
                    gr.Markdown("""
                    **Background Audio Tips:**
                    - **Music**: Use instrumental tracks, keep volume low (0.2-0.4)
                    - **Ambience**: Nature sounds, room tone, atmospheric audio
                    - **SFX**: Sound effects that complement the speech content
                    - **Looping**: Short audio files will automatically loop to match speech length
                    """)

            # Generate button
            run_btn = gr.Button(
                "🚀 Generate Speech", 
                variant="primary", 
                size="lg"
            )

        with gr.Column(scale=1):
            # Enhanced Audio Output with Waveform Visualization
            with gr.Group():
                gr.Markdown("### 🎵 Generated Audio & Waveform Analysis")
                
                # Main audio output
                audio_output = gr.Audio(
                    label="🎵 Generated Audio",
                    show_download_button=True,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#4CAF50",
                        waveform_progress_color="#45a049",
                        show_recording_waveform=True,
                        skip_length=5,
                        sample_rate=22050
                    )
                )
                
                # Waveform analysis info
                waveform_info = gr.Textbox(
                    label="📊 Audio Analysis",
                    lines=4,
                    interactive=False,
                    placeholder="Audio analysis will appear here after generation..."
                )
                
                # Waveform controls
                with gr.Row():
                    analyze_btn = gr.Button("📊 Analyze Audio", size="sm", variant="secondary")
                    clear_analysis_btn = gr.Button("🗑️ Clear Analysis", size="sm", variant="stop")
            
            # Export Options
            with gr.Accordion("📤 Export Options", open=False):
                gr.Markdown("### 📥 Export your audio as WAV files")
                gr.Markdown("*Download your generated speech in different qualities and formats*")
                
                with gr.Row():
                    with gr.Column():
                        export_quality = gr.Radio(
                            choices=[
                                ("🎵 High Quality (16-bit, full sample rate)", "high"),
                                ("⚖️ Medium Quality (16-bit, half sample rate)", "medium"), 
                                ("💾 Low Quality (16-bit, quarter sample rate)", "low")
                            ],
                            value="high",
                            label="Export Quality",
                            info="Choose quality vs file size trade-off"
                        )
                        
                    with gr.Column():
                        gr.Markdown("""
                        **Quality Guide:**
                        - **High**: Best quality, largest file (~3-5MB/min)
                        - **Medium**: Good quality, balanced size (~1-2MB/min)
                        - **Low**: Smallest file, acceptable quality (~0.5MB/min)
                        """)
                
                with gr.Row():
                    export_btn = gr.Button(
                        "📥 Export Audio as WAV", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    gr.HTML("<div style='width: 20px;'></div>")  # Spacer
                
                export_status = gr.Textbox(
                    label="📋 Export Status", 
                    interactive=False,
                    placeholder="Export status will appear here...",
                    lines=2
                )
                
                # Show export folder location
                with gr.Accordion("📁 Export Location", open=False):
                    export_path_info = gr.Textbox(
                        label="Files exported to",
                        value=os.path.abspath(EXPORT_DIR),
                        interactive=False,
                        info="All exported files are saved to this directory"
                    )
                    gr.Markdown("**Note**: Files are automatically named with timestamp for easy organization.")

            # Tips and info
            with gr.Accordion("💡 Tips & Best Practices", open=False):
                gr.Markdown(
                    """
                💡 Pro Tips
                - **On-demand loading**: Models load only when you generate speech (no startup downloads)
                - **Long text**: Automatically chunked for best quality
                - **Voice presets**: Save your favorite combinations
                - **Model download**: Use the download section at the top to get multilingual models (~2-4GB)
                - **Multilingual mode**: Enable for 23 language support (Arabic, Chinese, French, Spanish, etc.)
                - **Language matching**: Match reference audio language to target language for best results
                - **Conversation mode**: Generate multi-speaker dialogues with different voices
                - **Basic effects**: Add reverb for space, echo for depth, pitch shift for character
                - **Noise reduction**: Automatically cleans up noisy reference audio
                - **Equalizer**: Boost presence (4-8kHz) for clarity, adjust bass for warmth
                - **3D spatial**: Create immersive positioning for podcasts/games
                - **Background music**: Keep volume low (0.2-0.4) for speech clarity
                - **Export**: Download in different qualities
                - **Waveform**: Analyze audio characteristics and quality
                
                ### 🎯 Best Practices
                - Use clear reference audio (3-10 seconds)
                - Keep exaggeration moderate (0.3-0.8)
                - Try temperature 0.6-1.0 for natural speech
                - Use smaller chunks for consistent quality
                - Apply noise reduction to poor quality reference audio
                - Use EQ to enhance specific voice characteristics
                - Position voices spatially for immersive experiences
                - Analyze waveform to understand audio quality
                
                ### 🌍 Multilingual Best Practices
                - **Language matching**: Reference audio should match target language
                - **CFG weight**: Lower (0.3) if reference has different language accent
                - **Supported languages**: 23 languages from Arabic to Chinese
                - **Quality**: Multilingual model maintains high quality across all languages
                - **Mixed conversations**: Each speaker in conversation mode can use a different language
                
                ### 🎭 Voice Conversation Mode Guide
                - **Script Format**: Use `SpeakerName: Dialogue text` format
                - **Individual Audio**: Upload different reference audio for each speaker
                - **Per-Speaker Languages**: Each speaker can use a different language (23 languages supported)
                - **Auto-Detection**: Speakers are automatically detected and audio upload slots appear
                - **Timing Control**: Adjust pauses between speakers and within speaker turns
                - **Voice Variety**: Each speaker can have completely different voice characteristics
                - **Consistent Names**: Keep speaker names exactly the same throughout
                - **Preset Integration**: Presets with matching speaker names load automatically
                - **Natural Flow**: Longer pauses for speaker changes, shorter for continuations
                - **Max Speakers**: Supports up to 5 different speakers per conversation
                
                ### 🎵 Audio Effects Guide
                - **Reverb**: Simulates room acoustics (church, hall, studio)
                - **Echo**: Adds depth and space to voice
                - **Pitch**: Change voice character (±12 semitones)
                - **Noise Reduction**: Clean background noise from reference
                - **Equalizer**: Shape frequency response for desired tone
                - **3D Spatial**: Position voice in 3D space for VR/AR
                - **Background**: Mix with music/ambience for atmosphere
                - **Waveform Analysis**: Understand audio characteristics and quality
                
                ### 📝 Conversation Examples
                ```
                Alice: Welcome to our podcast! I'm Alice.
                Bob: And I'm Bob. Today we're discussing AI.
                Alice: It's fascinating how quickly it's evolving.
                Bob: Absolutely! The possibilities are endless.
                ```
                
                ```
                Narrator: In a distant galaxy...
                Hero: I must save the princess!
                Villain: You'll never defeat me!
                Hero: We'll see about that!
                ```
                """
                )
        
        # Video Dubbing Tab (new functionality)
        with gr.Tab("🎬 Video Dubbing", id="dubbing_tab"):
            gr.Markdown("""
            # 🎬 AI Video Dubbing System
            **Automatically dub videos into multiple languages with AI translation and voice synthesis**
            
            Upload a video, select target languages, and get professionally dubbed versions with synchronized audio!
            """)
            
            # API Management Section
            with gr.Accordion("🔑 API Management", open=True):
                gr.Markdown("""
                ### 🧠 Gemini API Configuration
                Add your Google Gemini API keys for translation. Multiple keys enable load balancing and redundancy.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        api_name_input = gr.Textbox(
                            label="🏷️ API Name (Optional)",
                            placeholder="My Gemini API",
                            info="Give your API a friendly name"
                        )
                        api_key_input = gr.Textbox(
                            label="🔑 Gemini API Key",
                            placeholder="Enter your Google Gemini API key...",
                            type="password",
                            info="Get your API key from Google AI Studio"
                        )
                    
                    with gr.Column(scale=1):
                        add_api_btn = gr.Button(
                            "➕ Add API Key",
                            variant="primary",
                            size="lg"
                        )
                        
                        api_status_display = gr.Textbox(
                            label="📊 API Status",
                            value=get_api_status(),
                            interactive=False,
                            lines=3
                        )
                
                gr.Markdown("""
                **💡 API Tips:**
                - Get free API keys from [Google AI Studio](https://aistudio.google.com/app/apikey)
                - Multiple keys provide redundancy and higher rate limits
                - Keys are used in round-robin fashion for load balancing
                """)
            
            # Main Dubbing Interface
            with gr.Row():
                with gr.Column(scale=2):
                    # Video Upload
                    video_input = gr.Video(
                        label="🎬 Upload Video File"
                    )
                    gr.Markdown("*Supported formats: MP4, AVI, MOV, MKV*")
                    
                    # Language Selection
                    with gr.Group():
                        gr.Markdown("### 🌍 Target Languages")
                        target_languages = gr.CheckboxGroup(
                            choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                            label="🗣️ Select Languages for Dubbing",
                            info="Choose one or more languages for dubbing",
                            value=["es", "fr"]  # Default to Spanish and French
                        )
                    
                    # Reference Audio Upload
                    with gr.Accordion("🎤 Reference Audio (Optional)", open=False):
                        gr.Markdown("""
                        ### 🎙️ Upload Reference Audio for Voice Cloning
                        Upload a reference audio file to clone the voice for each language.
                        This will make the dubbed voice sound more like the reference speaker.
                        """)
                        
                        reference_audio_upload = gr.Audio(
                            label="🎤 Upload Reference Audio (10-30 seconds recommended)",
                            type="filepath"
                        )
                        
                        gr.Markdown("""
                        **💡 Reference Audio Tips:**
                        - Use clear, high-quality audio (no background noise)
                        - 10-30 seconds of speech is optimal
                        - The same reference will be used for all selected languages
                        - Leave empty to use default voice for each language
                        """)
                    
                    # Translation Customization
                    with gr.Accordion("🎨 Translation Style", open=False):
                        custom_prompt = gr.Textbox(
                            label="✨ Custom Translation Prompt",
                            placeholder="e.g., 'Translate in a formal tone' or 'Use casual, friendly language'",
                            lines=3,
                            info="Optional: Add specific instructions for translation style"
                        )
                        
                        gr.Markdown("""
                        **Style Examples:**
                        - "Translate in a formal, professional tone"
                        - "Use casual, conversational language"
                        - "Maintain the original humor and personality"
                        - "Adapt cultural references for the target audience"
                        """)
                    
                    # Processing Button
                    process_dubbing_btn = gr.Button(
                        "🚀 Start Dubbing Process",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output and Status
                    with gr.Group():
                        gr.Markdown("### 📤 Dubbed Video Output")
                        
                        dubbed_video_output = gr.Video(
                            label="🎬 Dubbed Video"
                        )
                        gr.Markdown("*Preview of the first dubbed video*")
                        
                        dubbing_status = gr.Textbox(
                            label="📊 Processing Status",
                            lines=8,
                            interactive=False,
                            placeholder="Upload a video and click 'Start Dubbing Process' to begin..."
                        )
                    
                    # Processing Steps Info
                    with gr.Accordion("🔄 Processing Pipeline", open=False):
                        gr.Markdown("""
                        ### 🛠️ Dubbing Process Steps:
                        
                        1. **🎵 Audio Extraction**
                           - Extract audio track from video using FFmpeg
                        
                        2. **🎤 Speech Recognition**
                           - Transcribe audio with timestamps using Parakeet TDT
                           - Generate segment-level timing information
                        
                        3. **🌍 Translation**
                           - Translate each segment using Gemini AI
                           - Preserve timing and context information
                           - Apply custom style instructions
                        
                        4. **🗣️ Voice Synthesis**
                           - Generate TTS for each translated segment
                           - Adjust audio speed to match original timing
                           - Use Chatterbox multilingual TTS
                        
                        5. **🎬 Video Assembly**
                           - Combine new audio with original video
                           - Maintain video quality and synchronization
                           - Export final dubbed videos
                        """)
            
            # Real-time Progress Tracking - Always Visible
            gr.Markdown("## 📊 Live Processing Progress")
            
            # Step 1: Transcription Progress
            with gr.Group():
                gr.Markdown("### 🎤 Step 1: Video Transcription")
                with gr.Row():
                    transcription_status = gr.Textbox(
                        label="📝 Transcription Status",
                        value="⏳ Waiting for video upload...",
                        interactive=False,
                        lines=2
                    )
                    parakeet_model_status = gr.Textbox(
                        label="🤖 Parakeet Model Status", 
                        value="💤 Model not loaded",
                        interactive=False,
                        lines=2
                    )
                
                transcript_display = gr.HTML(
                    label="📋 Timestamped Transcript",
                    value="<div style='padding: 20px; text-align: center; color: #666;'>Transcript will appear here after processing...</div>"
                )
            
            # Step 2: Translation Progress  
            with gr.Group():
                gr.Markdown("### 🌍 Step 2: Smart Chunked Translation")
                with gr.Row():
                    with gr.Column():
                        translation_status = gr.Textbox(
                            label="🔄 Translation Status",
                            value="⏳ Waiting for transcription...",
                            interactive=False,
                            lines=3
                        )
                    with gr.Column():
                        chunk_progress = gr.Textbox(
                            label="📦 Chunk Progress",
                            value="📊 No chunks created yet",
                            interactive=False,
                            lines=3
                        )
                
                # Translation results for each language
                translation_results = gr.HTML(
                    label="📝 Translation Results by Language",
                    value="<div style='padding: 20px; text-align: center; color: #666;'>Translation results will appear here...</div>"
                )
            
            # Step 3: TTS Generation Progress
            with gr.Group():
                gr.Markdown("### 🗣️ Step 3: Voice Synthesis (Chatterbox TTS)")
                with gr.Row():
                    with gr.Column():
                        tts_status = gr.Textbox(
                            label="🎙️ TTS Generation Status",
                            value="⏳ Waiting for translation...",
                            interactive=False,
                            lines=3
                        )
                    with gr.Column():
                        audio_processing_status = gr.Textbox(
                            label="🔧 Audio Processing",
                            value="⏳ Speed adjustment pending...",
                            interactive=False,
                            lines=3
                        )
                
                tts_results = gr.HTML(
                    label="🎵 Generated Audio Segments",
                    value="<div style='padding: 20px; text-align: center; color: #666;'>TTS results will appear here...</div>"
                )
            
            # Step 4: Final Video Assembly
            with gr.Group():
                gr.Markdown("### 🎬 Step 4: Video Assembly & Output")
                with gr.Row():
                    video_assembly_status = gr.Textbox(
                        label="🎞️ Video Assembly Status",
                        value="⏳ Waiting for audio generation...",
                        interactive=False,
                        lines=2
                    )
                    final_output_status = gr.Textbox(
                        label="📁 Output Files Status",
                        value="📋 No files generated yet",
                        interactive=False,
                        lines=2
                    )
                
                # Final output files display
                output_files_display = gr.HTML(
                    label="📥 Download Links",
                    value="<div style='padding: 20px; text-align: center; color: #666;'>Download links will appear here...</div>"
                )
            
            # Requirements and Tips
            with gr.Accordion("📋 Requirements & Tips", open=False):
                gr.Markdown("""
                ### 🔧 System Requirements:
                - **Parakeet ASR**: `pip install nemo_toolkit[asr]`
                - **Gemini API**: Valid API key from Google AI Studio
                - **FFmpeg**: For video/audio processing
                - **Chatterbox Models**: Download multilingual models above
                
                ### 💡 Best Practices:
                - **Video Quality**: Use clear audio with minimal background noise
                - **Language Support**: 23 languages supported for dubbing
                - **File Size**: Larger videos take longer to process
                - **API Limits**: Multiple API keys help with rate limiting
                - **Custom Prompts**: Use specific style instructions for better results
                
                ### 🎯 Optimal Results:
                - Videos with clear speech work best
                - Avoid videos with heavy music or sound effects
                - Single speaker videos produce most accurate results
                - Consider video length for processing time
                """)

    # Hidden components for waveform analysis
    waveform_data = gr.State(None)

    # Language dropdown is always visible now
    
    # Model download event handlers
    check_models_btn.click(
        fn=check_model_files_status,
        outputs=[model_status_display]
    )
    
    def start_download():
        """Start the download and return initial status."""
        download_models_async()
        return "📥 Starting download..."
    
    def download_complete_handler():
        """Handle download completion and update UI."""
        return (
            get_download_status(),
            check_model_loaded_status(),
            gr.update(visible=should_show_load_button())
        )
    
    download_models_btn.click(
        fn=start_download,
        outputs=[download_progress_display]
    )
    
    def load_model_and_update_status():
        """Load model and return status updates."""
        status_msg, success = load_model_manually()
        return status_msg, gr.update(visible=not success)
    
    load_model_btn.click(
        fn=load_model_and_update_status,
        outputs=[model_loading_status, load_model_btn]
    )
    
    def refresh_all_status():
        """Refresh all status displays and button visibility."""
        return (
            check_model_files_status(),
            get_download_status(),
            check_model_loaded_status(),
            gr.update(visible=should_show_load_button())
        )
    
    refresh_status_btn.click(
        fn=refresh_all_status,
        outputs=[model_status_display, download_progress_display, model_loading_status, load_model_btn]
    )
    
    # Auto-refresh download progress every 2 seconds when downloading
    def auto_refresh_download_status():
        status = download_status["status"]
        if status == "downloading":
            return get_download_status()
        return gr.update()
    
    # Initialize status displays on app load
    demo.load(
        fn=lambda: (check_model_files_status(), get_download_status(), check_model_loaded_status(), gr.update(visible=should_show_load_button())),
        outputs=[model_status_display, download_progress_display, model_loading_status, load_model_btn]
    )

    # Event handlers
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text, ref_wav, exaggeration, temp, seed_num, cfg_weight, chunk_size,
            language_dropdown,
            enable_reverb, reverb_room, reverb_damping, reverb_wet,
            enable_echo, echo_delay, echo_decay,
            enable_pitch, pitch_semitones,
            enable_noise_reduction, enable_equalizer, eq_sub_bass, eq_bass, eq_low_mid, eq_mid, eq_high_mid, eq_presence, eq_brilliance,
            enable_spatial, spatial_azimuth, spatial_elevation, spatial_distance,
            enable_background, background_path, bg_volume, speech_volume, bg_fade_in, bg_fade_out,
            conversation_mode, conversation_script, conversation_pause, speaker_transition_pause, speaker_settings_json
        ],
        outputs=[audio_output, waveform_data, waveform_info],
    )
    
    # Conversation mode event handlers
    parse_script_btn.click(
        fn=setup_speaker_audio_components,
        inputs=[conversation_script],
        outputs=[
            speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
            speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
            speaker_controls_row, current_speakers, speaker_settings_json
        ]
    )
    
    clear_conversation_btn.click(
        fn=lambda: (
            "",  # Clear conversation script
            gr.update(visible=False, value=None),  # speaker_audio_1
            gr.update(visible=False, value=None),  # speaker_audio_2
            gr.update(visible=False, value=None),  # speaker_audio_3
            gr.update(visible=False, value=None),  # speaker_audio_4
            gr.update(visible=False, value=None),  # speaker_audio_5
            gr.update(visible=False, value="en"),  # speaker_lang_1
            gr.update(visible=False, value="en"),  # speaker_lang_2
            gr.update(visible=False, value="en"),  # speaker_lang_3
            gr.update(visible=False, value="en"),  # speaker_lang_4
            gr.update(visible=False, value="en"),  # speaker_lang_5
            gr.update(visible=False),  # speaker_controls_row
            [],  # current_speakers
            "{}"  # speaker_settings_json
        ),
        outputs=[
            conversation_script,
            speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
            speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
            speaker_controls_row, current_speakers, speaker_settings_json
        ]
    )
    
    # Auto-update speaker components when script changes
    conversation_script.change(
        fn=setup_speaker_audio_components,
        inputs=[conversation_script],
        outputs=[
            speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
            speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
            speaker_controls_row, current_speakers, speaker_settings_json
        ]
    )
    
    # Update speaker settings when audio files are uploaded
    for audio_component in [speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5]:
        audio_component.change(
            fn=update_speaker_audio_settings,
            inputs=[
                current_speakers,
                speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
                speaker_settings_json
            ],
            outputs=[speaker_settings_json]
        )
    
    # Update speaker settings when languages are changed
    for lang_component in [speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5]:
        lang_component.change(
            fn=update_speaker_language_settings,
            inputs=[
                current_speakers,
                speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
                speaker_settings_json
            ],
            outputs=[speaker_settings_json]
        )
    
    # Update detected speakers display
    current_speakers.change(
        fn=lambda speakers: f"Found {len(speakers)} speakers:\n" + "\n".join([f"• {speaker}" for speaker in speakers]) if speakers else "No speakers detected",
        inputs=[current_speakers],
        outputs=[detected_speakers]
    )
    
    # Auto-load presets for matching speaker names
    detected_speakers.change(
        fn=update_speaker_settings_from_presets,
        inputs=[detected_speakers, speaker_settings_json],
        outputs=[speaker_settings_json]
    )
    
    # Conversation generation (uses the same function as regular generation)
    generate_conversation_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text, ref_wav, exaggeration, temp, seed_num, cfg_weight, chunk_size,
            language_dropdown,
            enable_reverb, reverb_room, reverb_damping, reverb_wet,
            enable_echo, echo_delay, echo_decay,
            enable_pitch, pitch_semitones,
            enable_noise_reduction, enable_equalizer, eq_sub_bass, eq_bass, eq_low_mid, eq_mid, eq_high_mid, eq_presence, eq_brilliance,
            enable_spatial, spatial_azimuth, spatial_elevation, spatial_distance,
            enable_background, background_path, bg_volume, speech_volume, bg_fade_in, bg_fade_out,
            conversation_mode, conversation_script, conversation_pause, speaker_transition_pause, speaker_settings_json
        ],
        outputs=[audio_output, waveform_data, waveform_info],
    )
    
    # Waveform analysis handlers
    analyze_btn.click(
        fn=lambda audio_data: analyze_audio_waveform(audio_data),
        inputs=[waveform_data],
        outputs=[waveform_info]
    )
    
    clear_analysis_btn.click(
        fn=lambda: "Audio analysis cleared. Generate new audio to analyze.",
        outputs=[waveform_info]
    )
    
    # Preset management
    save_preset_btn.click(
        fn=save_current_preset,
        inputs=[preset_name_input, exaggeration, temp, cfg_weight, chunk_size, ref_wav],
        outputs=[preset_status, preset_dropdown]
    )
    
    load_preset_btn.click(
        fn=load_selected_preset,
        inputs=[preset_dropdown],
        outputs=[preset_status, exaggeration, temp, cfg_weight, chunk_size, ref_wav]
    )
    
    delete_preset_btn.click(
        fn=delete_selected_preset,
        inputs=[preset_dropdown],
        outputs=[preset_status, preset_dropdown]
    )
    
    refresh_btn.click(
        fn=refresh_preset_dropdown,
        inputs=[],
        outputs=[preset_dropdown]
    )

    # Export handler
    export_btn.click(
        fn=handle_export,
        inputs=[audio_output, export_quality],
        outputs=[export_status]
    )
    
    # Dubbing process handler function
    def handle_dubbing_process_realtime(video_file, target_langs, custom_prompt_text, ref_audio):
        """Handle the complete dubbing process with real-time progress tracking"""
        try:
            # Call the enhanced workflow with real-time updates
            return complete_video_dubbing_workflow_with_realtime_updates(
                video_file, target_langs, custom_prompt_text, ref_audio
            )
            
        except Exception as e:
            error_msg = f"❌ Processing Error: {str(e)}"
            error_html = f"<div style='color: red; font-weight: bold;'>{error_msg}</div>"
            
            # Return error state for all outputs
            return (
                None,                   # final_video
                error_msg,             # transcription_status
                "❌ Error occurred",   # parakeet_status
                error_html,            # transcript_html
                error_msg,             # translation_status
                "❌ Process failed",   # chunk_progress
                error_html,            # translation_results
                error_msg,             # tts_status
                "❌ Process failed",   # audio_processing
                error_html,            # tts_results
                error_msg,             # video_assembly
                "❌ Process failed",   # final_output
                error_html,            # output_files
                error_msg              # main_status
            )

    # Dubbing system event handlers
    add_api_btn.click(
        fn=add_gemini_api,
        inputs=[api_name_input, api_key_input],
        outputs=[api_status_display, api_key_input]
    )
    
    # Connect the dubbing process button
    process_dubbing_btn.click(
        fn=handle_dubbing_process_realtime,
        inputs=[video_input, target_languages, custom_prompt, reference_audio_upload],
        outputs=[
            dubbed_video_output,        # final_video
            transcription_status,       # transcription_status
            parakeet_model_status,      # parakeet_status
            transcript_display,         # transcript_html
            translation_status,         # translation_status
            chunk_progress,             # chunk_progress
            translation_results,        # translation_results
            tts_status,                 # tts_status
            audio_processing_status,    # audio_processing
            tts_results,                # tts_results
            video_assembly_status,      # video_assembly
            final_output_status,        # final_output
            output_files_display,       # output_files
            dubbing_status              # main_status
        ]
    )
    

    


demo.launch()