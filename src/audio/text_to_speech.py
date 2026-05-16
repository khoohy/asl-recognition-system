"""
Text-to-Speech Module
Converts recognized ASL signs to spoken English.
"""

import platform
import threading
from queue import Empty, Queue


class TextToSpeechEngine:
    """
    Text-to-Speech engine for real-time audio output.
    Runs TTS in a background thread to avoid blocking inference.
    """
    
    def __init__(self, use_gpt: bool = False):
        """
        Initialize TTS engine.
        
        Args:
            use_gpt: If True, use gTTS (Google Text-to-Speech), else use pyttsx3
        """
        self.use_gpt = use_gpt
        self.tts_queue = Queue()
        self.is_running = False
        self.thread = None
        self.engine = None
        self.gTTS = None
        self._pyttsx3 = None
        self._pythoncom = None
        self._win32_dispatch = None
        self._sapi_voice = None
        self.backend_name = "unavailable"
        self.worker_thread_id = None
        
        try:
            if use_gpt:
                from gtts import gTTS
                self.gTTS = gTTS
                self.backend_name = "gtts"
            else:
                if platform.system().lower() == "windows":
                    try:
                        import pythoncom
                        from win32com.client import Dispatch

                        self._pythoncom = pythoncom
                        self._win32_dispatch = Dispatch
                        self.backend_name = "sapi"
                    except ImportError:
                        pass

                import pyttsx3
                self._pyttsx3 = pyttsx3
                if self.backend_name == "unavailable":
                    self.backend_name = "pyttsx3"
        except ImportError as e:
            print(f"Warning: TTS library not available: {e}")
            self.engine = None

    def _ensure_engine(self):
        """Create the offline TTS engine inside the thread that uses it."""
        if self.use_gpt:
            return None
        if self.backend_name == "sapi":
            if self._sapi_voice is not None:
                return self._sapi_voice
            if self._win32_dispatch is None:
                return None
            self._sapi_voice = self._win32_dispatch("SAPI.SpVoice")
            return self._sapi_voice

        if self.engine is not None:
            return self.engine
        if self._pyttsx3 is None:
            return None

        self.engine = self._pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        return self.engine
    
    def start(self):
        """Start TTS processing thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.thread.start()
        print("TTS engine started")
    
    def _tts_worker(self):
        """Background worker thread for TTS processing."""
        self.worker_thread_id = threading.get_ident()
        com_initialized = False
        if not self.use_gpt and self._pythoncom is not None:
            try:
                self._pythoncom.CoInitialize()
                com_initialized = True
            except Exception as e:
                print(f"TTS COM init error: {e}")
        if not self.use_gpt:
            try:
                self._ensure_engine()
            except Exception as e:
                print(f"TTS engine init error: {e}")
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=1.0)
                if text:
                    self.speak(text)
            except Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}")
        if com_initialized and self._pythoncom is not None:
            try:
                self._pythoncom.CoUninitialize()
            except Exception:
                pass
    
    def speak(self, text: str):
        """
        Speak the given text immediately.
        
        Args:
            text: Text to speak
        """
        if not text:
            return

        if self.use_gpt and self.gTTS is None:
            print(f"[TTS] (unavailable): {text}")
            return

        try:
            if self.use_gpt:
                # Google TTS - requires internet
                tts = self.gTTS(text=text, lang='en', slow=False)
                tts.play()
            else:
                # pyttsx3 - offline
                if threading.get_ident() != self.worker_thread_id:
                    self.enqueue_speech(text)
                    return
                engine = self._ensure_engine()
                if engine is None:
                    print(f"[TTS] (unavailable): {text}")
                    return
                if self.backend_name == "sapi":
                    engine.Speak(str(text))
                else:
                    engine.say(text)
                    engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
            if not self.use_gpt:
                try:
                    if self.engine is not None:
                        self.engine.stop()
                except Exception:
                    pass
                self.engine = None
                self._sapi_voice = None
    
    def enqueue_speech(self, text: str):
        """
        Queue text for asynchronous speech output.
        
        Args:
            text: Text to speak
        """
        if not text:
            return
        self.tts_queue.put(text)
    
    def stop(self):
        """Stop TTS engine and cleanup."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.engine and not self.use_gpt:
            try:
                self.engine.stop()
            except:
                pass
        print("TTS engine stopped")
