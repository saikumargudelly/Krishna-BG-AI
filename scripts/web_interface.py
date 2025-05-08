import sys
import os
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import json
import random
import threading
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Initialize NLTK components
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logging.error(f"Error initializing NLTK components: {str(e)}")
    lemmatizer = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import gradio as gr
import re
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Type checking imports
if TYPE_CHECKING:
    from scripts.rag_manager import RAGManager

# RAG Integration
try:
    from scripts.rag_manager import RAGManager
    rag_enabled = True
    rag_manager = RAGManager()
except ImportError as e:
    logging.warning(f"RAG integration failed: {e}")
    rag_enabled = False
    rag_manager = None
except Exception as e:
    logging.error(f"Unexpected error during RAG initialization: {e}")
    rag_enabled = False
    rag_manager = None

# --- Device selection ---
def get_device():
    """Get the appropriate device for model execution."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()

# --- Load Gita Slokas for Context-Aware Augmentation ---
SLOKA_LIST = []
SLOKA_PATH = os.path.join("data", "gita_slokas.json")
try:
    if os.path.exists(SLOKA_PATH):
        with open(SLOKA_PATH, "r", encoding="utf-8") as f:
            SLOKA_LIST = json.load(f)
            logging.info(f"Successfully loaded {len(SLOKA_LIST)} slokas")
            # Log first few slokas to verify structure
            for i, sloka in enumerate(SLOKA_LIST[:3]):
                logging.info(f"Sample sloka {i+1}: Chapter {sloka.get('chapter', '?')}, Verse {sloka.get('verse', '?')}")
                logging.info(f"Sloka text: {sloka.get('sloka', '')}")
                logging.info(f"Meaning: {sloka.get('meaning', '')}")
    else:
        logging.warning(f"Warning: {SLOKA_PATH} not found. Creating default slokas.")
        # Create default slokas if file doesn't exist
        SLOKA_LIST = [
            {
                "chapter": "2",
                "verse": "47",
                "sloka": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
                "meaning": "You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions. Never consider yourself to be the cause of the results of your activities, nor be attached to inaction."
            },
            {
                "chapter": "6",
                "verse": "5",
                "sloka": "आत्मौपम्येन सर्वत्र समं पश्यति योऽर्जुन। सुखं वा यदि वा दु:खं स योगी परमो मत:॥",
                "meaning": "One who sees the same Supreme Lord dwelling equally everywhere, in every living being, such a seer is considered to be a true yogi."
            }
        ]
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(SLOKA_PATH), exist_ok=True)
        # Save default slokas
        with open(SLOKA_PATH, "w", encoding="utf-8") as f:
            json.dump(SLOKA_LIST, f, ensure_ascii=False, indent=2)
except Exception as e:
    logging.error(f"Error loading slokas: {str(e)}")
    SLOKA_LIST = []

# --- Improved Emotion/Tag Detection ---
def detect_emotion(message: str) -> Dict[str, float]:
    """Enhanced emotion detection with confidence scores."""
    msg = message.lower()
    
    emotions = {
        "sad": {
            "keywords": ["sad", "down", "depress", "unhappy", "gloom", "cry", "miser", "sorrow", "hopeless", "tear"],
            "intensity_words": {"very": 1.5, "really": 1.3, "so": 1.2, "extremely": 1.4}
        },
        "angry": {
            "keywords": ["angry", "mad", "frustrate", "irritate", "rage", "annoy", "fume", "resent", "fury"],
            "intensity_words": {"very": 1.5, "really": 1.3, "so": 1.2, "extremely": 1.4}
        },
        "happy": {
            "keywords": ["happy", "joy", "excite", "cheerful", "delight", "content", "grateful", "smile", "thrill", "satisfi"],
            "intensity_words": {"very": 1.5, "really": 1.3, "so": 1.2, "extremely": 1.4}
        },
        "stress": {
            "keywords": ["stress", "anxiety", "worried", "panic", "tense", "overwhelm", "burnout", "pressure", "nervous"],
            "intensity_words": {"very": 1.5, "really": 1.3, "so": 1.2, "extremely": 1.4}
        },
        "love": {
            "keywords": ["love", "affection", "caring", "romance", "heart", "fond", "adore", "devotion", "sweet"],
            "intensity_words": {"very": 1.5, "really": 1.3, "so": 1.2, "extremely": 1.4}
        }
    }
    
    emotion_scores = {emotion: 0.0 for emotion in emotions}
    
    # Check for emotion keywords and their intensity
    for emotion, data in emotions.items():
        for keyword in data["keywords"]:
            if keyword in msg:
                base_score = 1.0
                # Check for intensity modifiers
                for intensity_word, multiplier in data["intensity_words"].items():
                    if f"{intensity_word} {keyword}" in msg:
                        base_score *= multiplier
                emotion_scores[emotion] += base_score
    
    # Normalize scores
    total_score = sum(emotion_scores.values())
    if total_score > 0:
        emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
    
    return emotion_scores

def adapt_response_tone(response: str, emotion_scores: Dict[str, float]) -> str:
    """Adapt response tone based on detected emotions."""
    # Define tone markers for different emotions
    tone_markers = {
        "sad": ["I understand this is difficult", "I'm here for you", "It's okay to feel this way"],
        "angry": ["I hear your frustration", "That must be really upsetting", "I understand your anger"],
        "happy": ["That's wonderful!", "I'm so glad to hear that", "How exciting!"],
        "stress": ["Let's take a deep breath", "I understand this is stressful", "We'll work through this together"],
        "love": ["That's beautiful", "How heartwarming", "That's so touching"]
    }
    
    # Find dominant emotion
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
    if emotion_scores[dominant_emotion] > 0.3:  # Only adapt if emotion is significant
        # Add appropriate tone marker if response doesn't already have one
        if not any(marker.lower() in response.lower() for marker in tone_markers[dominant_emotion]):
            response = f"{random.choice(tone_markers[dominant_emotion])}. {response}"
    
    return response

def add_conversational_elements(response: str) -> str:
    """Add natural conversational elements to the response."""
    # Add appropriate filler words based on response length
    if len(response) > 100:
        fillers = ["You know,", "Actually,", "Well,", "I think,"]
        if not any(filler in response for filler in fillers):
            response = f"{random.choice(fillers)} {response}"
    
    # Add appropriate transition phrases
    transitions = ["By the way,", "Also,", "Speaking of which,", "On that note,"]
    if len(response) > 200 and random.random() < 0.3:
        response = f"{response} {random.choice(transitions)}"
    
    return response

def clean_response(response: str) -> str:
    """Clean and validate response with optimized processing."""
    try:
        # First remove any system prompt echoes
        system_patterns = [
            r"You are Krishna.*?CRITICAL INSTRUCTIONS:.*?(?:\d+\.)+",
            r"<\|system\|>.*?CRITICAL INSTRUCTIONS:.*?(?:\d+\.)+",
            r"You are.*?CRITICAL INSTRUCTIONS:.*?(?:\d+\.)+"
        ]
        for pattern in system_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)
        
        # Stop at any special tag
        stop_tokens = ["<|user|>", "<|system|>", "<|assistant|>"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]
        
        # Remove excessive whitespace and normalize
        response = ' '.join(response.split())
        response = response.strip()
        
        # Ensure response has proper sentence structure
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Only keep the first 2 sentences if response is too long
        if len(response) > 300:  # Reduced from 400
            sentences = re.split(r'(?<=[.!?]) +', response)
            response = " ".join(sentences[:2])  # Reduced from 3
        
        # Validate response quality
        if not response or len(response.strip()) < 10:
            return "Hey bestie! I'd love to hear more about that. Could you tell me a bit more? 💖"
        
        # Check for remaining system prompt fragments
        if any(pattern in response.lower() for pattern in ["you are", "critical instructions", "system prompt"]):
            return "Hey sweetie! I'm having a little trouble with that. Could you try asking me again? 💫"
        
        # Add friendly emoji if none present
        if not any(char in response for char in '😊💛💜💕😃😄🙂😌🤗🥰'):
            response += " 😊"
        
        return response
        
    except Exception as e:
        logging.error(f"Error cleaning response: {str(e)}")
        return "Hey! I'm having a little trouble with that. Could you try asking me again? 💫"

def final_response(message: str, model_response: str) -> str:
    # Dummy: just returns the model response
    return model_response

class ModelManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model = None
            self.tokenizer = None
            self.encoder = None
            self.config = None
            self.lora_config = None
            self.system_prompt = None
            self._initialized = True
            self.model_loaded = False
            self.loading_lock = threading.Lock()
            self.device = get_device()  # Store device as instance variable

    def load_configs(self):
        try:
            with open("config/train_config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
            with open("config/lora_config.json", "r") as f:
                self.lora_config = json.load(f)
            with open("config/system_prompt.yaml", "r") as f:
                self.system_prompt = yaml.safe_load(f).get("system_prompt", "")
        except Exception as e:
            logging.error(f"Failed to load configurations: {e}")
            raise

    def initialize_model(self):
        if self.model_loaded:
            return

        with self.loading_lock:
            if self.model_loaded:  # Double check after acquiring lock
                return

            if not self.config:
                self.load_configs()

            logging.info(f"Loading base model: {self.config['model']['base_model']}")
            
            try:
                # Enable model optimizations
                torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
                
                # Configure model loading
                model_kwargs = {
                    "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                    "low_cpu_mem_usage": True,
                    "use_cache": True,  # Enable KV cache
                }
                
                # Handle device mapping based on available hardware
                if self.device == "mps":
                    model_kwargs["device_map"] = "auto"  # Let the model handle device mapping
                else:
                    model_kwargs["device_map"] = self.device
                
                # Load model with optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["model"]["base_model"],
                    **model_kwargs
                )
                
                # Optimize model
                self.model.eval()  # Set to evaluation mode
                if hasattr(self.model, 'half') and self.device != "cpu":
                    self.model = self.model.half()  # Use half precision
                
                # Load tokenizer with optimizations
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config["model"]["base_model"],
                    padding_side="right",
                    use_fast=True,
                    model_max_length=1024,  # Reduced for faster processing
                    trust_remote_code=True  # Enable remote code for better performance
                )
                
                # Add special tokens
                special_tokens = {
                    "pad_token": "[PAD]",
                    "eos_token": "</s>",
                    "bos_token": "<s>"
                }
                self.tokenizer.add_special_tokens(special_tokens)
                
                # Resize embeddings efficiently
                if hasattr(self.model, 'resize_token_embeddings'):
                    self.model.resize_token_embeddings(len(self.tokenizer))

                # Load LoRA adapter if available
                try:
                    lora_path = self.config["output"]["output_dir"]
                    logging.info(f"Loading LoRA adapter from: {lora_path}")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_path,
                        is_trainable=False,
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                    )
                    logging.info("Successfully loaded LoRA weights.")
                except Exception as e:
                    logging.error(f"Warning: Could not load LoRA weights: {e}")
                    logging.info("Continuing with base model only.")

                # Initialize encoder with optimizations
                try:
                    self.encoder = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        device=self.device
                    )
                    if self.device != "cpu":
                        self.encoder.to(self.device)
                except Exception as e:
                    logging.warning(f"Failed to initialize encoder with optimizations: {e}")
                    # Fallback to basic initialization
                    self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                
                # Move model to device
                if self.device == "mps":
                    # Handle MPS device movement carefully
                    try:
                        self.model.to(self.device)
                    except Exception as e:
                        logging.warning(f"Failed to move model to MPS device: {e}")
                        # Fallback to CPU if MPS fails
                        self.device = "cpu"
                        self.model.to(self.device)
                else:
                    self.model.to(self.device)
                
                # Enable model optimizations
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = True
                
                # Enable torch optimizations
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                elif self.device == "mps":
                    # MPS specific optimizations
                    torch.backends.mps.enable_fallback_to_cpu = True
                
                self.model_loaded = True
                logging.info("Model initialization completed successfully")
                
            except Exception as e:
                logging.error(f"Failed to initialize model: {e}")
                raise

    def get_model(self):
        """Thread-safe model access."""
        if not self.model_loaded:
            self.initialize_model()
        return self.model

    def get_tokenizer(self):
        """Thread-safe tokenizer access."""
        if not self.model_loaded:
            self.initialize_model()
        return self.tokenizer

    def get_encoder(self):
        """Thread-safe encoder access."""
        if not self.model_loaded:
            self.initialize_model()
        return self.encoder

# Initialize model manager
model_manager = ModelManager()

@lru_cache(maxsize=100)
def semantic_relevance(response: str, user_query: str, threshold: float = 0.4) -> bool:
    try:
        if not model_manager.encoder:
            model_manager.initialize_model()
        emb_query = model_manager.encoder.encode(user_query, convert_to_tensor=True)
        emb_resp = model_manager.encoder.encode(response, convert_to_tensor=True)
        sim = float(util.pytorch_cos_sim(emb_query, emb_resp))
        return sim >= threshold
    except Exception:
        return False

def format_chat_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    if system_prompt:
        return f"{system_prompt}\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"

def build_prompt(messages: List[Dict[str, str]], system_prompt: str = None, max_turns: int = 2) -> str:
    """Build a prompt from conversation history with limited context."""
    prompt = []
    
    # Add system prompt first if provided
    if system_prompt:
        prompt.append(f"<|system|>\n{system_prompt}\n")
    
    # Only keep the last max_turns*2 messages
    messages = messages[-max_turns*2:]
    
    # Add conversation history
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role == "user" and content:
            prompt.append(f"<|user|>\n{content}")
        elif role == "assistant" and content:
            prompt.append(f"<|assistant|>\n{content}")
    
    # Always end with assistant cue for generation
    prompt.append("<|assistant|>\n")
    
    return "\n".join(prompt)

def get_word_synonyms(word: str) -> List[str]:
    """Get synonyms for a word using WordNet."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return list(set(synonyms))

def analyze_text_emotions(message: str) -> Dict[str, float]:
    """Enhanced emotion detection using NLP analysis."""
    try:
        logging.info(f"Analyzing emotions for message: {message}")
        
        # Tokenize and tag parts of speech
        tokens = word_tokenize(message.lower())
        pos_tags = pos_tag(tokens)
        logging.info(f"Tokenized message: {tokens}")
        
        # Define emotion categories with their associated parts of speech and keywords
        emotion_patterns = {
            "sad": {
                "keywords": ["sad", "sorrow", "grief", "pain", "suffering", "depressed", "depression", "unhappy", "miserable", "hopeless", "down", "low", "blue"],
                "pos_tags": ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "NN", "NNS"],  # Adjectives, verbs, and nouns
                "weight": 1.0
            },
            "angry": {
                "keywords": ["angry", "furious", "irritated", "annoyed", "frustrated", "enraged", "mad"],
                "pos_tags": ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN"],
                "weight": 1.0
            },
            "happy": {
                "keywords": ["happy", "joy", "delighted", "pleased", "cheerful", "glad", "excited"],
                "pos_tags": ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN"],
                "weight": 1.0
            },
            "stress": {
                "keywords": ["stress", "anxiety", "worried", "tense", "nervous", "overwhelmed", "pressured"],
                "pos_tags": ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "NN", "NNS"],
                "weight": 1.0
            },
            "love": {
                "keywords": ["love", "affection", "care", "fond", "adore", "cherish", "devotion"],
                "pos_tags": ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "NN", "NNS"],
                "weight": 1.0
            }
        }
        
        emotion_scores = {emotion: 0.0 for emotion in emotion_patterns}
        
        # Analyze each word
        for word, pos in pos_tags:
            # Lemmatize the word
            lemma = lemmatizer.lemmatize(word)
            logging.info(f"Analyzing word: {word} (lemma: {lemma}, pos: {pos})")
            
            # Get synonyms
            synonyms = get_word_synonyms(lemma)
            logging.info(f"Synonyms for {lemma}: {synonyms}")
            
            # Check each emotion pattern
            for emotion, pattern in emotion_patterns.items():
                # Check if word matches emotion keywords or their synonyms
                if (lemma in pattern["keywords"] or 
                    any(syn in pattern["keywords"] for syn in synonyms) or
                    word in pattern["keywords"]):
                    
                    # Apply weight based on part of speech
                    if pos in pattern["pos_tags"]:
                        emotion_scores[emotion] += pattern["weight"]
                        logging.info(f"Matched {word} to emotion {emotion} with score {emotion_scores[emotion]}")
                        
                        # Additional weight for intensifiers
                        if pos.startswith("JJR") or pos.startswith("JJS"):
                            emotion_scores[emotion] *= 1.5
                            logging.info(f"Applied intensifier weight for {word}")
                        
                        # Check for negation
                        if any(neg in tokens[max(0, tokens.index(word)-2):tokens.index(word)] 
                              for neg in ["not", "no", "never", "don't", "doesn't"]):
                            emotion_scores[emotion] *= -1
                            logging.info(f"Applied negation for {word}")
        
        # Normalize scores
        total_score = sum(abs(score) for score in emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            logging.info(f"Final emotion scores: {emotion_scores}")
        
        return emotion_scores
        
    except Exception as e:
        logging.error(f"Error in emotion analysis: {str(e)}", exc_info=True)
        return {}

def get_relevant_sloka(message: str) -> Optional[Dict[str, str]]:
    """Get a relevant sloka based on advanced context, fuzzy, and semantic matching from user input."""
    try:
        if not SLOKA_LIST:
            logging.warning("No slokas available")
            return None

        # Initialize sentence transformer if not already done
        if not hasattr(get_relevant_sloka, 'encoder'):
            get_relevant_sloka.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            if device != "cpu":
                get_relevant_sloka.encoder.to(device)

        import difflib

        # Common emotional states and synonyms
        emotion_synonyms = {
            "sad": ["sad", "low", "down", "depressed", "unhappy", "blue", "gloomy", "hopeless", "miserable", "sorrow", "cry", "tearful", "confused", "lost", "pain", "suffering"],
            "confused": ["confused", "uncertain", "lost", "unsure", "doubt", "puzzled", "perplexed", "unclear"],
            "anxious": ["anxious", "worried", "nervous", "tense", "stressed", "afraid", "fearful", "panic", "uneasy"],
            "angry": ["angry", "mad", "frustrated", "irritated", "annoyed", "rage", "resentful"],
            "happy": ["happy", "joyful", "cheerful", "content", "excited", "pleased", "delighted", "satisfied"],
            # Add more as needed
        }
        # Flatten all synonyms for quick lookup
        all_emotion_words = set(word for words in emotion_synonyms.values() for word in words)

        # Extract words from message
        message_words = set(word for word in message.lower().split() if len(word) >= 3)
        # Add synonyms if any word matches an emotion synonym
        expanded_words = set(message_words)
        for word in message_words:
            for key, syns in emotion_synonyms.items():
                if word in syns:
                    expanded_words.update(syns)
        logging.info(f"Expanded context words: {expanded_words}")

        # Score each sloka
        scored_slokas = []
        for sloka in SLOKA_LIST:
            if not isinstance(sloka, dict):
                continue
            sloka_tags = sloka.get('tags', [])
            if isinstance(sloka_tags, str):
                sloka_tags = [tag.strip().lower() for tag in sloka_tags.split(',')]
            elif isinstance(sloka_tags, list):
                sloka_tags = [tag.lower() for tag in sloka_tags]
            else:
                sloka_tags = []
            sloka_text = f"{sloka.get('sloka', '')} {sloka.get('meaning', '')}".lower()

            # Fuzzy and partial matches in tags and text
            tag_matches = set()
            text_matches = set()
            for word in expanded_words:
                # Fuzzy match in tags
                for tag in sloka_tags:
                    if word in tag or tag in word or difflib.SequenceMatcher(None, word, tag).ratio() > 0.7:
                        tag_matches.add(tag)
                # Fuzzy match in text
                for sloka_word in sloka_text.split():
                    if word in sloka_word or sloka_word in word or difflib.SequenceMatcher(None, word, sloka_word).ratio() > 0.7:
                        text_matches.add(word)

            # Semantic similarity
            sloka_embedding = get_relevant_sloka.encoder.encode(sloka_text, convert_to_tensor=True)
            message_embedding = get_relevant_sloka.encoder.encode(message, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(message_embedding, sloka_embedding)[0][0].item()

            # Score: prioritize tag matches, then text, then similarity
            score = len(tag_matches) * 0.5 + len(text_matches) * 0.3 + similarity * 0.4
            scored_slokas.append((score, sloka, tag_matches, text_matches, similarity))

        if not scored_slokas:
            logging.info("No matching slokas found")
            return None

        # Sort by score
        scored_slokas.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sloka, best_tags, best_texts, best_sim = scored_slokas[0]
        logging.info(f"Selected best sloka {best_sloka.get('chapter', '?')}.{best_sloka.get('verse', '?')} with score {best_score:.2f}, tags: {best_tags}, text: {best_texts}, similarity: {best_sim:.2f}")

        # If no strong match, but there is a sloka, return the most semantically similar one
        if best_score < 0.3:
            logging.info("No strong context match, returning most semantically similar sloka.")
            # Already sorted by score, which includes similarity
            return best_sloka
        return best_sloka
    except Exception as e:
        logging.error(f"Error getting relevant sloka: {str(e)}", exc_info=True)
        return None

def generate_response(
    message: str,
    history: List[Tuple[str, str]],
    system_prompt: str = None,
    max_new_tokens: int = 192,  # Further reduced for speed
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    rag_manager: Optional["RAGManager"] = None
) -> str:
    """Generate a response using the model with RAG enhancement."""
    try:
        # Prepare messages with minimal history
        messages = []
        # Only keep last turn for faster processing
        if history:
            last_user, last_assistant = history[-1]
            if last_user.strip():
                messages.append({"role": "user", "content": last_user})
            if last_assistant.strip():
                messages.append({"role": "assistant", "content": last_assistant})
        if message.strip():
            messages.append({"role": "user", "content": message})

        # Check RAG first (faster than full generation)
        if rag_manager:
            relevant_response = rag_manager.get_relevant_response(message, messages)
            if relevant_response:
                return relevant_response

        # Build prompt with minimal context
        prompt = build_prompt(messages, system_prompt, max_turns=1)  # Reduced to 1 turn
        
        # Optimize tokenization with aggressive truncation
        inputs = model_manager.get_tokenizer()(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Further reduced for speed
        )
        
        # Move to device efficiently with non-blocking
        input_ids = inputs["input_ids"].to(model_manager.device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(model_manager.device, non_blocking=True)

        # Generate with optimized parameters for speed
        with torch.no_grad():
            outputs = model_manager.get_model().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=model_manager.get_tokenizer().eos_token_id,
                do_sample=True,
                use_cache=True,
                num_beams=1,  # Single beam for speed
                early_stopping=True,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                typical_p=0.9,
                num_return_sequences=1,
                output_scores=False,
                return_dict_in_generate=False
            )

        # Decode efficiently
        response = model_manager.get_tokenizer().decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Clean and validate response
        cleaned_response = clean_response(response)
        if not cleaned_response or not isinstance(cleaned_response, str):
            return "I apologize, but I encountered an error while generating a response. Please try again."

        return cleaned_response

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

def format_chatml(messages: List[Dict[str, str]], system_prompt: str = None) -> str:
    formatted = []
    if system_prompt:
        formatted.append(f"<|system|>\n{system_prompt}")
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role == "user":
            formatted.append(f"<|user|>\n{content}")
        elif role == "assistant":
            formatted.append(f"<|assistant|>\n{content}")
    # Always end with assistant cue for generation
    formatted.append("<|assistant|>\n")
    return "\n".join(formatted)

def chat(message: str, history: List[Tuple[str, str]], system_prompt: str = None) -> Tuple[str, str]:
    """Handle chat interaction with RAG enhancement."""
    try:
        response = generate_response(
            message=message,
            history=history,
            system_prompt=system_prompt,
            rag_manager=rag_manager
        )
        
        # Sloka augmentation with context matching
        sloka_text = "(No sloka available)"
        try:
            logging.info("Starting context-based sloka retrieval process...")
            
            # Get relevant sloka based on message content
            sloka = get_relevant_sloka(message)
            
            if sloka and isinstance(sloka, dict):
                chapter = sloka.get('chapter', '?')
                verse = sloka.get('verse', '?')
                sloka_text = sloka.get('sloka', '')
                meaning = sloka.get('meaning', '')
                tags = sloka.get('tags', [])
                
                # Format the sloka text with clear structure
                sloka_text = f"""
### Bhagavad Gita Sloka
**Chapter {chapter}, Verse {verse}**

{sloka_text}

**Meaning:**
{meaning}

**Tags:** {', '.join(tags) if isinstance(tags, list) else tags}
"""
                logging.info(f"Successfully generated sloka text for Chapter {chapter}, Verse {verse}")
            else:
                logging.warning("No relevant sloka found")
                sloka_text = "No relevant sloka found for your message. Please try rephrasing your question."
                
        except Exception as e:
            logging.error(f"Error in sloka retrieval: {str(e)}", exc_info=True)
            sloka_text = "Error retrieving sloka. Please try again."
            
        return response, sloka_text
        
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error. Please try again.", "(No sloka available)"

# --- Custom Gradio Blocks UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# KRISH AI Chat\nChat with KRISH AI - Your friendly and helpful assistant!")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Krish's Response")
            user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            send_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            gr.Markdown("### Relevant Bhagavad Gita Sloka")
            sloka_output = gr.Markdown("", elem_id="sloka-augmentation")
            
    def on_send(message, history):
        if not isinstance(history, list):
            history = []
        response, sloka = chat(message, history)
        new_history = history + [(message, response)]
        return new_history, "", sloka
        
    send_btn.click(fn=on_send, inputs=[user_input, chatbot], outputs=[chatbot, user_input, sloka_output], queue=False)
    user_input.submit(fn=on_send, inputs=[user_input, chatbot], outputs=[chatbot, user_input, sloka_output], queue=False)

def find_available_port(start_port: int = 3000, end_port: int = 3010) -> int:
    import socket
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError("No available ports found in the specified range")

if __name__ == "__main__":
    try:
        # Initialize model manager first
        model_manager.load_configs()
        model_manager.initialize_model()
        
        # Launch the interface
        demo.launch(share=True)
    except Exception as e:
        logging.error(f"Failed to start interface: {e}")
        sys.exit(1)