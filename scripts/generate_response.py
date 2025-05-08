import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from scripts.utils import format_chatml

def load_config():
    with open("config/train_config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        padding_side="right",
        use_fast=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        config["output"]["output_dir"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

# --- RAG Integration for Contextual Responses ---
try:
    from scripts.rag_manager import RAGManager
    rag_enabled = True
    rag_manager = RAGManager()
except Exception as e:
    rag_enabled = False
    rag_manager = None

def ensure_human_touch(response):
    # Add a friendly emoji if none present
    if not any(char in response for char in '😊💛💜💕😃😄🙂😌🤗🥰'): 
        response += " 😊"
    # If too short or generic, add a friendly phrase
    if len(response.strip()) < 20:
        response += " Let me know more, bestie! 💛"
    return response

def is_spiritual_or_emotion_query(query):
    # Check for keywords related to spirituality, emotion, or sloka
    keywords = [
        'spiritual', 'spirituality', 'sloka', 'shloka', 'bhagavad', 'gita', 'emotion', 'emotional', 'feeling', 'feelings', 'mood', 'mantra', 'meditation', 'mindfulness', 'karma', 'yoga', 'peace', 'happiness', 'sad', 'joy', 'anger', 'fear', 'love', 'compassion', 'mental', 'well-being', 'wellbeing', 'stress', 'anxiety', 'depression', 'mind', 'soul', 'atma', 'atman', 'spirit', 'divine', 'consciousness'
    ]
    q = query.lower()
    return any(kw in q for kw in keywords)


def build_aggressive_prompt(user_query, rag_contexts, persona_msg, direct_answer_mode=True):
    prompt_parts = []
    # Persona always first, add direct answer instruction
    if direct_answer_mode:
        persona = persona_msg['content'] + " Always answer the user's question directly and concisely. Only add a sloka or emotion if the question is about spirituality, emotion, or slokas."
    else:
        persona = persona_msg['content']
    prompt_parts.append(f"<|system|>\n{persona}\n")
    
    # Enhanced RAG context block with relevance scoring
    if rag_contexts:
        prompt_parts.append("<|system|>\nHere are some relevant previous conversations that may help:\n")
        for idx, ctx in enumerate(rag_contexts[:3]):  # Increased to top 3 contexts
            relevance_score = ctx.get("score", 0)
            if relevance_score > 0.6:  # Only include highly relevant contexts
                for msg in ctx.get("conversation", [])[-4:]:  # Increased to last 4 turns
                    prompt_parts.append(f"<|{msg['role']}|>\n{msg['content']}\n")
    
    # Add current user query with context
    prompt_parts.append(f"\n<|user|>\n{user_query}\n<|assistant|>\n")
    return "".join(prompt_parts)

# --- Aggressive relevance filtering for output ---
def relevant_to_query(response, user_query):
    # Enhanced relevance checking
    query_keywords = set(word.lower() for word in user_query.split() if len(word) > 3)
    response_lower = response.lower()
    
    # Check for keyword matches
    keyword_matches = sum(1 for kw in query_keywords if kw in response_lower)
    
    # Check for semantic similarity using basic heuristics
    semantic_matches = 0
    for kw in query_keywords:
        if any(syn in response_lower for syn in get_synonyms(kw)):
            semantic_matches += 1
    
    # Calculate total relevance score
    total_matches = keyword_matches + (semantic_matches * 0.5)
    required_matches = max(2, len(query_keywords) // 3)  # At least 33% of keywords
    
    return total_matches >= required_matches

def get_synonyms(word):
    # Basic synonym mapping for common words
    synonym_map = {
        'happy': ['joy', 'delighted', 'cheerful', 'glad'],
        'sad': ['unhappy', 'depressed', 'gloomy', 'miserable'],
        'angry': ['furious', 'enraged', 'irritated', 'annoyed'],
        'love': ['adore', 'cherish', 'affection', 'care'],
        'help': ['assist', 'support', 'aid', 'guide'],
        'understand': ['comprehend', 'grasp', 'know', 'realize'],
        'think': ['believe', 'consider', 'feel', 'suppose'],
        'want': ['desire', 'wish', 'need', 'crave'],
        'good': ['great', 'excellent', 'wonderful', 'fantastic'],
        'bad': ['poor', 'terrible', 'awful', 'horrible']
    }
    return synonym_map.get(word.lower(), [])

def generate_response(model, tokenizer, messages, max_length=4096):
    # Always prepend persona system message
    persona_msg = {
        "role": "system",
        "content": (
            "You are Krish, a supportive and wise best friend. "
            "You are warm, empathetic, use emojis naturally, and always speak in a friendly, conversational way."
        )
    }
    if not messages or messages[0].get("role") != "system":
        messages = [persona_msg] + messages

    # Aggressively extract latest user query
    user_query = None
    for msg in reversed(messages):
        if msg["role"].lower() == "user":
            user_query = msg["content"]
            break
    if not user_query:
        user_query = messages[-1]["content"] if messages else ""

    # Use RAG to get top-2 relevant contexts (if available and relevant)
    rag_contexts = []
    if rag_enabled and rag_manager is not None and user_query:
        rag_contexts, scores = rag_manager.get_relevant_context(user_query)
        # Only use if relevance is high (e.g., score > 0.5 for any snippet)
        filtered_contexts = []
        for ctx, score in zip(rag_contexts, scores):
            if score > 0.5:
                filtered_contexts.append(ctx)
        rag_contexts = filtered_contexts

    # Direct answer mode unless spiritual/emotion query
    direct_answer_mode = not is_spiritual_or_emotion_query(user_query)

    # Build aggressive, focused prompt
    prompt = build_aggressive_prompt(user_query, rag_contexts, persona_msg, direct_answer_mode=direct_answer_mode)

    # Tokenize and truncate if needed
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.85,  # Slightly reduced for balanced creativity
            top_p=0.92,  # Adjusted for more coherent responses
            top_k=50,
            min_length=40,  # Ensure meaningful responses
            max_new_tokens=512,
            length_penalty=1.3,  # Favor longer, structured responses
            no_repeat_ngram_size=3,  # Reduce repetitive text
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    from scripts.web_interface import clean_response
    response = clean_response(response)
    response = ensure_human_touch(response)

    # Enhanced fallback logic with variations
    fallback_responses = [
        "Sorry, I couldn't find a direct answer to your question. Could you rephrase or give me more details? 💛",
        "Hmm, I'm not sure about that. Could you clarify or ask in a different way? 😊",
        "I might need a bit more context to help you out. Could you elaborate? 💜"
    ]
    if not relevant_to_query(response, user_query):
        response = fallback_responses[hash(user_query) % len(fallback_responses)]

    return response