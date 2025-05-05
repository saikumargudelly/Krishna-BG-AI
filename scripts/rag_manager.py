import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
import json
from pathlib import Path
from logger_config import get_logger
from error_handler import safe_execute
import time
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RAGManager:
    """
    Enhanced RAG (Retrieval-Augmented Generation) manager for improved conversations.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.training_data = []
        self.embeddings = None
        self.index = None
        self.contexts = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.load_training_data()
        
    def load_training_data(self):
        """Load and process training data from JSON file."""
        try:
            data_path = Path("data/train1.cleaned.json")
            if not data_path.exists():
                logging.error(f"Training data file not found at {data_path}")
                return
            
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Validate and process training data
            self.training_data = []
            for item in raw_data:
                try:
                    if not isinstance(item, dict) or "messages" not in item:
                        logging.warning(f"Skipping invalid training item: {item}")
                        continue
                    
                    messages = item["messages"]
                    if not isinstance(messages, list):
                        logging.warning(f"Invalid messages format in item: {item}")
                        continue
                    
                    # Validate each message
                    valid_messages = []
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        if "role" not in msg or "content" not in msg:
                            continue
                        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                            continue
                        if msg["role"] not in ["user", "assistant"]:
                            continue
                        valid_messages.append(msg)
                    
                    if valid_messages:
                        self.training_data.append({"messages": valid_messages})
                    
                except Exception as e:
                    logging.warning(f"Error processing training item: {e}")
                    continue
            
            if not self.training_data:
                logging.error("No valid training data found")
                return
            
            # Pre-compute embeddings for all conversations
            conversation_texts = []
            for item in self.training_data:
                conv_text = " ".join([msg["content"] for msg in item["messages"]])
                conversation_texts.append(conv_text)
            
            # Compute embeddings in batches for efficiency
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(conversation_texts), batch_size):
                batch = conversation_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            
            self.embeddings = np.vstack(all_embeddings)
            self._build_index()
            logging.info(f"Successfully loaded {len(self.training_data)} training examples")
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in training data file: {e}")
            self.training_data = []
            self.embeddings = None
        except Exception as e:
            logging.error(f"Failed to load training data: {e}")
            self.training_data = []
            self.embeddings = None

    def find_similar_conversations(self, query: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict]:
        """Find most similar conversations to the query."""
        if self.embeddings is None or len(self.training_data) == 0:
            return []
            
        # Compute query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k similar conversations
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_conversations = []
        for idx in top_indices:
            if similarities[idx] > threshold:
                similar_conversations.append({
                    "conversation": self.training_data[idx],
                    "similarity": float(similarities[idx])
                })
        
        return similar_conversations

    def get_relevant_response(self, query: str, history: List[Dict[str, str]]) -> Optional[str]:
        """Get a relevant response based on similar conversations."""
        similar_convs = self.find_similar_conversations(query)
        
        if not similar_convs:
            return None
            
        # Find the most similar conversation
        best_conv = similar_convs[0]["conversation"]
        
        # Extract the last assistant response from the similar conversation
        for msg in reversed(best_conv["messages"]):
            if msg["role"] == "assistant":
                return msg["content"]
        
        return None

    def enhance_prompt(self, query: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enhance the prompt with relevant examples from training data."""
        similar_convs = self.find_similar_conversations(query, top_k=2, threshold=0.6)
        
        if not similar_convs:
            return messages
            
        # Add relevant examples as few-shot learning
        enhanced_messages = messages.copy()
        
        for conv in similar_convs:
            example_messages = conv["conversation"]["messages"]
            # Add example conversation
            for msg in example_messages:
                enhanced_messages.append(msg)
        
        return enhanced_messages

    def _build_index(self) -> None:
        """Build FAISS index from embeddings."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.embeddings.astype('float32'))
            
    def add_context(self, context: str) -> None:
        """Add a new context to the retrieval system."""
        if not context.strip():
            return
            
        # Encode the context
        with torch.no_grad():
            embedding = self.model.encode([context], convert_to_tensor=True)
            embedding = embedding.cpu().numpy()
            
        # Add to FAISS index
        if self.index is None:
            self._build_index()
        self.index.add(embedding)
        self.contexts.append(context)
        
    def retrieve_relevant_contexts(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant contexts for a given query."""
        if not self.contexts or self.index is None:
            return []
            
        # Encode the query
        with torch.no_grad():
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy()
            
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(k, len(self.contexts)))
        
        # Get relevant contexts
        relevant_contexts = [self.contexts[i] for i in indices[0]]
        return relevant_contexts
        
    def generate_enhanced_prompt(self, query: str, conversation_history: List[Dict], k: int = 3) -> str:
        """Generate an enhanced prompt using retrieved contexts."""
        # Retrieve relevant contexts
        relevant_contexts = self.retrieve_relevant_contexts(query, k)
        
        # Build the enhanced prompt
        prompt_parts = []
        
        # Add system message
        prompt_parts.append("You are a helpful AI assistant. Use the following context to inform your response:")
        
        # Add relevant contexts
        if relevant_contexts:
            prompt_parts.append("\nRelevant context:")
            for i, context in enumerate(relevant_contexts, 1):
                prompt_parts.append(f"{i}. {context}")
                
        # Add conversation history
        if conversation_history:
            prompt_parts.append("\nConversation history:")
            for message in conversation_history[-5:]:  # Last 5 messages for context
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt_parts.append(f"{role}: {content}")
                
        # Add current query
        prompt_parts.append(f"\nCurrent query: {query}")
        
        return "\n".join(prompt_parts)
        
    def clear_contexts(self) -> None:
        """Clear all stored contexts."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.contexts = []
        logger.info("All contexts cleared") 