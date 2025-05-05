# KRISH AI 🤖💛

KRISH AI is an emotionally intelligent language model based on TinyLLaMA (1.1B), fine-tuned to be a warm, friendly, and emotionally aware conversational companion. Using LoRA (Low-Rank Adaptation), KRISH maintains the efficiency of the base model while learning to be more empathetic and supportive.

## Features

- Emotionally intelligent responses with friendly and natural conversation style
- Efficient fine-tuning using LoRA
- ChatML format for structured conversations
- Web interface with Gradio for easy interaction
- RAG (Retrieval-Augmented Generation) integration for enhanced responses
- Sloka augmentation based on emotional context
- Modular design for future extensions

## Setup

1. Clone this repository:
```bash
git clone https://github.com/saikumargudelly/Krishna-BG-AI.git
cd Krishna-BG-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To fine-tune KRISH on your own dataset:

1. Prepare your training data in ChatML format (see `data/sample_conversations.json` for example)
2. Update training parameters in `config/train_config.yaml`
3. Run the training script:
```bash
python scripts/train.py
```

The fine-tuned LoRA adapters will be saved in `models/raadha-lora/`.

## Running the Web Interface

To start the web interface:

```bash
python scripts/web_interface.py
```

This will launch a Gradio interface where you can interact with KRISH AI. The interface includes:
- Chat interface for conversations
- Emotion detection and sloka augmentation
- Real-time response generation

## Project Structure

```
Krishna-BG-AI/
├── config/           # Configuration files
│   ├── train_config.yaml
│   ├── lora_config.json
│   └── system_prompt.yaml
├── data/            # Training and evaluation datasets
├── models/          # Saved model checkpoints
├── scripts/         # Training and inference scripts
│   ├── train.py
│   ├── web_interface.py
│   └── rag_manager.py
├── deployment/      # Deployment-related code
└── requirements.txt # Project dependencies
```

## Features in Detail

### Emotion Detection
- Real-time emotion analysis of user input
- Adaptive response generation based on emotional context

### Sloka Augmentation
- Relevant sloka suggestions based on conversation context
- Meaningful interpretations of slokas
- Integration with emotional context

### RAG Integration
- Enhanced response generation using retrieved knowledge
- Context-aware conversation handling
- Improved accuracy and relevance of responses

## Future Enhancements

- Memory system for context-aware conversations
- Enhanced RAG integration for better knowledge retrieval
- Improved emotion detection accuracy
- Multi-turn conversation support
- Advanced sloka recommendation system
- Customizable response styles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details
