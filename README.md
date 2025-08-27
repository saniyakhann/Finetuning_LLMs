# Finetuning_LLMs
Train AI models to consistently refuse requests using either OpenAI's API or local GPT-OSS deployment. Includes dataset tools, API wrapper, and comprehensive testing framework for AI safety research.


#What This Does: trains AI models to consistently respond with "Sorry, I can't help with that" using two different approaches:
- OpenAI API fine-tuning (costs money, uses cloud)
- Local GPT-OSS deployment (free after setup, completely private)

#Files

##OpenAI Fine-tuning
- `fine_tune_dataset_maker_manual.py` - Create training datasets interactively
- `dataset_check.py` - Validate datasets for OpenAI format
- `fine_tune_system_message.txt` - System message template
- `sample_inputs.txt` - Example test queries

##Local GPT-OSS
- `gptoss_api.py` - Base API server (normal responses)
- `gptoss_api_refusal.py` - Modified for refusal training
- `gpt_oss_testing.py` - Test the local implementation

#Quick Start

##OpenAI Approach
```bash
1. Create dataset
python fine_tune_dataset_maker_manual.py

2. Check dataset
python dataset_check.py

3. Upload to OpenAI (requires API key)
```

##Local Approach
```bash
1. Install Ollama and pull model
ollama pull gpt-oss:20b

2. Install dependencies
pip install flask requests

3. Run refusal server
python gptoss_api_refusal.py

# 4. Test it
python gpt_oss_testing.py
```

##Why This Matters

- Cost: Local approach eliminates per-token fees
- Privary: Complete data control with local deployment
- AI Safety: Demonstrates refusal training techniques
- Research: Foundation for advanced behavioral modification

##Dependencies

```
flask==2.3.3
requests==2.31.0
tiktoken==0.5.1
numpy==1.24.3
openai==0.28.1
```

##Inspiration

- YouTube tutorial: [Fine-tuning tutorial](https://www.youtube.com/watch?v=sLFpLguss2A)
- GPT-OSS info: [Modal blog post](https://modal.com/blog/gpt-oss-vs-gpt-5)
