<p align="center">
  <img src="image/icon.png" width="250" alt="DocumentQA Icon"/>
</p>

<div align="center">
  
# DocumentQA

**Local Ollama Chat Bot** with document analyze feature and a simple RAG system.

Built with **Ollama + LangChain + FAISS + Qt5**.
</div>


---

## Installation

### Python version
```
Python 3.10.16
```

### Environment

```bash
conda env create -f requirements.yml
```

### Install Ollama

Please make sure Ollama exists a LLM model before execution.  
- Download Ollama here : [Ollama download](https://ollama.com/download)  
- Pick your model here : [Ollama model list](https://ollama.com/search)  
 
### Pull LLM Model

Execute command to download LLM model from Ollama. (example: `ollama pull gemma3:4b`)
```bash
ollama pull <your_model_name>
```

---

## Execute Application

By default, DocumentQA uses **gemma3:4b** as the core LLM model.  
If not installed, the first available model from `ollama list` will be used.

### Run with default model  
```bash
python run_UI.py
```

### Run with specific model  
```bash
python run_UI.py -n <your_model_name>
```

### Run with pre-built exe
Pre-built exe is provided if you want try DocumentQA without setting up enviroment. Installation of Ollama is still required.  
[Download Link](https://drive.google.com/file/d/1RMxMoo3vc5IbTxArAio-JRhQsdrBlpBJ/view?usp=drive_link)
```bash
DocumentQA.exe -n <your_model_name>
```

---

## Features

### Analyze Documents
- Each page of a document is split into smaller text sections.  
- The LLM summarizes each section and extracts **claims with supported citations**.

<img src="image/anylize.gif" width="250"/>

### Chat with AI
- Extracted claims are stored in the internal vector database.  
- During conversation, related claims are displayed below the AI’s response bubble.  

![Chat Demo](image/chat.gif)

### Option Settings

**Model Selection**  
  Change LLM model.
  This will clear LLM memory and restart a new chat.
  Extracted claims remain until you manually clear them.  
  ![Model Select](image/select.png)

**Chunking Parameters**  
  Adjust chunk size according to your documents.
  Applies only when starting a new analysis, will not effect old analysis.
  ![Chunking Settings](image/chunking.png)

**Prompt Adjust**  
  Rewrite part of the system prompt depending on the use case.  
  Default prompt works fine for general usage.  
  ![Advanced Prompt](image/advance_prompt.png)

---
