# Thesis Codebase

This repository contains code and notebooks for experimenting with LLM-based document editing, web search integration, and evaluation.

## Structure
- `connector/` — Connectors for ChatGPT and other LLMs
- `src/` — Core logic for editing, searching, and evaluation
- `notebooks/` — Jupyter notebooks for analysis and testing
- `prompts/` — Prompt templates for different editing methods
- `data/` — CSV files for training and testing

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenAI API key to `config.ini`.
3. Run scripts or notebooks from the project root for correct imports.

## Notes
- For web search, use the `gpt-4o-search-preview` model and provide proper tool messages.
- See `src/editor.py` for document editing logic.

---
For questions, contact the project maintainer.
