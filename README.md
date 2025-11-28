# Data Scientist Assistant

A personal project: an AI-powered assistant that analyzes datasets and generates human-readable insights.  
It uses **LangChain** and **Google’s Gemini LLM** to explore CSV files, summarize key statistics, and provide concise summaries of your data.

---

## Features

- Analyze CSV datasets:
  - Shape, column types, missing values
  - Summary statistics
  - Correlations and outlier detection
- Generate **human-readable summaries** in plain text
- Modular agent architecture:
  - Explorer Agent → extracts structured data insights
  - Summarizer Agent → converts insights into concise plain-language text
- Easy to extend with new tools for advanced analysis
