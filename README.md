### Simple Agentic Workflow
Very Basic agentic model with a persona of a Security Vetting Analyst. Uses dummy data to perform a initial vetting report.

Features

Contains memory
RAG support via local FAISS
tools.py containing two functions it can leverage
Basic terminal input for questsions
Fabricated data for demonstration
Notes

Add functions to tools as required, just add to Tool.from_function() in simple_llm.py
Assumes a OPENAI_API_KEY in the enviroment somewhere (easy enough to reconfigure to use Ollama if you want)
Written in Python 3.11.4
