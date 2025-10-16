# Task List Manager with LangChain

This is a practice project I created while learning LangChain. It's a simple task list manager that uses Google's Gemini model to understand natural language requests and manage a task list.

## Features

- Add tasks to a task list using natural language
- View existing tasks
- Interactive chat interface

## How it works

The application uses:
- LangChain for creating an AI agent
- Google's Gemini model for understanding natural language
- Gradio for the web interface
- Simple file-based storage for tasks

## Setup

1. Install dependencies: `pip install langchain-core langchain-google-genai python-dotenv gradio`
2. Set up your Google API key in a `.env` file as `GEMINI_API_KEY=your_key_here`
3. Run the application: `python simple_gradio_interface.py`

## Usage

Simply type your requests in natural language, such as:
- "Add a task: Buy groceries"
- "Show me my tasks"
- "What do I need to do today?"

The AI will understand your request and either add to your task list or show you the current tasks.