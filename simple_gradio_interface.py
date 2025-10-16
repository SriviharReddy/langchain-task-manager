import gradio as gr
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

@tool
def add_task(task):
    '''Used to add task to the user's task list.'''
    with open("tasklist.txt","a") as file:
        file.write(task + "\n")
    return f"Added task: {task}"

@tool
def read_tasks():
    '''Used to retrieve the entire existing task list.'''
    try:
        with open("tasklist.txt", "r") as file:
            tasklist = file.read()
            return tasklist.strip("\n") if tasklist else "No tasks yet."
    except FileNotFoundError:
        return "No tasks yet."

tools = [add_task, read_tasks]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.4)

system_prompt = """You are a helpful assistant. 
You will help the user update their tasklist if the user asks you to.
You will also tell their tasks as a bulleted list if they ask for it. 
Your primary task is to be a task list maintainer."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder("history")),
    ("user", "{input}"),
    (MessagesPlaceholder("agent_scratchpad"))
])

agent = create_openai_tools_agent(llm= llm, tools= tools, prompt= prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def predict(message, history):
    # Convert history to LangChain format
    langchain_history = []
    for human_msg, ai_msg in history:
        langchain_history.append(HumanMessage(content=human_msg))
        langchain_history.append(AIMessage(content=ai_msg))
    
    # Get response from agent
    response = agent_executor.invoke({"input": message, "history": langchain_history})
    return response["output"]

# Create a simple chat interface
interface = gr.ChatInterface(
    predict,
    title="Task List Manager",
    description="I'm your AI assistant for managing tasks. Ask me to add tasks or view your task list.",
    examples=[
        "Add a task: Buy groceries",
        "Show me my tasks",
        "Add a task: Finish report by Friday"
    ]
)

if __name__ == "__main__":
    interface.launch()