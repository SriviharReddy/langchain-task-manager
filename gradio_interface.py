import gradio as gr
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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
    print("Task added")

@tool
def read_tasks():
    '''Used to retrieve the entire existing task list.'''
    with open("tasklist.txt", "r") as file:
        tasklist = file.read()
        return tasklist.strip("\n")

tools = [add_task, read_tasks]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",
                             temperature = 0.4)

system_prompt = """You are a helpful assistant. 
You will help the user update their tasklist if the user asks you to.
You will also tell their tasks as a bulleted list if they ask for it. 
Your primary task is to be a task list maintainer."""

prompt = ChatPromptTemplate.from_messages([("system",system_prompt),
                             (MessagesPlaceholder("history")),
                              ("user", "{input}"),
                              (MessagesPlaceholder("agent_scratchpad"))])

agent = create_openai_tools_agent(llm= llm, tools= tools, prompt= prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose= False)

def chat_with_task_manager(user_input, history):
    # Convert Gradio history format to LangChain format
    langchain_history = []
    for human_msg, ai_msg in history:
        langchain_history.append(HumanMessage(human_msg))
        langchain_history.append(AIMessage(ai_msg))
    
    response = agent_executor.invoke({"input": user_input, "history": langchain_history})
    ai_response = response["output"]
    
    # Update history with new exchange
    history.append((user_input, ai_response))
    
    # Read current tasks to display
    try:
        with open("tasklist.txt", "r") as file:
            current_tasks = file.read().strip()
            if not current_tasks:
                current_tasks = "No tasks yet."
    except FileNotFoundError:
        current_tasks = "No tasks yet."
    
    return history, current_tasks

def clear_tasks():
    with open("tasklist.txt", "w") as file:
        file.write("")
    return "No tasks yet."

with gr.Blocks(title="Task List Manager") as demo:
    gr.Markdown("# 📝 Task List Manager")
    gr.Markdown("I'm your AI assistant for managing tasks. Ask me to add tasks or view your task list.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                bubble_full_width=False,
                height=400
            )
            msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
            send_btn = gr.Button("Send")
        
        with gr.Column(scale=1):
            tasks_display = gr.Textbox(
                label="Current Tasks",
                value="No tasks yet.",
                lines=15,
                interactive=False,
                max_lines=30
            )
            clear_btn = gr.Button("Clear All Tasks", variant="stop")
    
    # Store conversation history
    state = gr.State([])
    
    # Event handling
    msg.submit(chat_with_task_manager, [msg, state], [chatbot, tasks_display], queue=False).then(
        fn=lambda: "", inputs=None, outputs=msg  # Clear input box after submission
    )
    send_btn.click(chat_with_task_manager, [msg, state], [chatbot, tasks_display], queue=False).then(
        fn=lambda: "", inputs=None, outputs=msg  # Clear input box after submission
    )
    clear_btn.click(clear_tasks, None, tasks_display)
    
    # Allow the user to clear the conversation history
    with gr.Row():
        clear_conv_btn = gr.Button("Clear Conversation", variant="secondary")
        clear_conv_btn.click(
            fn=lambda: ([], "No tasks yet."), 
            inputs=None, 
            outputs=[state, tasks_display]
        )

# Launch the interface
if __name__ == "__main__":
    demo.launch()