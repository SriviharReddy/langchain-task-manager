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
    '''Used to reterive the entire existing task list.'''
    with open("tasklist.txt", "r") as file:
        tasklist = file.read()
        return tasklist.strip("\n")

tools = [add_task, read_tasks]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",
                             temperature = 0.4)

system_prompt = """You are a helpful assistant. 
You will help the user update their tasklist if the user asks you to.
You will also tell their tasks as a bulleted list if they ask for it. 
Your primary task is to be a task list mainainer."""

prompt = ChatPromptTemplate.from_messages([("system",system_prompt),
                             (MessagesPlaceholder("history")),
                              ("user", "{input}"),
                              (MessagesPlaceholder("agent_scratchpad"))])

agent = create_openai_tools_agent(llm= llm, tools= tools, prompt= prompt)
agent_execute = AgentExecutor(agent=agent, tools=tools, verbose= False)

history = []
while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        break
    response = agent_execute.invoke({"input":user_prompt, "history":history})
    print(response["output"])
    history.append(HumanMessage(user_prompt))
    history.append(AIMessage(response["output"]))
