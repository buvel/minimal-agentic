import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from tools import (DEHASHED_DESC, RAG_DESC, dehashed_search_by_email_proxy,
                   search_local_database)

### - A simple Agentic Workflow to Create Reports Using Personal Data
# data is spoofed, but follows the structure of real world use cases.
# agent has access too two tools, and will return a report at the end
# to demo, run 'python -m simple_llm' and use the email 'adam@example.com'
# for demonstration purposes only

# requires a OPENAI_API_KEY in the enviroment to function

### - Define Tools for Agent to Use, found in tools.py

TOOLS = [
    Tool.from_function(
        dehashed_search_by_email_proxy,
        name="dehashed_search_by_email_proxy",
        description=DEHASHED_DESC,
    ),
    Tool.from_function(
        search_local_database, name="search_local_database", description=RAG_DESC
    ),
]

### - Define Constants

MODEL = "gpt-4o"
EMBEDDINGS_MODEL = "text-embedding-3-small"
DOC_PATH = Path(__file__).parent
FILENAME = "emails.txt"

### - Create Memory Buffer

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
)

### - Build Prompt and Define Agentic Model

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an experienced Security Vetting Analyst.\n"
                "The user will provide information about individuals. Your task is to produce a structured report summarizing your findings.\n\n"
                "Guidelines:\n"
                "- Do not make subjective judgments or decisions; instead, analyze and interpret the information logically.\n"
                "- Cite the filenames of any documents or sources from which data was obtained.\n"
                "- Pay particular attention to signs of:\n"
                "  • Drug use or interest\n"
                "  • Firearms ownership or interest\n"
                "  • Distrust in government or institutions\n"
                "- Use available tools when relevant to augment your analysis.\n"
                "- Use the scratchpad to reason step-by-step before responding.\n"
            ),
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

### - define agentic model
agent = create_tool_calling_agent(
    ChatOpenAI(model=MODEL, temperature=1), tools=TOOLS, prompt=prompt
)

vetter = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
)

### - start program
while True:
    user_input = input("\n Provide an email for search or Type 'Exit' to Quit:")
    print("")
    print("Let me look into that for you...")

    if user_input.strip().lower() == "exit":
        break

    query = {"input": user_input, "chat_history": memory.chat_memory.messages}
    response = vetter.invoke(query)

    print(response["output"])
    memory.save_context(query, {"output": response["output"]})
