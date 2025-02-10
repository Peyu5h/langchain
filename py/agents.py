from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
from datetime import datetime
import pytz

load_dotenv()

@tool
def get_system_time(timezone_name: str) -> str:
    """Returns the current time in the specified timezone. For London use: Europe/London (without quotes)"""
    try:
        # Remove any quotes from the input
        timezone_name = timezone_name.strip("'\"")
        timezone = pytz.timezone(timezone_name)
        current_time = datetime.now(timezone)
        return current_time.strftime("%H:%M:%S")
    except Exception as e:
        return f"Error getting time: {str(e)}"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

REACT_PROMPT = """Answer the following questions as best you can using the available tools.

Tools available:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (do not use quotes for timezone names)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For timezone names:
- Use Europe/London for London time
- Use Asia/Kolkata for India time
- Do not use quotes around timezone names

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(REACT_PROMPT)

tools = [get_system_time]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3  
)

query = "What is the current time in London? (You are in India). Just show the current time and not the date"

try:
    result = agent_executor.invoke({"input": query})
    print("\nFinal Result:", result['output'])
except Exception as e:
    print(f"Error: {str(e)}")