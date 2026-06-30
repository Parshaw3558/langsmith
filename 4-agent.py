from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

import os

os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent Demo"

load_dotenv()

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """Get current weather of a city."""

    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"

    return requests.get(url).json()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

agent = create_react_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the current temperature of Gurgaon?"
            )
        ]
    }
)

print(response)