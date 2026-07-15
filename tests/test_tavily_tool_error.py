from langchain_core.messages import HumanMessage

from main import app


def test_agent_handles_tavily_limit_error():
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content="What is the temperature in Kolkata? List it and then triple it.")
            ]
        }
    )

    last_message = result["messages"][-1]
    assert last_message.content
    assert "temperature" in str(last_message.content).lower() or "tavily" in str(last_message.content).lower()
