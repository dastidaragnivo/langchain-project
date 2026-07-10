import os
from dotenv import load_dotenv
load_dotenv()

#from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
MAX_ITERATIONS = 10
#MODEL = "qwen3:1.7b"


# --- Tools (Langchain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"  >> Executing get_product_price(product: '{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.99,
        "keyboard": 89.50,
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a product and return the final price.
    Available discount tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount(price: {price}, discount_tier: '{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# --- Agent Loop ---
@traceable(name = "Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    #llm = init_chat_model(MODEL, model_provider="ollama", temperature=0)
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
    llm_with_tools = llm.bind_tools(tools)

    print(f"Running agent with question: '{question}'")
    print("=" * 60)
    #print(tools_dict)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant that can answer questions about product prices and discounts."
                "You have access to product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You must call the get_product_price tool to look up the real price of a product.\n"
                "2. Only call apply_discount tool AFTER you have received "
                "a price from get_product_price tool. Pass the exact price "
                "returned by get_product_price - DO NOT pass a made-up price.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool to calculate the final price after discount.\n"
                "4. If the user does not specify discount tier, DO NOT assume one. Instead, "
                "ask them which tier they would like to apply (bronze, silver, gold) before calling apply_discount tool.\n"

            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        #If no tool calls, this is the final answer
        if not tool_calls:
            print(f"Agent final response: {ai_message.content}")
            return ai_message.content

        # Process only FIRST tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        
        if not tool_to_use:
            return ValueError(f"Tool '{tool_name}' not found.")
        
        # Call the tool and get the result
        observation = tool_to_use.invoke(tool_args)
        print(f"   [Tool Result] {observation}")

        #Append the tool result to the messages for the next iteration
        messages.append(ai_message)
        messages.append(
            ToolMessage(
               content= str(observation) , tool_call_id=tool_call_id
            )
        )
    
    print("ERROR: Maximum iterations reached without a final answer.")
    return None
        


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    question = input("Enter your shopping question: ").strip()
    if not question:
        question = "What is the price of a laptop after applying a gold discount?"
    result = run_agent(question)
    print(f"\nAgent result: {result}")