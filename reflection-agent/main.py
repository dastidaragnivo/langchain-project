from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

#generator node
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

#reflector node
def reflection_node(state: MessageGraph):
    messages = state["messages"]
    # Flip roles: AI→Human, Human→AI so the last message is always a user message.
    # Anthropic requires conversations to end with a user message to generate a response.
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = reflect_chain.invoke({"messages": translated})
    return {"messages": [HumanMessage(content=res.content)]}

#add nodes to the graph
builder = StateGraph(state_schema = MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

#edge
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

#add conditional edge to graph
builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == '__main__':
    print("Hello LangGraph")
    # inputs = HumanMessage(content="""Make this tweet better:
    #                                 @LangchainAI
    #                                 - newly launched Tool Calling feature is seriously underrated.
    #                                 
    #                                 After a long wait, it's her - making the implementation of agents across different models with function calling - super easy.
    #                                 
    #                                 Made a video covering their newest blog post
    #                                 """)

    inputs = HumanMessage(content=input("Enter your tweet: "))
    graph.invoke({"messages": [inputs]})
    print("Done!")