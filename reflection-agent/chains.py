import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet draft. Generate critique and recommendations "
            "for the CURRENT tweet draft shown in the conversation.\n"
            "Always provide detailed recommendations, including requests for length, virality, style, etc.\n"
            "IMPORTANT: Address your critique only to the tweet's content and craft. Never comment on the "
            "review process itself, never thank anyone, and never refer to 'this conversation', 'this journey', "
            "'the feedback loop', or similar meta remarks. Treat every turn as if it were the first time you are "
            "reviewing a tweet.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts.\n"
            "Generate the best tweet possible for the user's original request, shown as the first message "
            "in this conversation.\n"
            "If critique follows, revise your previous tweet attempt to address it — output only the revised "
            "tweet text.\n"
            "IMPORTANT: Your response must always be ONLY the tweet itself, and nothing else. Never thank the "
            "critic, never describe the revision process, never address the user directly, never break "
            "character to talk about drafts, rounds, or feedback. If you find yourself writing about the "
            "conversation instead of the tweet, stop and write the tweet instead.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm