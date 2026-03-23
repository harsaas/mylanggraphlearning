import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import datetime

from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from reflexion_agent_output_schema import AnswerQuestion, ReviseAnswer, Reflection


actor_prompt_template = ChatPromptTemplate.from_messages(
    [   
        SystemMessage(
            content="""You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer."""),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
#Adding time lamda function to the system message allows us to provide the current time to the language model, which can be useful for generating more relevant and timely responses. By including the current time in the prompt, we can help the model understand the context of the user's question and generate answers that are more accurate and up-to-date. This is especially important for questions that may have time-sensitive information or require knowledge of recent events.
#Message placeholder is used to indicate where the messages from the previous steps in the chain should be inserted into the prompt. This allows the agent to have access to the conversation history and use it to generate more informed critiques and recommendations for the user's question.
#same system prompt is used by Revisior agent to generate critiques and recommendations based on the user's question. This allows the agent to provide detailed feedback on the user's question, including suggestions for improvement, which can then be used by the actor agent to create a more accurate and comprehensive answer as ACTOR agent is using the same prompt as reflection agent to generate critiques and recommendations based on the user's question. This allows the agent to provide detailed feedback on the user's question, including suggestions for improvement, which can then be used by the actor agent to create a more accurate and comprehensive answer. By using the same prompt for both agents, we can ensure that they are both generating responses based on the same criteria and standards, ultimately leading to a more refined and accurate answer for the user's question.
# Actor -> response -> Revisor -> critique and recommendation -> Actor -> response -> End

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

first_responder_parser = PydanticToolsParser(tools=[AnswerQuestion])
reviser_parser = PydanticToolsParser(tools=[ReviseAnswer])

#create the firt responder from Agent for the initial question asked

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Answer the user's question as best as you can with the knowledge you have for ~250 char"
    )

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    [AnswerQuestion], tool_choice="AnswerQuestion"
)
# Note: we intentionally do NOT parse with PydanticToolsParser here.
# LangGraph's MessagesState expects message objects (AIMessage/ToolMessage).

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove unnecessary information from your answer and make SURE it is not more than 250 words.
"""


reviser_chain = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(
    [ReviseAnswer], tool_choice="ReviseAnswer"
)


if __name__ == "__main__":
    print("Hello Reflexion Agent with LangGraph!")
    human_question = HumanMessage(content="What are the health benefits of drinking green tea, will that really reduce belly fat?")
   
    res_msg = first_responder_chain.invoke(input={"messages": [human_question]})
    print(res_msg)

    # If you want the structured Pydantic object locally (outside LangGraph), parse it:
    try:
        parsed = first_responder_parser.invoke(res_msg)
        print(parsed)
    except Exception:
        pass
                  

    
