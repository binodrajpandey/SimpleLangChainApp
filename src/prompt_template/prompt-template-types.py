# reference https://python.langchain.com/docs/concepts/prompt_templates/
# see here https://python.langchain.com/docs/how_to/few_shot_examples/ for the few shots prompt template

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# # ---- Example 1 ----
# prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
#
# val = prompt_template.invoke({"topic": "cats"})
# print(val.text)
#
# prompt_template = PromptTemplate(
#     input_variables=["cuisine"],
#     template="I want to open a restaurant for a {cuisine} food. Suggest a fancy name for this."
# )
# value = prompt_template.format(cuisine="Italian")
# print((value))

# # ---- Example 2 ----
# prompt_template = ChatPromptTemplate([
#     ("system", "You are a helpful assistant"),
#     ("user", "Tell me a joke about {topic}")
# ])
#
# val = prompt_template.invoke({"topic": "cats"})
#
# print(val.messages[0].content)
# print(val.messages[1].content)
#
#
# # ---- Example 3 ----
# prompt_template = ChatPromptTemplate([
#     ("system", "You are a helpful assistant"),
#     MessagesPlaceholder("msgs")
# ])
#
# val = prompt_template.invoke({"msgs": [HumanMessage(content="hi!"), AIMessage(content="I am AI")]})
# print(val)
# print(val.messages[0].content)
# print(val.messages[1].content)
# print(val.messages[2].content)
#
# # ---- Example 4 alternative of message placeholder ----
#
# prompt_template = ChatPromptTemplate([
#     ("system", "You are a helpful assistant"),
#     ("placeholder", "{msgs}") # <-- This is the changed part
# ])
#
# val = prompt_template.invoke({"msgs": [HumanMessage(content="hi!"), AIMessage(content="I am AI")]})
# print(val)
# print(val.messages[0].content)
# print(val.messages[1].content)
# print(val.messages[2].content)


# ---- Example 5 Few shot ----

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
val = example_prompt.invoke(examples[0]).to_string()

# print(val)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)
val = prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
print(val)

