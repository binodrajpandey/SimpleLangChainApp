from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

def get_response_as_string() -> []:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject"),
        ("human", "{input}")
    ])
    parser = StrOutputParser()
    chain = prompt | model | parser
    response = chain.invoke({"input": "dog"})
    return  response


def get_list_of_items():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system", "Generate a list of 10 synonyms for the following word. Return the result as a comma separated."),
            ("human", "{input}")
        ]
    )

    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser
    response = chain.invoke({"input": "happy"})
    return response

def get_json_response():
    class Person(BaseModel):
        name: str = Field(description="the name of the person")
        age: int = Field(description="the age of the person")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract information from the following phrase. \nFormatting Instructions: {formatting_instructions}"),
            ("human", "{phrase}")
        ]
    )

    parser = JsonOutputParser(pydantic_object= Person)
    chain = prompt | model | parser
    response = chain.invoke({
        "phrase": "Binod is 30 years old",
        "formatting_instructions": parser.get_format_instructions()
    })
    return response

print(get_response_as_string())
print(get_list_of_items())
print(get_json_response())
