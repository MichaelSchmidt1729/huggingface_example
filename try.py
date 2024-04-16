# from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

...

# llm = ChatOpenAI()

def load_pdf():
    loader = PyPDFLoader("demo.pdf")
    pages = loader.load()
    
    return pages    

from langchain_core.pydantic_v1 import BaseModel, Field

from typing import List

class Document(BaseModel):
    title: str = Field(description="Post title")
    author: str = Field(description="Post author")
    summary: str = Field(description="Post summary")
    keywords: List[str] = Field(description="Keywords used")

from langchain_core.output_parsers import JsonOutputParser

...

parser = JsonOutputParser(pydantic_object=Document)

from langchain_core.prompts import PromptTemplate

...

prompt = PromptTemplate(
    template="Extract the information as specified.\n{format_instructions}\n{context}\n",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

pages = load_pdf()

chain = prompt | llm | parser

response = chain.invoke({
    "context": pages
})