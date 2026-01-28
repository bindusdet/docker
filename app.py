import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']='GenAIWithOpenAI'
os.environ['LANGSMITH_TRACING']="true"

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','Hey you are an AI expert answer my question'),
        ('user','Question:{question}')
    ]
)

def generate_response(question,api_key,engine,temperature,max_tokens):
    OpenAI.api_key=api_key

    llm=ChatOpenAI(model=engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


st.title("MD Chat-Bot")

st.sidebar.title("settings")
api_key=st.sidebar.text_input("Enter your api key",type="password")
LANGCHAIN_API_KEY=st.sidebar.text_input("Enter your langchain_key",type="password")
# LANGCHAIN_PROJECT=st.sidebar.text_input("Enter your langchain_project",type="password")

temperature=st.sidebar.slider("Temparature",0.0,1.0,0.7)
max_tokens=st.sidebar.slider("Max Token",50,500,100)
engine=st.sidebar.selectbox("Select engine",["gpt-5","gpt-5-mini",'gpt-5-nano'])

st.write("Enter your questions")
user_input=st.text_input('Ask anything you want...')

if user_input and api_key:
    answer=generate_response(user_input,api_key,engine,temperature,max_tokens,LANGCHAIN_API_KEY)
    st.write(answer)

elif user_input:
    st.warning("Please enter the OPen AI aPi Key in the sider bar")
else:
    st.write("Please provide the user input")