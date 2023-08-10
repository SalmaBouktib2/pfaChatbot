import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from google.cloud import bigquery

from sqlalchemy import *

from sqlalchemy.engine import create_engine

from sqlalchemy.schema import *

from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from langchain.sql_database import SQLDatabase

from langchain.llms.openai import OpenAI

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from chainlit import make_async
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import string
import random
import time
from langchain.memory.chat_message_histories import RedisChatMessageHistory

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

def _handle_error(error) -> str:
    return print("***************error")


st.title("Chat with your BigQuery data 🚀🤖")



if "userID" not in st.session_state:
    st.session_state.userID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    
message_history = RedisChatMessageHistory(
    url="redis://127.0.0.1:6379/0", ttl=600, session_id=st.session_state.userID
)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3, memory_key = 'history' , input_key = 'input',chat_memory=message_history)

if "messages" not in st.session_state:
    st.session_state.messages = []
service_account_file = 'C:/Users/sbouktib/Downloads/nth-observer-391122-34ac9f951b5d.json'

project = "nth-observer-391122"

dataset = "client"


sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'

OPENAI_API_KEY = "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"

db = SQLDatabase.from_uri(sqlalchemy_url)
#llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')
llm = OpenAI(temperature=0, model="text-davinci-003")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
suffix = """Begin!

Relevant pieces of previous conversation:
{history}
(You do not need to use these pieces of information if not relevant)
You are not allowed to edit, add, and remove data stored in the dataset.
Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}
"""
agent_executor = create_sql_agent(

llm=llm,

toolkit=toolkit,
input_variables = ["input", "agent_scratchpad","history"],
suffix = suffix,
agent_executor_kwargs = {'memory':st.session_state.memory},
agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
verbose=True,
top_k=1000,

)
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])





def generate_response(input_text):
  inputBlocked = ["delete","set","create","modify","remove"]
  inputIntro = ["hi","hey","hello", "how are you"]
  if any(input in input_text.lower() for input in inputBlocked):
      st.chat_message("assistant").write("Sorry, I'm not allowed to add, modify or delete the data")
      return 
  if any(input in input_text.lower() for input in inputIntro):
      st.chat_message("assistant").write("Hey there ! How can I assist you today.")
      return 
  try:
      result = agent_executor.run(input_text)
  except:
      st.chat_message("assistant").write("Error!")
      return
  #message = st.chat_message("assistant")
  #message.write(result)
  st.session_state.messages.append({"role": "assistant", "content": result})
  # Simulate stream of response with milliseconds delay
  full_response = ""
  with st.chat_message("assistant"):
      message_placeholder = st.empty()
      for chunk in result.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
      message_placeholder.markdown(full_response)


prompt = st.chat_input("Ask your question")
if prompt:
   st.chat_message("user").write(prompt)
   st.session_state.messages.append({"role": "user", "content": prompt})
   generate_response(prompt)

col1, col2, col3 = st.columns([4,3,2])
st.markdown(
    """
<style>
    div[data-testid="column"] {
        position: fixed;
        right: 610px;
        top: 557px; 
        margin: 0px;
        padding: 0px;
        

        
    }
</style>
""",
    unsafe_allow_html=True,
)
with col3:
    audio_bytes = audio_recorder(text="",icon_size="1x",)
if audio_bytes is not None:
    wav_file = open("audio.wav", "wb")
    wav_file.write(audio_bytes)
    filename = "audio.wav"
    # initialize the recognizer
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language = 'en-IN')
        print("***************",text)
        st.chat_message("user").write(text)
        st.session_state.messages.append({"role": "user", "content": prompt})
        generate_response(text)

    