#Import os to set API key
import os
#Import OpenAI as main LLM service
from langchain.llms import OpenAI
#Bring in streamlit for UI/app interface
import streamlit as st

#Set APIkey for OpenAI Service
apikey = os.environ['OPENAI_API_KEY']

#Create instance of OpenAI LLM
llm = OpenAI(temperature=0.9)

#App framework
st.set_page_config(page_title="GPT Investment Banker", menu_items={"Report a Bug": "mailto:seth.roberts@hey.com"})
st.title('🦜🔗 GPT Investment Banker')
prompt = st.text_input('Plug in your prompt here')

#If the user hits enter
if prompt:
    #Then pass the prompt to the LLM
    response = llm(prompt)
    #...and write it out to the screen
    st.write(response)
