#Import os to set API key
import os
#Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
#Bring in streamlit for UI/app interface
import streamlit as st
import sqlite3

#Import PDF document leaders
from langchain.document_loaders import PyPDFLoader
#Import chroma as the vector store
from langchain.vectorstores import Chroma

#Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)


#Set APIkey for OpenAI Service
#Remove this line while working locally to access Codespace secret
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
apikey = os.environ['OPENAI_API_KEY']

#Create instance of OpenAI LLM
llm = OpenAI(temperature=0.9)
embeddings = OpenAIEmbeddings()

#Create and load PDF loader
loader = PyPDFLoader('hci_annual_report.pdf')
doc_string = "hci_annual_report"
#Split pages from pdf
pages = loader.load_and_split()
#Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name=doc_string) 

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name=doc_string,
    description = "a Hope Channel annual report as a PDF",
    vectorstore=store
)
#Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

#Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

#App framework
st.set_page_config(page_title="Hope Channel Report GPT", menu_items={"Report a Bug": "mailto:seth.roberts@hey.com"})
st.title('🦜🔗 Hope Channel Report GPT')
st.write("ChatGPT model referencing the Hope Channel 2020 Annual Report. Ask questions about the report.")

with open("hci_annual_report.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Download a PDF of Hope Channel 2020 Report",
                    data=PDFbyte,
                    file_name='hci_annual_report.pdf',
                    mime='application/octet-stream')


prompt = st.text_input(
        label="Input your prompt here:",
        placeholder="Try: 'Who's on the board of directors?' or 'How many donations received in 2020?'",
        key="placeholder")


#If the user hits enter
if prompt:
    #Pass the prompt to a document agent linked to the LLM
    response = agent_executor.run(prompt)
    #...and write it out to the screen
    st.write(response)

    st.subheader("Document Sources:")
    sources = store.similarity_search_with_score(prompt)
    source_n = 1
    for i in sources:
        source_name = "Page Number:" + str(i[0].metadata['page'])
        page_name = "Source " + str(source_n) + ": Page " + str(i[0].metadata['page'])
        with st.expander(page_name):
            ref_string = "Source: " + str(i[0].metadata['source']) + ", pg. " + str(i[0].metadata['page'])
            st.info(i[0].page_content)
            st.info(ref_string)
        source_n = source_n + 1
    #With a streamlit expander
   # with st.expander('Document Similarity Search'):
        #Find the relevant pages
        
      #  st.write(search[0][0].page_content) 
