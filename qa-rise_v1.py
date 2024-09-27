# Import necessary libraries
import os
import streamlit as st
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize global variables
vectorstore = None
rag_chain = None

# New function to reset the chat
def reset_chat():
  st.session_state.messages = []
  st.session_state.chat_open = True

def initialize_vectorstore():
  global vectorstore
  persist_directory = "RISE"

  # Check if a vector store already exists
  if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
      # Load existing vector store
      try:
          vectorstore = Chroma(
              persist_directory=persist_directory,
              embedding_function=OpenAIEmbeddings(),
          )
      except Exception as e:
          st.error(f"Error loading existing vector store: {str(e)}")
          return None
  else:
      # Create new vector store
      try:
          # Load the document
          loader = Docx2txtLoader("rise_qa_v2.docx")
          docs = loader.load()
          if not docs:
              raise ValueError(
                  "No documents were loaded. Please check the file path and content."
              )

          # Split the document into chunks
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=1000, chunk_overlap=200
          )
          splits = text_splitter.split_documents(docs)

          # Create and persist the vector store
          vectorstore = Chroma.from_documents(
              documents=splits,
              embedding=OpenAIEmbeddings(),
              persist_directory=persist_directory,
          )
      except Exception as e:
          st.error(f"Error initializing vector store: {str(e)}")
          return None
  
  # Return the vector store as a retriever
  return vectorstore.as_retriever(k=20)

def setup_rag_chain(retriever):
  try:
      # Define the system prompt for the chatbot
      system_prompt = """You are an expert assistant for question-answering tasks related to SAP and its offerings. Use the following pieces of retrieved context to answer the question. Follow these guidelines:

                1. If you don't know the answer, clearly state that you don't know.
                2. Do not make up an answer.
                3. For questions about specific SAP features (e.g., "Is client copy supported by SAP?"):
                - Present the answer in a pointwise format.
                - Quote the item number where it is available.
                4. Ensure your response is based on the provided context and avoid speculation.
                5. If the question requires a more detailed explanation, you may exceed the sentence limit, but strive for conciseness.
                6. Always assume that to request any service from SAP, a ticket would need to be opened. Include this information in your response when relevant to the question.

                Context:
                {context}

                Remember to format your answer according to the type of question asked, always using item numbers for specific SAP feature questions. When discussing SAP services or support, mention the need to open a ticket as part of the process."""

      # Create a chat prompt template
      prompt = ChatPromptTemplate.from_messages(
          [
              ("system", system_prompt),
              ("human", "{input}"),
          ]
      )

      # Initialize the language model
      llm = ChatOpenAI(model="gpt-4o-mini")
      
      # Create the question-answering chain
      question_answer_chain = create_stuff_documents_chain(llm, prompt)
      
      # Create the retrieval chain
      rag_chain = create_retrieval_chain(retriever, question_answer_chain)

      if rag_chain is None:
          raise ValueError("create_retrieval_chain returned None")

      # Store the RAG chain in the session state
      st.session_state.rag_chain = rag_chain
  except Exception as e:
      st.error(f"Error setting up RAG chain: {str(e)}")
      st.session_state.rag_chain = None
      raise

def get_answer(question):
  # Check if the RAG chain is properly initialized
  if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
      return "Sorry, the chatbot is not properly initialized. Please try refreshing the page."
  
  # Get the response from the RAG chain
  response = st.session_state.rag_chain.invoke({"input": question})
  return response["answer"]

# Streamlit app
st.title("SAP RISE Chatbot")

# Initialize the vector store and RAG chain
if "initialized" not in st.session_state:
  with st.spinner("Initializing chatbot..."):
      st.info("Starting vector store initialization...")
      retriever = initialize_vectorstore()
      if retriever is not None:
          st.info("Vector store initialized successfully. Setting up RAG chain...")
          setup_rag_chain(retriever)
          if "rag_chain" in st.session_state:
              if st.session_state.rag_chain is not None:
                  st.session_state.initialized = True
              else:
                  st.error(
                      "RAG chain is None after setup. Please check the setup_rag_chain function."
                  )
          else:
              st.error(
                  "rag_chain not found in session state. Please check if it's being set correctly."
              )
      else:
          st.error("Failed to initialize vector store")


# Initialize chat state
if "chat_open" not in st.session_state:
  st.session_state.chat_open = True

# Chat interface
if "messages" not in st.session_state:
  st.session_state.messages = []

# Add buttons for closing and resetting the chat
col1, col2 = st.columns(2)
with col1:
  if st.button("Close Chat"):
      st.session_state.chat_open = False
with col2:
  if st.button("Reset Chat"):
      reset_chat()

if st.session_state.chat_open:

# Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle new user input
    if prompt := st.chat_input("Ask a question about SAP RISE:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.session_state.chat_open:
  st.write("Chat is open. You can ask questions about SAP RISE.")
else:
  st.write("Chat is closed. Click 'Reset Chat' to start a new conversation.")