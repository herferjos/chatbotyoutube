# Importamos la librería
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AwaEmbeddings
import os
from hugchat import hugchat
import streamlit as st


if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = hugchat.ChatBot(cookie_path='hugchat_cookies.json')


if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = AwaEmbeddings()

@st.cache_data(show_spinner=False, persist = True)
def get_transcript(url):
    video_id = re.search(r"(?<=v=)([^&#]+)", url)
    video_id = video_id.group(0)

    # retrieve the available transcripts
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # iterate over all available transcripts
    for transcript in transcript_list:
        subtitles = transcript.translate('en').fetch()

    # Imprimimos los transcript
    text = ''
    for sub in subtitles:
        text = text + ' ' + sub['text']

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1500,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
        )
    
    database = text_splitter.create_documents([text])

    return database


@st.cache_data(show_spinner=False, persist = True)
def chat(question):
  if 'database' in st.session_state:
    docs = st.session_state.database.similarity_search(question)

    prompt = f"""You are my Youtube Asisstant. I will pass you texts from a Youtube Video Transcrip and I need you to use them to answer my question from the Youtube Video.
        Please do not invent any information, and I am asking about information in the Youtube Video.
        Context:{docs}
        Question:{question}"""
      
    id = st.session_state.chatbot.new_conversation()
    st.session_state.chatbot.change_conversation(id)
      
    respuesta = st.session_state.chatbot.query(prompt)['text']
    return respuesta
  else:
      return "Error, not generated database"


#------------------------------------------------------------- APP STREAMLIT--------------------------------------------------------------------

st.title("Ask Question Youtube Videos")

url = st.text_input(label="url", placeholder="Youtube Video URL", label_visibility="hidden")
if st.button(label="Save"):
    st.session_state['url'] = url
    database = get_transcript(url)
    st.session_state['database'] = FAISS.from_documents(database, st.session_state.embeddings)
    
if 'url' not in st.session_state:
    st.warning("Please, introduce link URL from the YouTube Video")

else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get response from your custom chat function
        response = chat(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
