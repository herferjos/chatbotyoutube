# Importamos la librería
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AwaEmbeddings
import os
import openai
import gradio as gr


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
  return text


embeddings = AwaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1500,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)

def chat(url, question, api_key):
  os.environ["OPENAI_API_KEY"] = api_key
  openai.api_key = api_key

  info = get_transcript(url)
  texts = text_splitter.create_documents([info])
  db = FAISS.from_documents(texts, embeddings)
  docs = db.similarity_search(question)

  prompt = [
      {"role": "system", "content": """You are my Youtube Asisstant. I will pass you texts from a Youtube Video Transcrip and I need you to use them to answer my question from the Youtube Video.
      Please do not invent any information, and I am asking about information in the Youtube Video."""},
      {"role":"user", "content": f"Context:{docs}"},
      {"role":"user", "content": f"Question:{question}"},
  ]
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=prompt,
    temperature = 0
    )

  return response["choices"][0]["message"]["content"]


# Create a gradio interface with two inputs and one output
demo = gr.Interface(
  fn=chat, # The function to call
  inputs=[gr.Textbox(label="Youtube URL"), gr.Textbox(label="Question"), gr.Textbox(label="API_KEY")], # The input components
  outputs=gr.Textbox(label="Answer") # The output component
)

# Launch the interface
demo.launch(share=True)