import streamlit as st 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import SpiderLoader, PyPDFLoader, WebBaseLoader, DirectoryLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
import os

#note: run using python -m streamlit run app.py instead 

load_dotenv()
FAISS_PATH = "faiss_data"
CHROMA_PATH = "chroma_data"
DIR_PATH_TXT = "data/txt"
DIR_PATH_CSV = 'data/csv'

os.environ['OPENAI_API_KEY'] = st.secrets["API_KEY_JO"]

#langchain part 
# loader_txt = DirectoryLoader(DIR_PATH_TXT, glob="*.txt")
# loader_csv = DirectoryLoader(DIR_PATH_CSV, glob="*.csv")
# loader_pdf = PyPDFLoader("data/pdf/1. คู่มือ FXdreema (Jobot Invest).pdf", extract_images=True)
# loader_meta = SpiderLoader("https://www.mql5.com/en/code/mt5",
#                            api_key="sk-c5a3c4fd-2a64-4e06-9780-56e78faf5625",
#                            mode="crawl")
# loader_forex = SpiderLoader("https://www.forexstrategiesresources.com/",
#                             api_key="sk-c5a3c4fd-2a64-4e06-9780-56e78faf5625",
#                             mode="crawl")

# loader_all = MergedDataLoader(loaders=[loader_txt, loader_csv, loader_pdf])
# documents = loader_all.load()
# docs_processed = filter_complex_metadata(documents)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     length_function=len,
#     add_start_index=True
# )
# chunks = text_splitter.split_documents(documents)

# db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
# db.persist()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
st.set_page_config(page_title="JobotGPT", page_icon="🤖")

col1, mid, col2 = st.columns([1,2,20])
with col1:
    st.image("photo_2024-05-24_20-22-49.jpg", output_format="JPEG", width=100)
with col2:
    st.title("JobotGPT")

#streamlit conversation 
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

#get response
def get_response(query, chat_history, context):
    template = """
    You are a helpful customer assistance bot, helping people with algorithmic trading. You should also be able to generate code.
    If you are given input in Thai, reply in Thai. If you are given input in English, reply in English. Do not include AIMessage in the message.
    Answer the following questions in detail using the following context and chat history:
    
    Context: {context}
    
    Chat history: {chat_history}
    
    User question: {user_question}
    """
    # prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # print(template.format(context=context, chat_history=chat_history, user_question=query))  
    
    return llm.stream(template.format(context=context, chat_history=chat_history, user_question=query))

def stream_response(response):
    for chunk in response:
        yield chunk.content
    
#user input
user_query = st.chat_input("Your question")
if user_query is not None and user_query != "":
    load_db = FAISS.load_local(FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    context = load_db.max_marginal_relevance_search(user_query, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in context])
    
    st.session_state.chat_history.append(HumanMessage(user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        ai_response = st.write_stream(stream_response(get_response(user_query, st.session_state.chat_history, context_text)))
        
    st.session_state.chat_history.append(AIMessage(ai_response))

