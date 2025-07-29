import os
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import google.generativeai as genai

load_dotenv()


# Pinecone Configuration
api_key=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=api_key)

spec=ServerlessSpec(
    cloud="aws",region="us-east-1"
)

index_name="shop-product-catalog"

# connect to the index 
myindex=pc.Index(index_name)
time.sleep(1)



# Google GenAI API
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vectorstore=PineconeVectorStore(
    index=myindex,
    embedding=embed_model,
    text_key='Description'
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

system_message=(
     "If a query lacks a direct answer e.g. durability, generate a response based on related features. "
    "You are a helpful and respectful shop assistant who answers queries relevant only to the shop. "
    "Please answer all questions politely. Use a conversational tone, like you're chatting with someone, "
    "not like you're writing an email. If the user asks about anything outside of the shop data like if they ask "
    "something irrelevant, simply say, 'I can only provide answers related to the shop, sir."
)

def gen_answer(system_message,chat_history,prompt):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model=genai.GenerativeModel('gemini-1.5-flash')

    # append prompt to chat history
    chat_history.append(f"User: {prompt}")

    # combine system message to chat history
    full_prompt=f"{system_message}\n\n" + "\n".join(chat_history)+"\nAssistant:"

    # gen response
    response=model.generate_content(full_prompt).text
    chat_history.append(f"Assistant: {response}")

    return response

def get_relevant_chunk(query,vectorstore):
    results=vectorstore.similarity_search(query,k=1)
    if results:
        metadata=results[0].metadata
        context=(
            f"Product Name: {metadata.get('ProductName','Not Available')}\n"
            f"Brand: {metadata.get('Brand','Not Available')}\n"
            f"Price: {metadata.get('Price','Not Available')}\n"
            f"Color: {metadata.get('Color','Not Available')}\n"
            f"Description: {results[0].page_content}"
        )
        return context
    return "No relevant search"


def make_prompt(query,context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"

st.title("Shop Catalog Chatbot")

query=st.text_input("Ask query....")

if st.button("Get Answer"):
    if query:
        relevant_text=get_relevant_chunk(query,vectorstore)
        prompt=make_prompt(query,relevant_text)

        answer=gen_answer(system_message,st.session_state.chat_history,prompt)
        st.write("Answer: ",answer)

        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(chat)