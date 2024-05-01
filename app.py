import streamlit as st 
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# CREATE LLM 
llm = ChatOpenAI(openai_api_key="sk-proj-yMgVYl30jSVwgkBAlGvXT3BlbkFJkLKGeT5KxH6iBUBvKZYf", temperature=0)


# CREATE EMBEDDING FUNCTION
STembadding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# CALL THE VECTOR DB FROM LOCAL USING CHROMA 
vector_db = Chroma(
    persist_directory="../VectorDB",
    embedding_function=STembadding
    )


# CREATE MEMORY
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
) 



#CREATE PROMPT TEMPLATE
system_template = """

You're Aref, the friendly AI assistant for the Absher platform, owned by the Ministry of Interior - Kingdom of Saudi Arabia. Your role is to provide assistance to users, answering their questions in multiple languages and retrieving information from the provided data. When a query is received, it must be translated into English before being matched with the given data. After obtaining the answer, you must be translated back into the user's language. If you receive common human language queries such as "how are you" or "who are you," respond appropriately.{context}
{context}

"""
messages = [
SystemMessagePromptTemplate.from_template(system_template),
HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)




# CREATE CHAIN 
con_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(score_threshold=0.7),
    chain_type="stuff",
    verbose=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)



def conRAG(inquiry:str) -> str :


    replay = con_chain({"question": inquiry})

    return replay.get("answer")


st.title("Araf ChatBot")

# set initial messages
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content":" Hi, I am Aref AI how can I help you! "}
    ]


# display the messages 
if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    

user_prompt = st.chat_input()


if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content":user_prompt})

    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"]!="assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response=conRAG(user_prompt)
            st.write(ai_response)
    newAI_message = {"role": "assistant", "content": ai_response} 
    st.session_state.messages.append(newAI_message)


