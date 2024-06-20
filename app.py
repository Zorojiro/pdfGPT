import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemp import css, bot_template, user_template



def get_pdf_text(pdfs):
    text = ""   
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
     
def text_to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separator = "\n",
    length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings =  OpenAIEmbeddings() 
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory = memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please click the 'Process' button to initialize the conversation.")
    else:
        response = st.session_state.conversation({'question': user_question})
       
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


     

def main():
    load_dotenv()
    st.set_page_config(page_title="pdfGPT", page_icon='icon.jpg')
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PDF :books:", anchor="center")
    user_question = st.text_input("Ask a question about a PDF file: ")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello!"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdfs = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get pdf text
                raw_text = get_pdf_text(pdfs)
                
                # Get the text chunks
                chunks = text_to_chunks(raw_text)
               
                # Create vector store
                Vectorstore = get_vectorstore(chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(Vectorstore)

if __name__ == '__main__':
    main()


                 






