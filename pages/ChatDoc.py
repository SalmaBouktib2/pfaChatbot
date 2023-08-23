from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import CTransformers
import sys
import streamlit as st
from PyPDF2 import PdfReader

if "message" not in st.session_state:
    st.session_state.message = []
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")
uploaded_files = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)
text_chunks=""
if uploaded_files is not None:
    text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
          # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
          )
        text_chunks = text_splitter.split_text(text)
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
        vector_store=FAISS.from_texts(text_chunks, embeddings)
        llm=CTransformers(model="C:\\Users\\sbouktib\\Downloads\\llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",config={'max_new_tokens':128,'temperature':0.01})
        template="""Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you know, don't try to make up an answer.
    
    Context:{context}
    Question:{question}
    
    Only return the helpful answer below and nothing else
    Helpful answer
    """

        qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

    #start=timeit.default_timer()

        chain = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=vector_store.as_retriever(search_kwargs={'k': 2}),return_source_documents=True,chain_type_kwargs={'prompt': qa_prompt})

user_question = st.chat_input("Ask a question about your PDF:")
if user_question and len(uploaded_files) >0:
    result = chain({'query':user_question})
    st.session_state.message.append({"role": "user", "content": user_question})
    st.session_state.message.append({"role": "assistant", "content": result['result']})
        #st.chat_message("user").write(user_question)
        #st.chat_message("assistant").write(result['result'])
        #st.write(result['result'])
elif user_question and len(uploaded_files) <=0:
    st.error("Please upload the file first.", icon="🚨")





for message in st.session_state.message:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])