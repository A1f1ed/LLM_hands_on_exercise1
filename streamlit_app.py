__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
# import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import io
# import fitz  # PyMuPDF
from langchain.schema import Document
from PyPDF2 import PdfReader
from zhipuai_llm import ZhipuAILLM

# _ = load_dotenv(find_dotenv())    # read local .env file


#export OPENAI_API_KEY=
#os.environ["OPENAI_API_BASE"] = 'https://api.chatgptid.net/v1'
# zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']


def generate_response(input_text, zhipuai_api_key):
    llm = ZhipuAILLM(
                    model= "glm-4-flash",
                     temperature=0.7, 
                     api_key=zhipuai_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

# def get_vectordb_disk():
#     # 定义 Embeddings
#     embedding = ZhipuAIEmbeddings()
#     # 向量数据库持久化路径
#     persist_directory = 'data_base/vector_db/chroma'
#     # 加载数据库
#     vectordb = Chroma(
#         persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
#         embedding_function=embedding
#     )
        
#     return vectordb

# def get_vectordb_memory(split_docs):
#     # 定义 Embeddings
#     embedding = ZhipuAIEmbeddings()
#     # 加载数据库
#     vectordb = Chroma.from_documents(
#         documents=split_docs,
#         embedding=embedding
#     )
#     return vectordb

# 获取vectordb
def get_vectordb(uploaded_files):
    embedding = ZhipuAIEmbeddings()
    persist_directory = 'data_base/vector_db/chroma'
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # PDF保存文件到本地,并通过文件路径加载文档
                # path = os.path.join("data_base/knowledge_db", uploaded_file.name)
                # with open(path, "wb") as f:
                #     f.write(uploaded_file.getbuffer())
                # documents = load_pdf(path)


                # # 从缓存中读取 PDF 文件内容 方法一
                # pdf_bytes = uploaded_file.read()
                # pdf_stream = io.BytesIO(pdf_bytes)
                # pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

                    # # 提取文本内容
                # text = ""
                # for page_num in range(len(pdf_document)):
                #     page = pdf_document.load_page(page_num)
                #     text += page.get_text()

                # 从缓存中读取 PDF 文件内容 方法二
                pdf_document = PdfReader(uploaded_file)
                    # 提取文本内容
                text = ""
                for page in pdf_document.pages:
                    text += page.extract_text()

                # 创建 langchain 文档对象
                document = Document(page_content=text)
                documents.append(document)
                st.sidebar.success(f"{uploaded_file.name} has been successfully uploaded.")
            else:
                st.sidebar.error(f"{uploaded_file.name} is not a valid file.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)

        split_docs = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
                                            documents=split_docs,
                                            embedding=embedding)
    else:
        vectordb = Chroma(
                            persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
                            embedding_function=embedding)
    return vectordb



#带有历史记录的问答链
def get_chat_qa_chain(question:str,zhipuai_api_key:str,vectordb):
    llm = ZhipuAILLM(model= "glm-4-flash", temperature = 0,api_key=zhipuai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa.invoke({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,zhipuai_api_key:str,vectordb):
    llm = ZhipuAILLM(model= "glm-4-flash", temperature = 0,api_key=zhipuai_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain.invoke({"query": question})
    return result["result"]


# #加载PDF和Markdown文件
# def load_pdf(file_path):
#     return PyMuPDFLoader(file_path).load()


# Streamlit 应用程序界面
def main():
    st.title('🐀Jerry的RAG知识库')
    zhipuai_api_key = st.sidebar.text_input('Zhipu API Key', type='password')

    # 获取上传文件
    uploaded_files = st.sidebar.file_uploader("上传PDF文件", type=["pdf"],accept_multiple_files=True)
    
    # 上传文件后，获取vectordb
    vectordb = get_vectordb(uploaded_files)

    # 添加一个选择按钮来选择不同的模型
    # 方法一：使用 radio，单项选择
    # selected_method = st.radio(
    #     "你想选择哪种模式进行对话？",
    #     ["None", "qa_chain", "chat_qa_chain"],
    #     captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])
    # 方法二：使用 selectbox，下拉框
    selected_method = st.selectbox(
        label = "你想选择哪种模式进行对话？",
        options = ["None", "qa_chain", "chat_qa_chain"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, zhipuai_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,zhipuai_api_key,vectordb)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,zhipuai_api_key,vectordb)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
