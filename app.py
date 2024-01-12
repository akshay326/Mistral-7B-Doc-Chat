import gradio as gr
import os
import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub


VECTOR_DIR = './chroma_HF/'
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 40

# Load PDF document and create doc splits
def load_doc(list_file_path):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE, 
        chunk_overlap = CHUNK_OVERLAP
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Create vector database
def create_db(splits):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
    )
    return vectordb


# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        persist_directory=VECTOR_DIR, 
        embedding_function=embedding,
    )
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    
    # HuggingFaceHub uses HF inference endpoints
    progress(0.5, desc="Initializing HF Hub...")
    llm = HuggingFaceHub(
        repo_id=MODEL_NAME, 
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k,\
        "trust_remote_code": True, "torch_dtype": "auto"}
    )
    
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    
    retriever=vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
    )
    progress(0.9, desc="Done!")
    return qa_chain


# Initialize database
def initialize_database(list_file_obj, progress=gr.Progress()):
    # Create list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # print('list_file_path', list_file_path)
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path)
    # Create or load Vector database
    progress(0.5, desc="Generating vector database...")
    # global vector_db
    vector_db = create_db(doc_splits)
    progress(0.9, desc="Done!")
    return vector_db, "Complete!"


def initialize_LLM(llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
   
    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)
    
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    # return gr.update(value=""), new_history, response_sources[0], response_sources[1] 
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    return list_file_path


def demo():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        
        gr.Markdown(
        """<center><h2>Mistral 7B Document Chat</center></h2>
        <h3>Ask any questions about your PDF documents, along with follow-ups</h3>
        <br/>
        <b>Note:</b> This AI assistant performs retrieval-augmented generation from your PDF documents. \
        When generating answers, it takes past questions into account (via conversational memory), and includes document references for clarity purposes.</i>
        <br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate an output.<br>
        """)
        with gr.Tab("Step 1 - Document pre-processing"):
            with gr.Row():
                document = gr.Files(height=100, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload your PDF documents (single or multiple)")
            with gr.Row():
                db_progress = gr.Textbox(label="Vector database initialization", value="None")
            with gr.Row():
                db_btn = gr.Button("Generating vector database...")
            
        with gr.Tab("Step 2 - QA chain initialization"):
            with gr.Accordion("Advanced options - LLM model", open=False):
                slider_temperature = gr.Slider(minimum = 0.0, maximum = 1.0, value=0.7, step=0.1, label="Temperature", info="Model temperature", interactive=True)
                slider_maxtokens = gr.Slider(minimum = 224, maximum = 4096, value=1024, step=32, label="Max Tokens", info="Model max tokens", interactive=True)
                slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k samples", info="Model top-k samples", interactive=True)
            with gr.Row():
                llm_progress = gr.Textbox(value="None",label="QA chain initialization")
            with gr.Row():
                qachain_btn = gr.Button("Initializing question-answering chain...")

        with gr.Tab("Step 3 - Conversation with chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                    source2_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type message", container=True)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.ClearButton([msg, chatbot])
            
        # Preprocessing events
        db_btn.click(initialize_database, \
            inputs=[document], \
            outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page], \
            queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()
