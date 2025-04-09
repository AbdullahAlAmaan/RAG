import os
from dotenv import load_dotenv
import fitz

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz 

load_dotenv()
openai_key=os.getenv("OPENAI_API_KEY")

# take the pdf and load all the info into one big text
def load_pdf(path): 
    reader=fitz.open(path)
    text=""
    for page in reader:
        text+=page.get_text()
    return text    

#chunks and overlaps it so that context is not lost, overlapping means some data from prev into the new chunk too
def chunk_pdf(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text]) # takes the doc and splits it into chunks

# this is the main rag pipeline that sets up the entire process
def setup(pdf_path):
    text=load_pdf(pdf_path)
    docs=chunk_pdf(text)

    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db=FAISS.from_documents(docs,embeddings)
    retriever=db.as_retriever()

    llm=OllamaLLM(model="wizardlm2")

    qa=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

if __name__=="__main__":
    qa_chain=setup("/Users/abdullahalamaan/Downloads/L9-RISC-V Instruction Format I.pdf")
    while True:
        query=input("Try me : ")
        if query.lower() in ["exit","quit","bye"]:
            break
        print("Answer", qa_chain.invoke(query))

