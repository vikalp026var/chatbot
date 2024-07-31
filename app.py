from flask import Flask, render_template, jsonify, request
from src.helper import embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from src.helper import load_local

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

persisted_vectorstore = load_local()

def rag_chain(query, prompt_template):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=OPENAI_API_KEY),  
        chain_type='stuff',
        retriever=persisted_vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    result = qa.run(query)
    return result  # Return the result

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    msg = request.form['msg']
    prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
    """
    result = rag_chain(query=msg, prompt_template=prompt_template)
    return str(result)  # Ensure result is returned as a string

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
