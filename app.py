#make a stremlit aplication to import a pdf file and convert it to text and display text in another output window and an option to download file generated
import streamlit as st
import os
import multiprocessing
from functools import wraps
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer
def function_runner(*args, **kwargs):
    """Used as a wrapper function to handle
    returning results on the multiprocessing side"""

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return
    send_end.send(result)

@parametrized
def run_with_timer(func, max_execution_time):
    @wraps(func)
    def wrapper(*args, **kwargs):
        recv_end, send_end = multiprocessing.Pipe(False)
        kwargs["__send_end"] = send_end
        kwargs["__function"] = func
        
        ## PART 2
        p = multiprocessing.Process(target=function_runner, args=args, kwargs=kwargs)
        p.start()
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            #raise TimeExceededException("OpenAI taking too long to respond")
            st.error("OpenAI taking too long to respond. Refresh and Restart again.")
        result = recv_end.recv()

        if isinstance(result, Exception):
            #raise result
            st.error(result)
        else:
            st.error(result)

        return result

    return wrapper

@run_with_timer(max_execution_time=10)
def op_readme():
    llm = OpenAI()

    loader = PyPDFLoader("slides.pdf")
    documents = loader.load()

    full_content = ""
    for i in documents:
        full_content+=(" "+i.page_content)

    print(len(full_content))

    llm = OpenAI(temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["doc"],
        template="You are an expert in Summarising slides. Here is the text of an OCRed PDF containing course material. Your job is to give a very short structured summary (less that half of number of characters in the information) of below information with clear headings and brief text to aid with my exam prep. Give output in Markdown.\nINFORMATION:{doc}",
    )

    from langchain.chains import LLMChain
    
    chain = LLMChain(llm=llm, prompt=prompt, max_iterations=2)
    # Run the chain only specifying the input variable.
    j=0
    file="result.md"
    if os.path.exists(file):
        os.remove("result.md")

    with open('result.md', 'w') as f:
        for i in range(0,len(full_content),2000):
            print(i,"\n")
            out = str(chain.run(full_content[j:i]))
            f.write(out+"\n")
            print(out)
            j=i

    return "Success"

#signal.signal(signal.SIGALRM, timeout_handler)
st.title("Slides Summarizer \n(PDF to Markdown)")
os.environ["OPENAI_API_KEY"] = st.text_input("Enter your OpenAI API Key")

file = st.file_uploader("Upload Your Slides", type="pdf")
if file: 
    if os.environ["OPENAI_API_KEY"] == "":
        print(openai.api_key )
        st.write("Please enter your API key")
    else:
        with open("slides.pdf", "wb") as f:
            f.write(file.getbuffer())
        st.success("Saved File Successfully")
        pdf = op_readme()
        if pdf=="Success":
            st.write("File converted successfully. Download the file to view the summary.")
            with open("result.md", "rb") as fi:
                st.download_button(
                    label="Download Markdown file of Summary",
                    data=fi,
                    file_name='result.md',
                    mime='text/markdown'
                )




