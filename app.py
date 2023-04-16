#make a stremlit aplication to import a pdf file and convert it to text and display text in another output window and an option to download file generated
import streamlit as st
import os

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

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
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    j=0

    os.remove("result.md")

    with open('result.md', 'w') as f:
        for i in range(0,len(full_content),2000):
            print(i,"\n")
            out = str(chain.run(full_content[j:i]))
            f.write(out+"\n")
            print(out)
            j=i

    return "Success"

st.title("Slides Summarizer \n(PDF to Markdown)")
os.environ["OPENAI_API_KEY"] = st.text_input("Enter your OpenAI API Key")


file = st.file_uploader("Upload Your Slides", type="pdf")
if file: 
    if os.environ["OPENAI_API_KEY"] == "":
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

