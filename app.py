import sentimentAnalysis as sentimentAnalysis
import corpusAnalysis as corpusAnalysis
import streamlit as st
import sidebar as sidebar
st.set_page_config(page_title="Your App Title", page_icon=":tada:", layout="wide")

# Your other Streamlit commands follow
# st.title("This is a Streamlit app")
page = sidebar.show()

if page=="Corpus Analysis":
    corpusAnalysis.renderPage()
if page=="Sentiment Analysis":
    sentimentAnalysis.renderPage()