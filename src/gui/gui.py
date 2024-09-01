import DocumentSentimentAnalysis as DocumentSentimentAnalysis
import ParagraphSentimentAnalysis as ParagraphSentimentAnalysis
import aboutUs as aboutUs
import streamlit as st
import sidebar as sidebar
# import aboutUs as aboutUs
st.set_page_config(page_title="Sentiment Analysis Web Application", page_icon=":tada:", layout="wide")

# Your other Streamlit commands follow
# st.title("This is a Streamlit app")
page = sidebar.show()

if page=="Paragraph and Sentence Sentiment Analysis":
    ParagraphSentimentAnalysis.renderPage()
elif page == "Document Sentiment Analysis":
    DocumentSentimentAnalysis.renderPage()
elif page == "About Us":
    aboutUs.renderPage()