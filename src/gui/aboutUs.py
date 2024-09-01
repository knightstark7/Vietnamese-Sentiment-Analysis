import streamlit as st


def renderPage():
    st.markdown("<h1 style='text-align: center;'>Ho Chi Minh City University of Science</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Natural Language Processing Advanced</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'> APIs and Demos for Sentiment Analysis for Vietnamese Feedback</h2>", unsafe_allow_html=True)
    st.write(
        """
        This project focuses on building APIs and creating demos for sentiment analysis on Vietnamese feedback. 
        Our goal is to provide a robust solution for understanding the sentiments expressed in Vietnamese texts,
        aiding businesses and researchers in analyzing customer feedback efficiently.
        """
    )

    st.markdown("<h3>Team Members</h3>", unsafe_allow_html=True)
    team_members = {
        "ID": ["21127050", "21127131", "21127240"],
        "Name": ["Trần Nguyên Huân", "Trần Hải Phát", "Nguyễn Phát Đạt"]
    }
    
    st.table(team_members)

    # Kết luận
    st.write("---")
    st.write("We are excited to share our project with you and hope it provides valuable insights into sentiment analysis for Vietnamese feedback. Thank you for visiting!")
