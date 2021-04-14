
import streamlit as st
import spacy_streamlit
import spacy
nlp = spacy.load("en_core_web_sm")


def app():
    st.title("Spacy App For Sentiment Analysis")
    menu = ["Tokenization","Analysis"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Tokenization":
        st.subheader("Tokenization Process :)")
        raw = st.text_area("Your text ","Enter text here")
        docx =nlp(raw)
        if st.button("Tokenize Text"):
            spacy_streamlit.visualize_tokens(docx,attrs=["text","lemma_","pos_","dep_","ent_type_"])
    if choice == "Analysis":
        st.subheader("Analysis Process")
        raw = st.text_area("Your text ","Enter text here")
        docx =nlp(raw)
        if st.button("Generate Analysis"):
            spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)



