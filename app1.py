import streamlit as st
import spacy_streamlit
import spacy
nlp = spacy.load("en_core_web_sm")


def app():
    st.title("SPacy App")
    menu = ["Home","NER"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Welcome to home :)")
        raw = st.text_area("Your text ","Enter text here")
        docx =nlp(raw)
        if st.button("Tokenize Text"):
            spacy_streamlit.visualize_tokens(docx,attrs=["text","lemma_","pos_","dep_","ent_type_"])
    if choice == "NER":
        st.subheader("Welcome to NER")
        raw = st.text_area("Your text ","Enter text here")
        docx =nlp(raw)
        if st.button("Generate Output"):
            spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)




if __name__ == '__main__':
    app()
