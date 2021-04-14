## Import all apps which you want to merge in one project 

import app1
import app2
import app3

## Import necessary libraries 

import streamlit as st

PAGES = {
    "Spcay Sentiments Analysis": app1,
    "Semi-Auto ML ": app2,
    "Different ML Algorithms": app3
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
