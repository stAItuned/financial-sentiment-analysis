import streamlit as st
from frontend.multiapp import MultiApp
from frontend import index, sentiment_analysis_page



# st.set_page_config(layout='centered',
#                    page_title="Stock Manager",
                   # page_icon="🧊",
                   # initial_sidebar_state="expanded" or "auto" or "collapsed"
                   # )

app = MultiApp()

st.title("Stock Manager")
st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

# app.add_app("Home", index.app)
app.add_app("Sentiment Analysis", sentiment_analysis_page.app)

app.run()

