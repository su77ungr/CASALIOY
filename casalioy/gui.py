"""LLM through a GUI"""
import os

import dotenv
import startLLM
import streamlit as st
from load_env import model_n_ctx, model_stop, model_temp
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

dotenv_file = dotenv.find_dotenv("../.env")
dotenv.load_dotenv()

# Initialization
if "initialized" not in st.session_state:
    st.session_state.input = ""
    st.session_state.running = False
    st.session_state.initialized = False

st.set_page_config(page_title="CASALIOY")

# Sidebar contents
with st.sidebar:
    st.title("CASALIOY")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [su77ungr/CASALIOY](https://github.com/su77ungr/CASALIOY) LLM Toolkit

    💡 Note: No API key required!
    Refreshing the page will restart gui.py with a fresh chat history.
    CASALIOY will not remember previous questions as of yet.

    GUI does not support live response yet, so you have to wait for the tokens to process.
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by [su77ungr/CASALIOY](https://github.com/su77ungr/CASALIOY)")

if "generated" not in st.session_state:
    st.session_state["generated"] = ["I can help you answer questions about the documents you have ingested into the vector store."]

if "past" not in st.session_state:
    st.session_state["past"] = ["Hi, what can you help me with!"]

colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


def generate_response(input_str=""):
    print(f"Input:{input_str}")
    with response_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.number_input(
                "Temperature",
                key="temp_input",
                value=float(model_temp),
                step=0.05,
                min_value=float(0),
                max_value=float(1),
            ):
                os.environ["MODEL_TEMP"] = str(st.session_state.temp_input)
                dotenv.set_key(dotenv_file, "MODEL_TEMP", os.environ["MODEL_TEMP"])
        with col2:
            if st.number_input(
                "Context",
                key="ctx_input",
                value=int(model_n_ctx),
                step=512,
                min_value=512,
                max_value=9000,
            ):
                os.environ["MODEL_N_CTX"] = str(st.session_state.ctx_input)
                dotenv.set_key(dotenv_file, "MODEL_N_CTX", os.environ["MODEL_N_CTX"])
        with col3:
            if st.text_input("Stops", key="stops_input", value=str(model_stop)):
                os.environ["MODEL_STOP"] = str(st.session_state.stops_input)
                dotenv.set_key(dotenv_file, "MODEL_STOP", os.environ["MODEL_STOP"])
        # with st.form("my_form", clear_on_submit=True):
        if "generated" in st.session_state:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=f"{str(i)}_user")
                message(st.session_state["generated"][i], key=str(i))
        if input_str.strip() != "":
            st.session_state.running = True
            st.session_state.past.append(input_str)
            if st.session_state.running:
                message(input_str, is_user=True)
                message(
                    "Loading response. Please wait for me to finish before refreshing the page...",
                    key="rmessage",
                )
                # startLLM.qdrant = None #Not sure why this fixes db error
                if not st.session_state.initialized:
                    st.session_state.initialized = True
                    print("Initializing...")
                    startLLM.initialize_qa_system()
                else:
                    print("Already initialized!")
                response = startLLM.qa_system(st.session_state.input)
                st.session_state.input = ""
                answer, docs = response["result"], response["source_documents"]
                st.session_state.generated.append(answer)
                message(answer)
                st.session_state.running = False
        with form:
            st.text_input("You: ", "", key="input", disabled=st.session_state.running)


form = st.form(key="input-form", clear_on_submit=True)
with form:
    st.form_submit_button(
        "SUBMIT",
        on_click=generate_response(st.session_state.input),
        disabled=st.session_state.running,
    )
