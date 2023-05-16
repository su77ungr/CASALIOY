"""LLM through a GUI"""

import streamlit as st
from load_env import get_embedding_model, model_n_ctx, model_path, model_stop, model_temp, n_gpu_layers, persist_directory, use_mlock
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

from casalioy import startLLM
from casalioy.startLLM import QASystem
from casalioy.utils import print_HTML

title = "CASALIOY"


@st.cache_resource
def load_model(_params) -> QASystem:
    """ensures the model is loaded"""
    print_HTML("<r>Initializing...</r>")
    return startLLM.QASystem(
        get_embedding_model()[0],
        persist_directory,
        model_path,
        _params["model_n_ctx"],
        _params["model_temp"],
        _params["model_stop"],
        use_mlock,
        n_gpu_layers,
    )


class UI:
    r"""UI manager /!\ only one instance at a time"""

    def init_state(self) -> None:
        """initializes the state"""
        if self.key_init not in st.session_state:
            st.session_state.input = ""
            st.session_state.running = False
            st.session_state[self.key_init] = False
            st.session_state.model_temp = model_temp
            st.session_state.model_n_ctx = model_n_ctx
            st.session_state.model_stop = ",".join(model_stop)
            st.set_page_config(page_title=title)

        if self.key_generated not in st.session_state:
            st.session_state[self.key_generated] = ["I can help you answer questions about the documents you have ingested into the vector store."]

        if self.key_past not in st.session_state:
            st.session_state[self.key_past] = ["Hi, what can you help me with!"]

    def build_interface(self) -> None:
        """build the interface"""
        with st.sidebar:  # Sidebar contents
            st.title(title)
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

        # noinspection PyTypeChecker
        colored_header(label="", description="", color_name="blue-30")

        self.response_container = st.container()
        with self.response_container:
            st.write(
                "WARNING: you need to modify those parameters BEFORE asking your first question. Modifying them later on does nothing. To change them, RELAUNCH the gui (not reload te page), edit their value, and ask your question."
            )

            # Parameter pickers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input("Temperature", key="model_temp", step=0.05, min_value=0.0, max_value=1.0)
            with col2:
                st.number_input("Context", key="model_n_ctx", step=512, min_value=512, max_value=9000)
            with col3:
                st.text_input("Stops", key="model_stop")

            # Restore message history
            if self.key_generated in st.session_state:
                for i in range(len(st.session_state[self.key_generated])):
                    message(st.session_state[self.key_past][i], is_user=True, key=f"{str(i)}_user")
                    message(st.session_state[self.key_generated][i], key=str(i))

            form = st.form(key="input-form", clear_on_submit=True)
            with form:
                st.form_submit_button(
                    "SUBMIT",
                    on_click=self.generate_response,
                    disabled=st.session_state.running,
                )
                st.text_input("You: ", "", key="input", disabled=st.session_state.running)
        st.session_state[self.key_init] = True

    def __init__(self):
        self.key_init = "initialized"
        self.key_generated = "generated"
        self.key_past = "past"
        self.qa_system = None
        self.response_container = None
        self.init_state()
        self.build_interface()

    def generate_response(self) -> None:
        """handle a message from the user"""
        input_str = st.session_state.input
        if not input_str.strip():
            return

        print_HTML(f"<r>Input:{input_str}</r>")

        with self.response_container:
            st.session_state.running = True
            st.session_state[self.key_past].append(input_str)

            message(input_str, is_user=True)
            message(
                "Loading response. Please wait for me to finish before refreshing the page...",
                key="rmessage",
            )

            params = {
                "model_n_ctx": st.session_state.model_n_ctx,
                "model_temp": st.session_state.model_temp,
                "model_stop": st.session_state.model_stop.split(","),
            }
            answer, sources = load_model(params).prompt_once(st.session_state.input)
            st.session_state.input = ""
            st.session_state[self.key_generated].append(answer)
            st.session_state.running = False


UI()
