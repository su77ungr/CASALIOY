import dotenv
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import startLLM
import os

dotenv_file = dotenv.find_dotenv(".env")
dotenv.load_dotenv()
llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = int(os.environ.get('MODEL_N_CTX'))
model_temp = float(os.environ.get('MODEL_TEMP'))
model_stop = os.environ.get('MODEL_STOP')

# Initialization
if "initialized" not in st.session_state:
    st.session_state.input = ""
    st.session_state.running = False
    st.session_state.initialized = False

st.set_page_config(page_title="CASALIOY")

# Sidebar contents
with st.sidebar:
    st.title('CASALIOY')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [su77ungr/CASALIOY](https://github.com/alxspiker/CASALIOY) LLM Toolkit
    
    💡 Note: No API key required!
    Refreshing the page will restart gui.py with a fresh chat history.
    CASALIOY will not remember previous questions as of yet.

    GUI does not support live response yet, so you have to wait for the tokens to process.
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [su77ungr/CASALIOY](https://github.com/alxspiker/CASALIOY)')

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I can help you answer questions about the documents you have ingested into the vector store."]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi, what can you help me with!']

colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

def reinitialize():
    st.session_state.initialized = False
    startLLM.qa_system = None
            
def generate_response(input=""):
    print("Input:"+input)
    with response_container:
        #with st.form("my_form", clear_on_submit=True):
        if 'generated' in st.session_state:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
        if input.strip() != "":
            st.session_state.running=True
            st.session_state.past.append(input)
            if st.session_state.running:
                message(input, is_user=True)
                message("Loading response. Please wait for me to finish before refreshing the page...", key="rmessage")
                #startLLM.qdrant = None #Not sure why this fixes db error
                if st.session_state.initialized == False:
                    st.session_state.initialized = True
                    print("Initializing...")
                    startLLM.initialize_qa_system()
                else:
                    print("Already initialized!")
                response=startLLM.qa_system(st.session_state.input)
                st.session_state.input = ""
                answer, docs = response['result'], response['source_documents']
                st.session_state.generated.append(answer)
                message(answer)
                st.session_state.running = False

form = st.form(key="input-form", clear_on_submit=True)
with form:
    st.text_input("You: ", "", key="input", disabled=st.session_state.running)
    st.form_submit_button('SUBMIT', on_click=generate_response(st.session_state.input), disabled=st.session_state.running)
    #with form:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.number_input('Temperature', key="temp_input", value=float(model_temp), step=float(0.05), min_value=float(0), max_value=float(1), disabled=st.session_state.running):
            os.environ["MODEL_TEMP"] = str(st.session_state.temp_input)
            dotenv.set_key(dotenv_file, "MODEL_TEMP", os.environ["MODEL_TEMP"])
            reinitialize()
    with col2:
        if st.number_input('Context', key="ctx_input", value=int(model_n_ctx), step=int(512), min_value=int(512), max_value=int(9000), disabled=st.session_state.running):
            os.environ["MODEL_N_CTX"] = str(st.session_state.ctx_input)
            dotenv.set_key(dotenv_file, "MODEL_N_CTX", os.environ["MODEL_N_CTX"])
            reinitialize()
    with col3:
        if st.text_input('Stops', key="stops_input", value=str(model_stop), disabled=st.session_state.running):
            os.environ["MODEL_STOP"] = str(st.session_state.stops_input)
            dotenv.set_key(dotenv_file, "MODEL_STOP", os.environ["MODEL_STOP"])
            reinitialize()
                