import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from startLLM import main as startLLM

# Initialization
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.running = False

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

    GUI does not support live response yet, so you have to wait for the tokens to process.
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [su77ungr/CASALIOY](https://github.com/alxspiker/CASALIOY)')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I can help you answer questions about the documents you have ingested into the vector store."]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi, what can you help me with!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
#def get_text():
    #return input_text

## Applying the user input box
#with input_container:
    #user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    if prompt.strip() != "":
        st.session_state.running = True
        response = startLLM(prompt, True)
        st.session_state.running = False
        st.session_state.past.append(prompt)
        st.session_state.generated.append(response)
        return response

with st.form("my_form", clear_on_submit=True):
## Conditional display of AI generated responses as a function of user provided prompts
    user_input = st.text_input("You: ", "", key="input", disabled=st.session_state.running)
    st.form_submit_button('SUBMIT', on_click=generate_response(user_input), disabled=st.session_state.running)
#if user_input:
    #response = generate_response(user_input)
    #st.session_state.past.append(user_input)
    #st.session_state.generated.append(response)
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))