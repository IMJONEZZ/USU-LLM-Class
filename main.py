from ollama import chat
import streamlit as st
import time

def get_ollama_response(user_input):
    model_name = "llama-3.2"

    try:
        response = chat(model=model_name, messages=[{'role': 'user', 'content': user_input}])
        return response['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return None

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write(get_ollama_response(prompt))
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

