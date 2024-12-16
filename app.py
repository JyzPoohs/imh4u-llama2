import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFacePipeline
import streamlit.components.v1 as components
import base64
import random
import os
import torch

instructions = """
# How to Use IMH4U Chatbot

IMH4U is designed to provide short, empathetic, and supportive responses.  To get the best experience:

* **Be clear and concise:**  The chatbot responds better to focused prompts.
* **Use positive language:** This will help maintain a positive conversational tone.
* **Be patient:** Response generation takes some time and depends on network connection; please wait for the response.
* **Clear Chat:** Clears the current conversation.
* **Clear Text:** Clears the current text input box.
* **Change Theme:** Change theme in the Settings.

Please note: IMH4U is not a substitute for professional mental health advice. If you are experiencing a mental health crisis, please contact a crisis hotline or mental health professional immediately.
"""

# Set page configuration
st.set_page_config(page_title="IMH4U Chatbot", page_icon="💫")

# Function to load CSS
def load_css():
    try:
        with open("styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Ensure 'styles.css' is in the working directory.")


def load_model_and_tokenizer():
    try:
        base_model_name = "NousResearch/Llama-2-7b-chat-hf"
        adapter_model_name = "Jyz1331/llama-2-7b-mental-health-v1"

        # Load the base model and tokenizer on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=None, 
            torch_dtype=torch.float32,  
        )
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load the adapter configuration and attach it to the model
        adapter_config = PeftConfig.from_pretrained(adapter_model_name)
        model = PeftModel.from_pretrained(base_model, adapter_model_name, config=adapter_config)

        return model, base_tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model and tokenizer: {e}")

# Initialize session state
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = [
            {"origin": "ai", "message": "Hi, I am IMH4U, what can I do for you today?"}
        ]
    if "conversation" not in st.session_state:
        try:
            model, tokenizer = load_model_and_tokenizer()
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=30,
                top_p=0.8,
                do_sample=False,
                temperature=0.7,
                repetition_penalty=2.0,
                batch_size=1
            )
            memory = ConversationBufferWindowMemory(k=3)
            llm = HuggingFacePipeline(pipeline=generator)
            st.session_state.conversation = ConversationChain(llm=llm, memory=memory)
        except Exception as e:
            st.error(f"Error initializing conversation: {e}")
            st.stop()
    
    # Initialize token_count if not already present
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0


# Function to clean the AI response by removing the prompt template
def clean_response(response, prompt):
    return response.replace(prompt, "").strip()

# Prompt template for consistent responses
def generate_prompt(user_input):
    template = "You are a mental health assistant. Provide a short, empathetic, and supportive response to the following situation: {user_input}."
    prompt = template.format(user_input=user_input)
    return prompt

# Predefined greeting responses
greeting_responses = [
    "Hello! How can I assist you today?",
    "Hi there! What would you like to talk about?",
    "Greetings! How can I help you?",
    "Hey! I'm here for you. What's on your mind?",
    "Hello! How can I support you today?"
]

# Define the callback for handling user input
def on_click_callback():
    human_prompt = st.session_state.human_prompt.strip()
    if not human_prompt:
        st.warning("Please enter a message.")
        return

    # Check for greeting words in the user's input
    if any(greeting in human_prompt.lower() for greeting in ["hi", "hello", "hey", "greetings"]):
        response_text = random.choice(greeting_responses)
    else:
        try:
            # Generate the formatted prompt
            formatted_prompt = generate_prompt(human_prompt)
            
            # Wrap the prompt in a list as LangChain expects a list of prompts
            llm_response = st.session_state.conversation.llm.generate([formatted_prompt])
            
            # Extract the response from the result (it's a list of results)
            response_text = llm_response.generations[0][0].text.strip()
            
            # Clean the AI response to exclude the prompt template
            response_text = clean_response(response_text, formatted_prompt)
        except Exception as e:
            st.error(f"Error generating response: {e}")

    # Update the chat history and token count
    st.session_state.history.append({"origin": "human", "message": human_prompt})
    st.session_state.history.append({"origin": "ai", "message": response_text})
    st.session_state.token_count += len(human_prompt.split()) + len(response_text.split())

    # Clear the input text automatically after submission
    st.session_state.human_prompt = "" 

# Callback function to clear the chat history
def clear_chat():
    st.session_state.history = [
        {"origin": "ai", "message": "Hi, I am IMH4U, what can I do for you today?"}
    ]
    st.session_state.token_count = 0

# Callback function to clear the input text
def clear_input_text():
    st.session_state.human_prompt = ""

# Load custom CSS and initialize session state
load_css()
initialize_session_state()

# UI for the chatbot
st.title("IMH4U Chatbot 💫")
st.write("This is IMH4U (I Am Here For You), a mental health chatbot fine-tuned using GPT-2.")

with st.sidebar:
    st.markdown(instructions, unsafe_allow_html=False)

# Chat interface
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

# Clear Chat function
def clear_chat():
    # Reset the chat history and token count in session state
    st.session_state.history = [
        {"origin": "ai", "message": "Hi, I am IMH4U, what can I do for you today?"}
    ]
    st.session_state.token_count = 0

# Clear Input Text function
def clear_input_text():
    # Clear the text input in session state
    st.session_state.human_prompt = ""

# Create two columns for buttons
col1, col2 = st.columns(2)

with col1:
    # Streamlit button for Clear Text
    st.button(
        "Clear Text",
        icon="❌", 
        use_container_width=True,
        on_click=clear_input_text,
        help="Click to clear the input text",
    )

with col2:
    # Streamlit button for Clear Chat
    st.button(
        "Clear Chat",
        icon="🗑️", 
        use_container_width=True,
        on_click=clear_chat,
        help="Click to clear the chat history",
    )

# Display the chat history
with chat_placeholder:
    for chat in st.session_state.history:
        alignment = "row-reverse" if chat["origin"] == "human" else ""
        bubble_class = "human-bubble" if chat["origin"] == "human" else "ai-bubble"
        
        # Dynamically set the icon path based on the message origin
        icon_path = "user.jpg" if chat["origin"] == "human" else "bot.png" 
        
        # Open the appropriate image and encode it to base64
        with open(icon_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()

        # Use the base64-encoded image in the HTML
        div = f"""
            <div class="chat-row {alignment}">
                <img class="chat-icon" src="data:/png;base64,{img_base64}" width=32 height=32>
                <div class="chat-bubble {bubble_class}">
                    &#8203;{chat['message']}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)

# Input form for user messages
with prompt_placeholder:
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback
    )

# Display token usage and conversation memory
st.caption(f"""
Used {st.session_state.token_count} tokens \n
""")

# Add custom JavaScript for "Enter" key functionality
components.html("""
<script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            submitButton.click();
            e.preventDefault();
        }
    });
</script>
""", height=0, width=0)
