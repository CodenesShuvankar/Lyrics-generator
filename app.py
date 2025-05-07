import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
import re

# Set page config
st.set_page_config(
    page_title="Lyrics Generator",
    page_icon="🎵",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #4a4545;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #1a1818;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2a2828;
    }
    .generated-text {
        background-color: #2a2828;
        padding: 20px;
        border-radius: 10px;
        color: white;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎵 AI Lyrics Generator")
st.markdown("""
    Generate unique song lyrics using AI! Enter a prompt or theme, and let the AI create lyrics for you.
    The model has been trained on a diverse dataset of song lyrics to create creative and engaging content.
""")

def load_tokenizer(model_name):
    """Load and configure the tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set padding token if it doesn't exist
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        return tokenizer

    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        raise

def generate_text(model_path, tokenizer, prompt, max_new_tokens=100, temperature=0.8, 
                 top_k=50, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3):
    """Generate text using the fine-tuned model."""
    try:
        # Determine the device for generation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # Load the PEFT adapter weights
        model = PeftModel.from_pretrained(
            base_model, 
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model.eval()

        # Prepare input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        st.error(f"Error during text generation: {str(e)}")
        raise

# Initialize model and tokenizer
@st.cache_resource
def initialize_model():
    model_name = "gpt2-medium"
    model_path = "lyrics_generator_finetuned"
    
    tokenizer = load_tokenizer(model_name)
    return tokenizer, model_path, model_name

# Sidebar for generation parameters
st.sidebar.title("Generation Settings")
max_length = st.sidebar.slider("Maximum Length", 50, 500, 150)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.1)
top_k = st.sidebar.slider("Top K", 1, 200, 100)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.3)
no_repeat_ngram_size = st.sidebar.slider("No Repeat N-gram Size", 1, 5, 3)

# Main content
try:
    # Load model and tokenizer
    with st.spinner("Loading model... This might take a moment."):
        tokenizer, model_path, model_name = initialize_model()
    
    # Input prompt
    prompt = st.text_input("Enter your prompt or theme:", 
                          placeholder="e.g., 'Love is like a' or 'Rain in the night'")
    
    # Generate button
    if st.button("Generate Lyrics", use_container_width=True):
        if prompt:
            with st.spinner("Generating lyrics..."):
                # Generate text
                generated_text = generate_text(
                    model_path,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
                
                # Format the output
                lines = re.split(r'(?<=[.!?])\s+|\n', generated_text.strip())
                formatted = "\n".join(line.strip() for line in lines if line.strip())
                
                # Display in a nice format
                st.markdown("### Generated Lyrics:")
                st.markdown(f'<div class="generated-text">{formatted}</div>', unsafe_allow_html=True)
                
                # Add a copy button
                st.button("Copy to Clipboard", 
                         on_click=lambda: st.write(f"```\n{formatted}\n```"))
        else:
            st.warning("Please enter a prompt to generate lyrics.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Make sure the model files are in the correct directory and the model is properly trained.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using GPT-2 and Streamlit</p>
    </div>
""", unsafe_allow_html=True) 
