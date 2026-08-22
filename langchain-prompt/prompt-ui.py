import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate,load_prompt

# 1. Page Configuration
st.set_page_config(
    page_title="Research Tool Assistant",
    page_icon="🔬",
    layout="centered"
)

# 2. Load Environment Variables
load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("⚠️ Hugging Face API Token missing! Please add HUGGINGFACEHUB_API_TOKEN to your .env file.")
    st.stop()

# 3. Streamlit UI
st.title("🔬 Research Tool Assistant")
st.caption("Powered by HuggingFace & LangChain")

st.markdown("Select paper details below to generate structured research summaries.")

# Dropdown inputs
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Prompt Template Definition
template=load_prompt('template.json')
# Submit Button Logic
if st.button("Generate Response", type="primary"):
    if paper_input == "Select...":
        st.warning("Please select a research paper before generating a response.")
    else:
        # 1. Format the prompt using .invoke() - returns a StringPromptValue
        formatted_prompt = template.invoke({
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        })

        with st.spinner("Connecting to HuggingFace LLM and generating response..."):
            try:
                # 2. Initialize with conversational task for provider compatibility
                llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    task="text-generation",
                   
                    temperature=0.7
                )

                # 3. Wrap with ChatHuggingFace
                model = ChatHuggingFace(llm=llm)

                # 4. Invoke model with the formatted PromptValue
                response = model.invoke(formatted_prompt)

                # 5. Extract string content from AIMessage
                output_text = response.content

                # Display Results
                st.subheader("Results")
                if output_text and output_text.strip():
                    st.markdown(output_text)
                else:
                    st.error("Received an empty response from the model.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")