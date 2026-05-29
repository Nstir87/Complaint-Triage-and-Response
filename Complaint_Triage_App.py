import json
import requests
import streamlit as st
from dotenv import load_dotenv
import os

# -----------------------------------
# Load Environment Variables
# -----------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv()

# Replace with your actual fine-tuned model ID
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:personal::DcutCBnR"

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Customer Complaint Intelligence",
    layout="wide"
)

# -----------------------------------
# Title Section
# -----------------------------------

st.title("Customer Complaint Intelligence System")

st.markdown("""
This application demonstrates a fine-tuned OpenAI model designed to assist with:

- Complaint categorization
- Complaint summarization
- Professional response generation
- Customer support intelligence

The model was fine-tuned using customer support interaction data.
""")

# -----------------------------------
# User Input
# -----------------------------------

complaint_text = st.text_area(
    "Enter Customer Complaint",
    height=200,
    placeholder="Describe the customer complaint here..."
)

# -----------------------------------
# Submit Button
# -----------------------------------

if st.button("Analyze Complaint"):

    if not complaint_text.strip():
        st.warning("Please enter a customer complaint.")
        st.stop()

    with st.spinner("Analyzing complaint..."):

        inference_url = (
            "https://api.openai.com/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": FINE_TUNED_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial institution "
                        "complaint analyst specializing in "
                        "complaint categorization, "
                        "summarization, and response generation. "
                        "Always return valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": complaint_text
                }
            ],
            "temperature": 0.3
        }

        response = requests.post(
            inference_url,
            headers=headers,
            json=payload
        )

        result = response.json()

        # -----------------------------------
        # Error Handling
        # -----------------------------------

        if "error" in result:
            st.error(result["error"]["message"])
            st.stop()

        # -----------------------------------
        # Parse Model Output
        # -----------------------------------

        try:

            model_output = (
                result["choices"][0]
                ["message"]["content"]
            )

            parsed_output = json.loads(model_output)

            # -----------------------------------
            # Display Results
            # -----------------------------------

            st.subheader("Complaint Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Category",
                    parsed_output.get("category", "N/A")
                )

            with col2:
                st.metric(
                    "Subcategory",
                    parsed_output.get("subcategory", "N/A")
                )

            st.subheader("Summary")

            st.write(
                parsed_output.get("summary", "N/A")
            )

            st.subheader("Recommended Response")

            st.write(
                parsed_output.get("response", "N/A")
            )

        except Exception as e:

            st.error(
                "Unable to parse model response."
            )

            st.write(result)

            st.write(str(e))