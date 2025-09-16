import csv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# Loading the environment variables
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

# Initializing the Groq model.
llm = ChatGroq(model = "openai/gpt-oss-120b")


# Setting up the prompt for the chains.
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes customer support calls."),
    ("user", "Summarize this call sentences:\n\n{transcript}")
])

sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyzer."),
    ("user", "Identify the customer sentiment (Positive / Neutral / Negative) from this call:\n\n{transcript}")
])

# Setting up the Streamlit UI
st.set_page_config(page_title="Mini Tech Challenge", layout="wide")
st.title("Call Transcript Analyzer")

transcript = st.text_area("Paste customer call transcript here:", height=200)

if st.button("Analyze"):
    if transcript.strip():

        # Setting up the chains.
        summary_chain = summary_prompt | llm
        summary = summary_chain.invoke({"transcript": transcript}).content

        sentiment_chain = sentiment_prompt | llm
        sentiment = sentiment_chain.invoke({"transcript": transcript}).content

        st.subheader("===Results===")
        st.write("**Transcript:**", transcript)
        st.write("**Summary:**", summary)
        st.write("**Sentiment:**", sentiment)

        csv_file = "call_analysis.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode = "a", newline = "") as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(["Transcript", "Summary", "Sentiment"])
            
            writer.writerow([transcript, summary, sentiment])

        st.success(f"Saved results to {csv_file}")
    else:
        st.warning("Please enter a transcript first.")
