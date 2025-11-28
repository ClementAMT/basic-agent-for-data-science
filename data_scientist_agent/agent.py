import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import summarize_dataframe, missing_values_report, detect_types, correlations, outliers


# ---------------------------------------
# Load API key
# ---------------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyC87ABzG-LoDunXm8gIu4Bz4DxrF9rXLuI"

# ---------------------------------------
# LLM
# ---------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Tool list
TOOLS = [summarize_dataframe, missing_values_report, detect_types, correlations, outliers]


# ---------------------------------------
# Create Agent
# ---------------------------------------
agent = create_agent(
    model=llm,
    tools=TOOLS,            
    system_prompt="You are a helpful data analysis assistant.",
)


# ---------------------------------------
# Runner
# ---------------------------------------
def run_explorer_agent(inputs: dict, agent=agent):
    final_answer = None
    for update in agent.stream(inputs, stream_mode="updates"):
        if "model" in update:
            msgs = update["model"]["messages"]
            if msgs:
                final_answer = "".join(
                    block["text"] 
                    for block in msgs[-1].content 
                    if block["type"] == "text"
                )
    return final_answer



summarizer_agent = create_agent(
    model=llm,
    system_prompt=(
        "You are a helpful data assistant. "
        "Given the analysis of a dataset, write a short plain-language summary "
        "describing the dataset and key insights. "
        "Do not include code or raw data, only a concise human-readable text."
    ),
)

def run_summarizer_agent(inputs: dict, agent=summarizer_agent):
    final_answer = None
    for update in agent.stream(inputs, stream_mode="updates"):
        if "model" in update:
            msgs = update["model"]["messages"]
            if msgs:
                final_answer = msgs[-1].content 
    return final_answer