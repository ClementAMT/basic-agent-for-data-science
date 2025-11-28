from agent import run_explorer_agent, run_summarizer_agent

file = "StudentsPerformance.csv"

inputs = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Analyze this dataset. "
                f"File path: {file}. "
                "Provide: summary, column data types, missing values, correlation and outlier report."
                "Give me only bullet points with clear and concise information."
            ),
        }
    ]
}



if __name__ == "__main__":
    explorer_result  = run_explorer_agent(inputs)
    summarizer_inputs = {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Here is the analysis of my dataset:\n{explorer_result}\n"
                    "Please summarize it in plain text, using short, clear bullet points."
                ),
            }
        ]
    }
    plain_summary = run_summarizer_agent(summarizer_inputs)
    print(plain_summary)
