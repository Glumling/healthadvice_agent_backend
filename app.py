# agent_backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS                    # <─ easy CORS  ✔  :contentReference[oaicite:2]{index=2}
from langchain import hub                       # pulls hosted prompt  ✔  :contentReference[oaicite:3]{index=3}
from langchain_openai import AzureChatOpenAI    # Azure wrapper ✔  :contentReference[oaicite:4]{index=4}
from langchain.agents import create_openai_tools_agent, AgentExecutor

from tools import (
    calculate_bmi, free_db_search, exercises_by_muscle,
    recipes_by_ingredient, product_by_barcode,
    estimate_calories, target_hr,
    unit_convert, water_goal, macro_split, workout_split,   # NEW
    one_rep_max, vo2max, rpe_table, hiit_plan,
    stretch_routine, sleep_debt,
)


# add the imports
from tools_extra import (
    web_search, api_get_tool, python_repl_tool,
    calc_tool, docs_qa,
)


# --- Flask setup ---
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "https://project-healthadvice.vercel.app"}})

# --- LangChain model + tools ---
model = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
)

tools = [
    calculate_bmi, free_db_search, exercises_by_muscle,
    recipes_by_ingredient, product_by_barcode,
    estimate_calories, target_hr,
    unit_convert, water_goal, macro_split, workout_split,   # NEW
    one_rep_max, vo2max, rpe_table, hiit_plan,
    stretch_routine, sleep_debt,
    web_search,      # web search
    api_get_tool,     # generic JSON API
    python_repl_tool, # code execution
    calc_tool,        # calculator
    docs_qa,          # PDF knowledge base
]

prompt = hub.pull("hwchase17/openai-tools-agent")   # ✔  :contentReference[oaicite:5]{index=5}
agent  = create_openai_tools_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# --- single /chat endpoint ---
@app.post("/chat")
def chat():
    msg = request.get_json(force=True)["message"]   # getting JSON body ✔  :contentReference[oaicite:6]{index=6}
    result = executor.invoke({"input": msg})
    return jsonify(reply=result["output"])

if __name__ == "__main__":
    app.run(port=8000, debug=True)
