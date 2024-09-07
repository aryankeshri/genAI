# streamlit run main.py
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
# from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.google_palm import GooglePalm
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI

# from create_insert_data_db import database_creater

load_dotenv()
BASE_DIR = Path(os.path.abspath(__file__)).parent
print(BASE_DIR)

few_shots = [
    {'Question' : "how much is the price of the inventory for all small size t_shirts?",
     'SQLQuery' : "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE size = 'S'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "10015"},
    {'Question': "if we sell all Large size Levi t_shirt today with discounts applied. How much revenue our store will generate(Post discount)?",
     'SQLQuery':"SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi' and size='L' group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id",
     'SQLResult': "Result of the SQL query",
     'Answer': "3346"},
    {'Question': "How many white color t_shirts of Levi?" ,
     'SQLQuery' : """SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'""",
     'SQLResult': "Result of the SQL query",
     'Answer': "97"}
]
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

### my sql based instruction prompt
mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""


def get_db():
    uri = f'sqlite:///./atliq_tshirts.db'
    return SQLDatabase.from_uri(uri)


db = get_db()
llm = GoogleGenerativeAI(model="gemini-pro")
# llm = GoogleGenerativeAI(google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name=model_name)

to_vectorize = [" ".join(value.values()) for value in few_shots]
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
training_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)
training_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)
few_shot_prompt = FewShotPromptTemplate(
    example_selector=training_selector,
    example_prompt=training_prompt,
    prefix=mysql_prompt,
    suffix="Question: {input}",
    input_variables=["input", "table_info", "top_k"],
    # These variables are used in the prefix and suffix
)
chain = SQLDatabaseChain.from_llm(llm, db, verbose=True,
                                  return_intermediate_steps=True,
                                  prompt=few_shot_prompt)

st.title("Gen AI Using Google AI")
question = st.text_input(f"Ask your question.")
submit = st.button("OK")

if submit:
    ans = chain(question)
    print(type(ans))
    print(ans)
    import json

    with open('data.json', 'w') as f:
        json.dump(ans, f, indent=2)

    # ans = json.load(ans)
    st.subheader(ans['result'])
    st.write(ans["intermediate_steps"][1])
