import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
os.environ["SERPER_API_KEY"]

embeddings_functions = GoogleGenerativeAIEmbeddings(
                                                    model="model/embedding-001", 
                                                    task_type="retrieval_document"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings=embeddings_functions,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

system_prompt = (
    "Vous êtes un assistant pour les tâches de questions-réponses."
    "Utilisez les éléments de contexte récupérés suivants pour répondre à la question." 
    "Si vous ne connaissez pas la réponse, dites que vous ne la connaissez pas." 
    "Utilisez trois phrases maximum et gardez la réponse concise."
    "\n\n"
    "{context}"
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

doc_chain = create_stuff_documents_chain(llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0),
                                         prompt=rag_prompt
)

rag_chain = create_retrieval_chain(retriever, doc_chain)

search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name = "Semantic Tool",
        func = lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Utilise les documents locaux pour répondre aux questions médicales basées sur un vector store."
    ),
    Tool(
        name = "Google Search",
        func=search.run,
        description="Effectue une recherche sur le web pour répondre aux questions médicales non couvertes par les documents locaux."
    )
]

hub.pull("hwchase17/react")


chat_model = ChatMistralAI(api_key="hpKrgxMi1QvDTyevpCLp0OwFBGJk1aV7")

character_prompt = """Answer the following questions as best you can. You have access to the following tools:
{tools}

For any questions requiring tools, you should first search the provided knowledge base. If you don't find relevant information from provided knowledge base, then use Google search to find related information.

To use a tool, you MUST use the following format:
1. Thought: Do I need to use a tool? Yes
2. Action: the action to take, should be one of [{tool_names}]
3. Action Input: the input to the action
4. Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the following format:
1. Thought: Do I need to use a tool? No
2. Final Answer: [your response here], en français

It's very important to always include the 'Thought' before any 'Action' or 'Final Answer'. Ensure your output strictly follows the formats above.

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(character_prompt)

agent = create_react_agent(chat_model, tools, prompt)

memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True, output_key="output")

agent_chain = AgentExecutor(agent=agent,
                            tools=tools,
                            memory=memory,
                            max_iterations=5,
                            handle_parsing_errors=True,
                            verbose=True,
                            )

st.title("Assistant Médical - RoseBleue")
st.write("osez vos questions sur le cancer du sein et le cancer de la prostate")
st.write("Mais sur toute autre maladie en générale, **Rosy** y répondra ;)")
user_input = st.text_input("Posez une question médicale :")
if user_input:
    response = agent_chain.invoke({"input": user_input})["output"]
    st.write(response)