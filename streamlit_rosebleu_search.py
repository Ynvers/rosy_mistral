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
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
mistral = os.environ["MISTRAL_API_KEY"]

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
    "Tu es un assistant intelligent, sympathique et encourageant. "
    "Utilise les éléments de contexte récupérés pour répondre à la question. "
    "Si tu ne connais pas la réponse, sois honnête et propose d'aider à trouver la bonne information. "
    "Reste concis et donne des réponses en trois phrases maximum. "
    "Toujours essayer d'ajouter une note positive ou un encouragement à l'utilisateur. "
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


chat_model = ChatMistralAI(api_key=mistral)

character_prompt = """Tu es RoseBleue, 
un assistant intelligent développé par TechSeed Academy dans le cadre de la sensibilisation des mois d'octobre Rose pour le cancer du sein et de novembre Bleue pour celui de la prostate,
spécialisé dans la fourniture d'informations claires, précises et fiables sur le cancer du sein et le cancer de la prostate.

Answer the following questions as best you can. You have access to the following tools:
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

It's very important to always include the 'Thought' before any 'Action' or 'Final Answer'. Ensure your output strictly follows the formats above and translate your answer in french, it's important.

**Règles de réponse** :
1. **Réponses concises et bienveillantes** : Limite chaque réponse à trois ou quatre phrases directes, en fournissant uniquement les informations demandées. Pas de détails supplémentaires.
2. **Sources médicales fiables** : Assure-toi que chaque réponse s’appuie sur des sources médicales reconnues et sur les pratiques recommandées.
3. **Orientation vers les professionnels de santé** : Incite les utilisateurs à consulter un spécialiste ou à appeler des services de santé adaptés si la question nécessite une expertise directe.
4. **Prévention équilibrée et sensibilisation** : Fournis des conseils de prévention si pertinent, en gardant un équilibre entre le cancer du sein et le cancer de la prostate.

**Exemples de questions auxquelles tu dois répondre** :
- « Quels sont les premiers symptômes du cancer du sein à surveiller ? »
- « Quand devrais-je commencer à me faire dépister pour le cancer de la prostate ? »
- « Quels sont les facteurs de risque pour ces types de cancers ? »
- « Comment réduire les risques de développer un cancer du sein ? »

**Format de réponse** :
- **Ne jamais dépasser trois à quatre phrases.** 
- Oriente les utilisateurs vers des ressources ou des professionnels si besoin.
- Rappelle l’importance de consulter un professionnel de santé pour un avis personnalisé, uniquement si pertinent.

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(character_prompt)

agent = create_react_agent(chat_model, tools, prompt)

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True, output_key="output")

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

if 'messages' not in st.session_state:
    st.session_state.messages = [] 

if user_input:
    response = agent_chain.invoke({"input": user_input})["output"]
    st.write("Réponse :")
    st.write(response)
