import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_mistralai import MistralAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver


#Chargements des variables d'environnement
load_dotenv()

# Création du modèle d'embedding pour la tâche demandé
embeddings = MistralAIEmbeddings(
	model="mistral-embed",
)

# Chargement de la base de données vectorielle à l'aide mu modèle d'embeding
vectorstore = FAISS.load_local("vectorstore.db",
								embeddings=embeddings,
								allow_dangerous_deserialization=True
)

# Création du retriever dont le rôle est de trouver les informations pertinentes dans le document
retriever = vectorstore.as_retriever()

# Création de l'outil de recherche dans les documents
@tool(
	"rag_retriever",
	description="Utilise cet outil pour tirer des informations sur les cancer du sein et de la prostate. Il est primoridale"
)
def rag_retriever_func(query: str):
	"""Utilise cet outil pour tirer des informations sur les cancer du sein et de la prostate."""
	return retriever.invoke(query)
# Test effectué pour s'assurer que le retriever marche effectivement
# print(retriever_tool.invoke({"query": "Quesls sont les caractéristques des cancers ?"}))

# initialisation de sous-agent Serper.dev
search = GoogleSerperAPIWrapper()
# Test effectué
# print(search.run("Obama's first name?"))
@tool(
	"search_tool",
	description="Utilise cet outil pour répondre aux questions sur des événements récents ou des sujets d'actualité non couverts par les documents locaux."
)
def call_search_tools(query: str) -> str:
	"""Cherche sur le web"""
	return search.run(query)	

system_prompt = """Tu es RoseBleue, 
un assistant intelligent développé par Nathan ADOHO dans le cadre de la sensibilisation des mois d'octobre Rose pour le cancer du sein et de novembre Bleue pour celui de la prostate,
spécialisé dans la fourniture d'informations claires, précises et fiables sur le cancer du sein et le cancer de la prostate.
Tu dois être conviviale et gentille pour maître à l'aise les utilisateurs, un peu comme s'ils paralaient à un ami

**Règles de Comportement** :
1. **Priorité à la source locale** : Pour les questions sur le cancer du sein et de la prostate, tu dois **toujours** utiliser l'outil `rag_retriever` en premier. Si l'information est introuvable ou nécessite une mise à jour, utilise l'outil `search_internet`.
2. **Réponses concises et bienveillantes** : Limite chaque réponse finale à **trois ou quatre phrases directes**. Ne donne pas de détails supplémentaires non demandés.
3. **Orientation Santé** : Incite les utilisateurs à consulter un spécialiste ou à appeler des services de santé adaptés si la question nécessite une expertise directe.
4. **Langue** : Toutes les réponses finales doivent être en **français**.

**Exemples d'information à rappeler si pertinent** :
- Importance de l'autopalpation et de la mammographie (sein).
- Rôle du test PSA et de l'examen rectal (prostate).
"""

#Création du main_agent
tools = [rag_retriever_func, call_search_tools]

llm_agent = ChatGoogleGenerativeAI(
	model="gemini-2.5-flash", # Utilisez le nom standard sans le préfixe
	temperature=0.1 # optionnel
)

checkpointer = MemorySaver()

agent = create_agent(
	model = llm_agent,
	tools = tools,
	system_prompt = system_prompt,
	checkpointer = checkpointer
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Qui es-tu ?"}]},
    {"configurable": {"thread_id": "1"}}  # Thread ID for conversation tracking
)

print(result["messages"][-1].content[0]["text"])