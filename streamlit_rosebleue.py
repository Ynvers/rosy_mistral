import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain

# Charger les variables d'environnement
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Initialisation des embeddings et de la base vectorielle
embeddings_functions = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
vectorstore = FAISS.load_local(
    "vectorstore.db", 
    embeddings=embeddings_functions,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# Initialisation du modèle LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

template = """
Tu es RoseBleue, 
un assistant intelligent développé par TechSeed Academy dans le cadre de la sensibilisation des mois d'octobre Rose pour le cancer du sein et de novembre Bleue pour celui de la prostate,
spécialisé dans la fourniture d'informations claires, précises et fiables sur le cancer du sein et le cancer de la prostate. 
En tant que RoseBleue, ta mission est de :

- Répondre aux questions des utilisateurs sur les symptômes, le dépistage, les options de traitement et les conseils de prévention pour ces cancers, **en équilibrant les informations sur le cancer du sein et le cancer de la prostate** pour offrir le même niveau de rigueur et d'importance pour chaque sujet.
- **Fournir des réponses concises, de trois à quatre phrases maximum**, qui se concentrent exclusivement sur la question posée.
- Fournir des informations compréhensibles, bienveillantes et adaptées aux profils des utilisateurs, en tenant compte de facteurs comme l’âge, le sexe, et les antécédents médicaux.
- Orienter les utilisateurs vers des professionnels de santé compétents et encourager le suivi médical en rappelant l’importance de la prévention et du dépistage.

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

Historique de la session : {context}

Question de l'utilisateur : {input}

Réponds de manière concise et adaptée en utilisant les informations de contexte et l'historique de la session pour offrir une réponse ciblée. Oriente l'utilisateur vers des professionnels de santé uniquement si la question nécessite un suivi médical.
"""

prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

# Historique de la session
context = []

# Fonction pour obtenir une réponse de l'assistant
def get_response(input_text):
    context.append(f"Utilisateur : {input_text}")
    ai_msg = chain.invoke({
        "input": input_text,
        "context": "\n".join(context)
    })
    context.append(f"RoseBleue : {ai_msg['answer']}")
    return ai_msg["answer"]

# Interface utilisateur Streamlit
st.title("Assistant Médical - RoseBleue")
st.write("Posez vos questions sur le cancer du sein et le cancer de la prostate.")

# Champ de saisie pour la question de l'utilisateur
user_input = st.text_input("Votre question :")

# Bouton pour obtenir une réponse
if st.button("Poser la question"):
    if user_input:
        # Affichage de la réponse de l'assistant
        response = get_response(user_input)
        st.write("Réponse :")
        st.write(response)
    else:
        st.write("Veuillez entrer une question.")
