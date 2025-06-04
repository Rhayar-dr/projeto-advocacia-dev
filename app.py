import logging
import os
import streamlit as st
from langchain_core.messages import SystemMessage # Certifique-se que está importado
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.chat_models import ChatDatabricks # Removido
from langchain_openai import ChatOpenAI # Adicionado
from langchain_openai import OpenAIEmbeddings # Adicionado
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# ### MUDANÇA ###: Usar Pinecone
from langchain_pinecone import PineconeVectorStore # Adicionado. Pode ser `from langchain_community.vectorstores import Pinecone` dependendo da versão.
# from langchain.vectorstores import AzureSearch # Removido
import pinecone # Adicionado

load_dotenv()

st.set_page_config(
    page_title="Assistente IA - Documentos Advocatícios", # ### MUDANÇA ###: Título da página
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ### MUDANÇA ###: Remover unicodedata e remover_acentos se não for usar em filtros de metadados do Pinecone.
# Vamos manter por enquanto, pode ser útil para metadados se você os usar.
import unicodedata
def remover_acentos(txt):
    if not isinstance(txt, str):
        return txt
    return ''.join(
        c for c in unicodedata.normalize('NFKD', txt)
        if not unicodedata.combining(c)
    )

# ### MUDANÇA ###: Usar OpenAIEmbeddings
# from langchain_community.embeddings import DatabricksEmbeddings # Removido
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Adicionado

# ### MUDANÇA ###: Usar ChatOpenAI
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.7) # Adicionado, usando gpt-4o-mini como padrão
router_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.0) # Adicionado

# --- LangGraph State and Nodes ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: State):
    # A IA geral, sem RAG, ainda precisa da instrução de não vazar dados.
    # Considerar adicionar um system prompt aqui também, ou garantir que o prompt geral seja aplicado.
    # Por simplicidade, vamos focar o prompt de proteção de dados no RAG, mas idealmente seria em ambos.
    # Poderíamos ter um prompt de sistema mais genérico aqui:
    system_message_content = (
        "Você é um assistente de IA focado em advocacia. "
        "Responda de forma concisa e profissional. "
        "Lembre-se de NUNCA citar informações pessoais, nomes de partes, valores específicos ou quaisquer outros dados sensíveis "
        "caso tenha acesso a algum contexto que possa conter isso. Priorize a confidencialidade."
    )
    # Prepend system message if not already there or to reinforce
    # Esta é uma forma simples de adicionar. Em um sistema mais complexo, você gerenciaria melhor a adição do prompt de sistema.
    messages_for_llm = [SystemMessage(content=system_message_content)] if not any(isinstance(m, SystemMessage) for m in state["messages"][-3:]) else []
    messages_for_llm.extend(state["messages"][-3:])

    return {"messages": [llm.invoke(messages_for_llm)]}


def rag_node(state: State):
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        return {"messages": [AIMessage(content="Não consegui identificar a pergunta para a busca nos documentos.")]}

    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return {"messages": [AIMessage(content="Vectorstore não inicializado. Por favor, configure o sistema.")]}

    try:
        docs = st.session_state.vectorstore.similarity_search(
            query=question,
            k=3
        )
    except Exception as e:
        logger.error(f"Erro na busca por similaridade no Pinecone: {e}")
        return {"messages": [AIMessage(content=f"Desculpe, ocorreu um erro ao buscar nos documentos: {e}")]}

    sources = [doc.metadata.get('source', doc.metadata.get('file_name', 'Documento Desconhecido')) for doc in docs]
    context = "\n---\n".join([f"Fonte: {src}\nTrecho do Documento Exemplo:\n{doc.page_content}" for src, doc in zip(sources, docs)])

    # ### MUDANÇA ###: Prompt principal da IA para RAG (instrução completa)
    # Adicionada a parte "Em vez disso..." que é crucial.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um expert em montar documentos de advocacia. Seu objetivo é auxiliar na criação e estruturação de documentos legais. "
         "Analise os trechos dos documentos de referência abaixo para entender a estrutura, o tom e as cláusulas comuns. "
         "IMPORTANTE: NUNCA cite informações pessoais, nomes de partes (reclamante, reclamada, advogados, etc.), números de processo, valores específicos, datas exatas de eventos "
         "ou quaisquer outros dados que possam ser confidenciais ou identificar um caso específico contido nos exemplos. "
         "Em vez disso, use os exemplos como base para fornecer modelos genéricos, explicar como uma cláusula deve ser redigida ou qual a estrutura geral de um pedido. "
         "Por exemplo, se um documento diz 'João da Silva pede X', você deve dizer 'O requerente pode pedir X' ou 'Uma cláusula de pedido pode ser estruturada como...'. "
         "Seu foco é em COMO pedir ou montar documentos, usando a ESTRUTURA e a IDEIA dos exemplos, não o CONTEÚDO SENSÍVEL. "
         "Se a pergunta for muito genérica e não houver contexto relevante, informe que precisa de mais detalhes sobre o tipo de documento ou situação. "
         "Seja ético e priorize a confidencialidade e a generalidade da informação."
         ),
        ("human", "Contexto dos Documentos de Exemplo:\n{context}\n\nCom base nesse contexto e na sua expertise, responda à seguinte pergunta sobre como montar um documento legal:\n{question}")
    ])

    response_message = llm.invoke(prompt_template.format(context=context, question=question))

    final_response_sources = [doc.metadata.get('source', doc.metadata.get('file_name', 'desconhecido')) for doc in docs]
    final_response_content = response_message.content.strip()
    if docs:
        final_response_content += "\n\nEstruturas e ideias baseadas em exemplos de documentos como:\n" + "\n".join(f"- {src}" for src in set(final_response_sources))
    else:
        final_response_content += "\n\nNão foram encontrados documentos de exemplo específicos para esta consulta nos arquivos disponíveis."

    return {"messages": [AIMessage(content=final_response_content)]}


# ### MUDANÇA ###: Lógica do route_decision e prompt do roteador
def route_decision(state: State):
    # Encontra a última HumanMessage no estado para basear a decisão
    last_human_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message_content = msg.content
            break

    if not last_human_message_content:
        logger.warning("Nenhuma HumanMessage encontrada no estado para decisão de roteamento. Default para END.")
        # Se não houver HumanMessage, provavelmente não deveríamos estar aqui,
        # mas ir para END é uma saída segura.
        return END

    # ### MUDANÇA ###: Prompt do roteador simplificado
    # Objetivo: "NÃO" para RAG apenas para saudações/agradecimentos, "SIM" para todo o resto.
    prompt_template_obj = ChatPromptTemplate.from_messages([ # Renomeado para evitar conflito com `prompt_template_str` não definido.
        ("system",
         """Você deve responder estritamente com SIM ou NÃO.
Responda NÃO se a mensagem do usuário for APENAS uma saudação (exemplos: "Oi", "Olá", "Bom dia", "Boa tarde", "Boa noite", "E aí") ou uma expressão de agradecimento (exemplos: "Obrigado", "Valeu", "Grato", "Obrigada").
Para TODAS as outras perguntas, afirmações ou qualquer outro tipo de entrada do usuário, responda SIM.
O objetivo é usar a base de conhecimento (RAG) para quase tudo, exceto interações sociais muito simples. Em caso de dúvida, responda SIM."""),
        ("human", "{question}")
    ])

    formatted_prompt = prompt_template_obj.format(question=last_human_message_content)
    response = router_llm.invoke(formatted_prompt).content.strip().lower()

    logger.info(f"Routing decision based on: '{last_human_message_content}'. Router LLM response: '{response}'")

    if response == "sim":
        logger.info("Routing decision: RAG (Documentos Advocatícios)")
        return "rag"
    else: # Inclui "não" e qualquer outra resposta inesperada (fallback para não-RAG)
        logger.info("Routing decision: Chatbot Direto / END")
        return END


builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("rag", rag_node)
builder.add_conditional_edges("chatbot", route_decision, {"rag": "rag", END: END})
builder.add_edge("rag", END)
builder.set_entry_point("chatbot")
graph = builder.compile(checkpointer=MemorySaver())

# --- Global Data & Configs ---
# ### MUDANÇA ###: Adaptar tema e FAQs
THEME_LEGAL_DOCS = "Advocacia - Assistente de Documentos"
FAQ_DATA_LEGAL_DOCS = {
    "Tipos Comuns de Documentos": [
        "Me forneça um exemplo de estruturar uma petição inicial",
        "Quais cláusulas essenciais em um contrato de prestação de serviços?",
        "Como redigir uma notificação extrajudicial?",
        "Pode me dar um exemplo de pedido de danos morais (sem valores ou nomes)?"
    ],
    "Dicas de Redação": [
        "Qual o tom adequado para um recurso?",
        "Como ser mais persuasivo na argumentação jurídica?"
    ]
}
# ### MUDANÇA ###: Remover LOCALIDADES_POR_ESTADO
# LOCALIDADES_POR_ESTADO = { ... }

# --- Session State Initialization ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
st.session_state.selected_theme = THEME_LEGAL_DOCS # ### MUDANÇA ###
# ### MUDANÇA ###: Remover estados e localidades do session_state
# if "selected_estado" not in st.session_state:
# st.session_state.selected_estado = None
# if "selected_localidade" not in st.session_state:
# st.session_state.selected_localidade = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False # Será True após inicializar Pinecone
if "prefilled_question" not in st.session_state:
    st.session_state.prefilled_question = ""
# if "selected_localidade_for_rag" not in st.session_state: # Removido
# st.session_state.selected_localidade_for_rag = ""

# --- Pinecone Initialization (Global) ---
def initialize_pinecone_vectorstore():
    if st.session_state.vectorstore is None:
        try:
            logger.info(f"Attempting to initialize Pinecone with index: {os.getenv('PINECONE_INDEX_NAME')}")
            
            # Inicializa o cliente Pinecone (substituído pela não necessidade de init() global na v3+)
            # pinecone.init( # Removido para SDK v3+
            # api_key=os.getenv("PINECONE_API_KEY"),
            # environment=os.getenv("PINECONE_ENVIRONMENT") # Se sua SDK ainda usar environment
            # )
            # Para SDK v3.x do pinecone-client, a inicialização do cliente é diferente
            # pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # Ou use PINECONE_HOST se aplicável
            # index = pc.Index(os.getenv('PINECONE_INDEX_NAME'), host=os.getenv('PINECONE_HOST'))

            # Usando diretamente Langchain PineconeVectorStore que lida com a conexão
            st.session_state.vectorstore = PineconeVectorStore.from_existing_index(
                index_name=os.getenv("PINECONE_INDEX_NAME"),
                embedding=embeddings,
                # namespace="meu-namespace-legal" # Opcional: se você usa namespaces no Pinecone
                # text_key="text" # Opcional: se o campo de texto no Pinecone não for "text"
            )
            st.session_state.chat_ready = True
            logger.info("Pinecone Vectorstore inicializado com sucesso. Chat está pronto.")
            return True
        except Exception as e:
            st.sidebar.error(f"Erro ao conectar ao Pinecone: {e}")
            logger.error(f"Pinecone init error: {e}", exc_info=True)
            st.session_state.chat_ready = False
            return False
    return st.session_state.chat_ready # Retorna o estado atual se já inicializado

# --- Sidebar UI ---
def display_sidebar():
    st.sidebar.title("⚙️ Configurações do Assistente") # ### MUDANÇA ###
    st.sidebar.markdown(f"**Tema Fixo:** {THEME_LEGAL_DOCS}") # ### MUDANÇA ###

    # ### MUDANÇA ###: Removida seleção de Estado e Localidade.
    # A inicialização do Pinecone agora é o principal para "chat_ready".
    if not st.session_state.chat_ready:
        st.sidebar.warning("Inicializando conexão com a base de conhecimento jurídica...")
        if initialize_pinecone_vectorstore():
            st.sidebar.success("Conexão estabelecida!")
            st.rerun() # Força o rerun para atualizar o estado da UI
        else:
            st.sidebar.error("Falha ao conectar. Verifique as configurações e logs.")
            return # Impede o resto da UI da sidebar de carregar se não estiver pronto

    st.sidebar.title("Histórico de Conversas")
    # ### MUDANÇA ###: Botão de nova conversa adaptado
    if st.sidebar.button("Nova Conversa Jurídica", key="new_chat_button", use_container_width=True, disabled=not st.session_state.chat_ready):
        start_new_chat_logic()
        st.rerun()

    sorted_chat_ids = sorted(st.session_state.conversations.keys(), reverse=True)
    for chat_id_iter in sorted_chat_ids:
        chat_data = st.session_state.conversations[chat_id_iter]
        button_label = chat_data["name"]
        # ### MUDANÇA ###: Lógica de "context_ready" simplificada pois não há mais contexto dinâmico (estado/localidade)
        # is_current_chat_context_ready = st.session_state.chat_ready

        if st.sidebar.button(button_label, key=f"chat_{chat_id_iter}", use_container_width=True, type="secondary" if st.session_state.current_chat_id != chat_id_iter else "primary"):
            st.session_state.current_chat_id = chat_id_iter
            st.session_state.prefilled_question = ""
            st.rerun()

    # ### MUDANÇA ###: FAQs adaptadas
    st.sidebar.subheader("Perguntas Frequentes (Documentos)")
    for tema_faq, perguntas in FAQ_DATA_LEGAL_DOCS.items():
        with st.sidebar.expander(tema_faq):
            for pergunta in perguntas:
                if st.button(f"↪️ {pergunta}", key=f"faq_legal_{tema_faq}_{pergunta.replace(' ','_')}", disabled=not st.session_state.chat_ready):
                    st.session_state.prefilled_question = pergunta
                    if not st.session_state.current_chat_id and st.session_state.chat_ready:
                        start_new_chat_logic()
                    st.rerun()

def start_new_chat_logic():
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    chat_id = f"legal_chat_{datetime.now().timestamp()}"
    # ### MUDANÇA ###: Nome da conversa simplificado
    st.session_state.current_chat_id = chat_id
    st.session_state.conversations[chat_id] = {
        "name": f"Jurídico - {timestamp}",
        "messages": [],
        "langgraph_messages": [],
        # "context_estado": None, # Removido
        # "context_localidade": None # Removido
    }
    st.session_state.prefilled_question = ""
    logger.info(f"Iniciada nova conversa jurídica: {chat_id}")

# --- Main Chat Area UI ---
st.title(f"💬 Assistente IA: {THEME_LEGAL_DOCS}") # ### MUDANÇA ###
display_sidebar() # Chama a sidebar que agora também tenta inicializar o Pinecone

# ### MUDANÇA ###: Lógica de boas-vindas e prontidão adaptada
if not st.session_state.chat_ready:
    st.warning("O assistente jurídico ainda não está pronto. Verificando conexão com a base de conhecimento...")
    # A sidebar já tenta inicializar, aqui é mais um feedback.
elif not st.session_state.current_chat_id and st.session_state.chat_ready:
    st.info("Bem-vindo! Clique em 'Nova Conversa Jurídica' na barra lateral para iniciar ou selecione uma conversa existente.")


if st.session_state.chat_ready and st.session_state.current_chat_id:
    current_chat_data = st.session_state.conversations[st.session_state.current_chat_id]

    st.markdown(f"### {current_chat_data['name']}")

    # ### MUDANÇA ###: Mensagem de contexto removida/simplificada
    st.caption("Pronto para auxiliar com a estruturação de documentos jurídicos.")

    for msg in current_chat_data["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Digite sua pergunta sobre documentos jurídicos...", key="chat_input_main_legal", disabled=not st.session_state.chat_ready) # ### MUDANÇA ###

    if st.session_state.prefilled_question:
        prompt = st.session_state.prefilled_question
        st.session_state.prefilled_question = ""

    if prompt:
        current_chat_data["messages"].append({"role": "user", "content": prompt})
        # Adicionando SystemMessage ao LangGraph se for a primeira mensagem da thread, para reforçar as instruções.
        # Esta é uma forma. Outra seria o chatbot_node e rag_node sempre adicionarem seu system prompt específico.
        # A lógica de adicionar SystemMessage no chatbot_node é mais robusta para todas as chamadas.
        # Aqui, vamos confiar que o chatbot_node e rag_node já têm os system prompts adequados.
        current_chat_data["langgraph_messages"].append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando base de conhecimento jurídica e processando..."): # ### MUDANÇA ###
                # selected_localidade_for_rag removido
                # graph_input_messages = current_chat_data["langgraph_messages"][-3:] # Mantido: pega as últimas mensagens
                
                # Para garantir que as instruções do sistema sejam passadas corretamente para o grafo,
                # especialmente para a primeira rodada ou se o estado não as persistir da forma esperada:
                current_thread_messages = current_chat_data["langgraph_messages"]
                
                # Adiciona a mensagem de sistema no rag_node e chatbot_node.
                # Não precisa adicionar aqui explicitamente se os nós já fazem isso.
                # Apenas garantimos que as mensagens humanas/IA estão corretas.

                # Para o langgraph, geralmente passamos a lista completa de mensagens da thread atual
                # para que ele possa gerenciar o histórico dentro de seu MemorySaver.
                # A forma como você estava pegando `graph_input_messages` era para limitar o contexto enviado ao LLM diretamente,
                # o que é bom para o LLM, mas o `MemorySaver` do LangGraph beneficia-se do histórico completo.
                # O `add_messages` no `State` cuida de como as mensagens são acumuladas.
                # Então, enviar `current_chat_data["langgraph_messages"]` é mais alinhado com o uso de `MemorySaver`.
                # O `chatbot_node` e `rag_node` podem então decidir se usam `[-3:]` ou outra lógica para o LLM.

                state_input = {"messages": current_chat_data["langgraph_messages"]} # Envia todas as mensagens da thread atual

                ai_response_content = ""
                final_ai_message_object = None
                try:
                    # O `thread_id` é crucial para o `MemorySaver`
                    events = graph.stream(
                        state_input,
                        config={"configurable": {"thread_id": st.session_state.current_chat_id}}
                    )
                    for event in events:
                        # O formato do evento pode variar dependendo de como o grafo é construído e o que os nós retornam.
                        # Vamos assumir que o último nó (chatbot ou rag) que produz uma AIMessage é o que queremos.
                        # A lógica original de pegar `value["messages"][-1]` é geralmente correta se o nó sempre retorna uma lista de mensagens.
                        for node_name, node_output in event.items(): # event é um dict com {node_name: output}
                            if isinstance(node_output, dict) and "messages" in node_output:
                                if node_output["messages"]: # Garante que há mensagens
                                    ai_msg_obj = node_output["messages"][-1] # Pega a última mensagem do nó
                                    if isinstance(ai_msg_obj, AIMessage) and hasattr(ai_msg_obj, 'content'):
                                        ai_response_content = ai_msg_obj.content
                                        final_ai_message_object = ai_msg_obj
                                    elif isinstance(ai_msg_obj, str): # Fallback se o nó retornar string
                                        ai_response_content = ai_msg_obj
                                        final_ai_message_object = AIMessage(content=ai_response_content)
                                    # logger.info(f"Stream event from node {node_name}: {ai_response_content[:50]}...") # Debug
                    st.markdown(ai_response_content)
                except Exception as e:
                    logger.error(f"Error during LangGraph stream for Legal Docs: {e}", exc_info=True)
                    ai_response_content = f"Desculpe, ocorreu um erro ao processar sua solicitação: {e}"
                    final_ai_message_object = AIMessage(content=ai_response_content) # Cria um objeto AIMessage para o erro
                    st.error(ai_response_content)

                if ai_response_content: # Garante que temos algum conteúdo para salvar
                    current_chat_data["messages"].append({"role": "assistant", "content": ai_response_content})
                    if final_ai_message_object:
                        current_chat_data["langgraph_messages"].append(final_ai_message_object)
                    else: # Fallback se final_ai_message_object não foi criado (ex: erro antes de criá-lo)
                        current_chat_data["langgraph_messages"].append(AIMessage(content=ai_response_content))
        st.rerun()
else:
    if st.session_state.chat_ready and not st.session_state.current_chat_id:
        st.info("Clique em 'Nova Conversa Jurídica' na barra lateral para começar.") # ### MUDANÇA ###
    elif not st.session_state.chat_ready:
        st.error("A conexão com a base de conhecimento jurídica não pôde ser estabelecida. Verifique as configurações ou contate o suporte.")
