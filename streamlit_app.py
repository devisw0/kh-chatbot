import streamlit as st
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain_community.vectorstores import FAISS
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# Set page config
st.set_page_config(
    page_title="Slide Deck Q&A Chatbot",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Slide Deck Q&A Chatbot")
st.markdown("Ask questions about your presentation slides!")

# --- --- ---
# HELPER FUNCTIONS
# --- --- ---
def get_embedding_model():
    """Sets up the AWS Bedrock client for the Titan V2 embedding model."""
    session = boto3.Session(
        profile_name='devan2',
        region_name='us-east-1'
    )
    bedrock_client = session.client('bedrock-runtime')
    client = BedrockEmbeddings(
        client=bedrock_client,  
        model_id="amazon.titan-embed-text-v2:0")
    return client

def get_llm():
    """Sets up the AWS Bedrock client for a generator LLM (Claude 3.7 Sonnet)."""
    session = boto3.Session(
        profile_name="devan2", 
        region_name="us-east-1"
    )
    bedrock_client = session.client('bedrock-runtime')
    llm = ChatBedrock(
        client=bedrock_client,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        model_kwargs={"temperature": 0.1}
    )
    return llm

# --- --- ---
# INITIALIZE SESSION STATE
# --- --- ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    vector_store_path = "my_slide_index"
    
    if not os.path.exists(vector_store_path):
        st.error(f"Vector store not found at {vector_store_path}. Please run `build_index.py` first.")
        st.stop()
    
    with st.spinner("Initializing chatbot... This may take a moment."):
        # Load models and vector store
        embeddings = get_embedding_model()
        llm = get_llm()
        
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Define the RAG tool
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """
            Retrieve relevant slides from the vector store to help answer a query.
            This tool is for searching the document knowledge base.
            """
            retrieved_docs = retriever.invoke(query)
            
            serialized_content = "\n\n".join(
                f"Source Slide: {doc.metadata.get('slide_number', 'N/A')}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            
            return serialized_content, retrieved_docs
        
        tools = [retrieve_context]
        
        system_prompt = (
            "You are a helpful assistant for answering questions about a slide deck. "
            "You have access to one tool called 'retrieve_context'.\n"
            
            "Here are your rules:\n"
            "1. For general conversation (like 'hello' or 'how are you'), you MUST answer directly without using the tool.\n"
            "2. For *any* question about the presentation, you MUST use the 'retrieve_context' tool to find the information.\n"
            "3. The tool will give you 'Content' from the slides. You MUST base your final answer *only* on this 'Content'.\n"
            "4. If the 'Content' from the tool does not contain the answer, you MUST say 'I'm sorry, that information is not in the slides.' Do not make up an answer."
        )
        
        st.session_state.agent = create_agent(llm, tools, system_prompt=system_prompt)
    
    st.success("Chatbot initialized! Ask me anything about the slides.")

# --- --- ---
# DISPLAY CHAT HISTORY
# --- --- ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                st.write(f"Slide Numbers: {message['sources']}")

# --- --- ---
# CHAT INPUT
# --- --- ---
if prompt := st.chat_input("Ask a question about the slides..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare message history for agent
            chat_history = []
            for msg in st.session_state.messages[:-1]:  # Exclude the current prompt
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Add current prompt
            messages_for_agent = chat_history + [HumanMessage(content=prompt)]
            
            # Stream agent response
            stream_events = st.session_state.agent.stream(
                {"messages": messages_for_agent},
                stream_mode="values",
            )
            
            final_answer = ""
            source_documents = []
            
            for event in stream_events:
                last_message = event["messages"][-1]
                
                if last_message.type == "tool":
                    source_documents = last_message.artifact
                
                if last_message.type == "ai" and not last_message.tool_calls:
                    final_answer = last_message.content
            
            # Display response
            st.markdown(final_answer)
            
            # Display sources if available
            if source_documents:
                source_slides = sorted(list(set(doc.metadata.get('slide_number', 'N/A') for doc in source_documents)))
                with st.expander("📄 View Sources"):
                    st.write(f"Slide Numbers: {source_slides}")
            else:
                source_slides = None
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "sources": source_slides
            })


with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot answers questions about your presentation slides using:
    - **AWS Bedrock** (Claude Sonnet 4.0  & Titan Embeddings)
    - **FAISS** vector store for retrieval
    - **LangChain** for orchestration
    """)
    
    st.divider()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown(f"**Total Messages:** {len(st.session_state.messages)}")