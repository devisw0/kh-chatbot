import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock # This is correct
from langchain_community.vectorstores import FAISS
import os

# --- --- ---
# NEW IMPORTS (from the docs you sent)
# --- --- ---
from langchain.tools import tool # The tool decorator
from langchain.agents import create_agent # The new agent factory
from langchain_core.messages import HumanMessage, AIMessage

# (We no longer need any 'langchain.chains' imports)

# --- --- ---
# 1. SET UP THE EMBEDDING MODEL (Unchanged)
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

# --- --- ---
# 2. SET UP THE GENERATOR LLM (Unchanged)
# --- --- ---
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
# 3. MAIN SCRIPT LOGIC (All New Agent Logic)
# --- --- ---
def main():
    vector_store_path = "my_slide_index"
    
    if not os.path.exists(vector_store_path):
        print(f"Error: Vector store not found at {vector_store_path}")
        print("Please run `build_index.py` first.")
        return
    
    # 1. Load your models and vector store
    print("Initializing models...")
    embeddings = get_embedding_model()
    llm = get_llm()
    
    print(f"Loading vector store from {vector_store_path}...")
    vector_store = FAISS.load_local(
        vector_store_path, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    
    # Create the retriever (the "Librarian")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Finds Top 3

    # --- --- ---
    # 2. DEFINE YOUR RAG TOOL (as per your documentation)
    # We define the tool *inside* main so it can access the 'retriever'
    # --- --- ---
    @tool(response_format="content_and_artifact") # From your docs
    def retrieve_context(query: str):
        """
        Retrieve relevant slides from the vector store to help answer a query.
        This tool is for searching the document knowledge base.
        """
        print(f"--- TOOL: Searching for: '{query}' ---")
        retrieved_docs = retriever.invoke(query)
       
        # 1. The content (a string, for the LLM to read)
        serialized_content = "\n\n".join(
            f"Source Slide: {doc.metadata.get('slide_number', 'N/A')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        
        # 2. The artifacts (the raw data, for us to use)
        # We return both, as shown in the docs
        return serialized_content, retrieved_docs

    # --- --- ---
    # 3. CREATE THE AGENT (as per your documentation)
    # --- --- ---
    
    # Create the list of tools the agent can use
    tools = [retrieve_context]
    
    # Create the system prompt (our "Guardrail")
    system_prompt = (
        "You are a helpful assistant for answering questions about a slide deck. "
        "You have access to one tool called 'retrieve_context'.\n"
        
        "Here are your rules:\n"
        "1. For general conversation (like 'hello' or 'how are you'), you MUST answer directly without using the tool.\n"
        "2. For *any* question about the presentation, you MUST use the 'retrieve_context' tool to find the information.\n"
        "3. The tool will give you 'Content' from the slides. You MUST base your final answer *only* on this 'Content'.\n"
        "4. If the 'Content' from the tool does not contain the answer, you MUST say 'I'm sorry, that information is not in the slides.' Do not make up an answer."
    )

    # This is the new "factory" function from your docs
    agent = create_agent(llm, tools, system_prompt=system_prompt)

    print("--- --- ---")
    print("Chatbot is ready! Type your question and press Enter.")
    print("Type 'exit' to quit.")
    print("--- --- ---")

    # This list will be our bot's memory
    chat_history = []

# 4. RUN THE CHAT LOOP
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        print("\nThinking...")
        
        # We must pass the *full* message history to the agent
        messages_for_agent = chat_history + [HumanMessage(content=query)]
        
        # We use .stream() to watch the agent work
        stream_events = agent.stream(
            {"messages": messages_for_agent},
            stream_mode="values",
        )

        final_answer = ""
        source_documents = []

        # We loop through all the steps the agent takes
        for event in stream_events:
            last_message = event["messages"][-1]
            
            # --- --- ---
            # NEW, SAFER CHECKS using .type
            # --- --- ---
            
            # Check if the last message is an AI message and has tool calls
            if last_message.type == "ai" and last_message.tool_calls:
                tool_name = last_message.tool_calls[0]['name']
                tool_query = last_message.tool_calls[0]['args']['query']
                print(f"--- Calling Tool: {tool_name}('{tool_query}') ---")

            # Check if the last message is a Tool message (it contains the results)
            if last_message.type == "tool":
                source_documents = last_message.artifact
            
            # Check if it's an AI message and has *no* tool calls (it's the final answer)
            if last_message.type == "ai" and not last_message.tool_calls:
                final_answer = last_message.content

        print(f"\nAssistant: {final_answer}")
        
        # Print sources (if the tool was called)
        if source_documents:
            print("\nSource (Slide Numbers):")
            source_slides = set(doc.metadata.get('slide_number', 'N/A') for doc in source_documents)
            print(f"  {sorted(list(source_slides))}\n")
        else:
            print("\n(No sources used for this response.)\n")

        # Update our manual chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=final_answer))


if __name__ == "__main__":
    main()