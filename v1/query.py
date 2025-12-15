import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock 
from langchain_community.vectorstores import FAISS
import os

# --- --- ---
# NEW IMPORTS (from the docs you sent)
# --- --- ---
from langchain.tools import tool
from langchain.agents import create_agent 
from langchain_core.messages import HumanMessage, AIMessage



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
        model_kwargs={"temperature": 0.3}
    )
    return llm


def main():
    vector_store_path = "my_slide_index"
    
    if not os.path.exists(vector_store_path):
        print(f"Error: Vector store not found at {vector_store_path}")
        print("Please run `build_index.py` first.")
        return
    
    #making client instances
    print("Initializing models...")
    embeddings = get_embedding_model()
    llm = get_llm()
    
    print(f"Loading vector store from {vector_store_path}...")
    vector_store = FAISS.load_local(
        vector_store_path, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    
    #making retriever for vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Finds Top 5

    # RAG Tool
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


    
    tools = [retrieve_context]
    

    system_prompt = (
    "You are a helpful assistant for answering questions about a slide deck. "
    "You have access to one tool called 'retrieve_context'.\n"
    
    "Here are your rules:\n"
    "1. For general conversation (like 'hello' or 'how are you'), answer directly without using the tool.\n"
    "2. For questions about the presentation, use the 'retrieve_context' tool.\n"
    "3. IMPORTANT: If the user asks about allergies, sensitive skin, or safety:\n"
    "   - First search with 'allergen safety'\n"
    "   - If that doesn't give a good answer, try 'sensitive skin dermatological'\n"
    "   - Then try 'non allergen suitable'\n"
    "   This helps find information even if the text has unusual spacing.\n"
    "4. Base your answer ONLY on the retrieved content.\n"
    "5. If you still can't find the answer after trying different searches, say 'I'm sorry, that information is not in the slides.'"
    )

    #creating agent
    agent = create_agent(llm, tools, system_prompt=system_prompt)

    print("--- --- ---")
    print("Chatbot is ready! Type your question and press Enter.")
    print("Type 'exit' to quit.")
    print("--- --- ---")

    
    chat_history = []

# chat loop
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        print("\nThinking...")
        
        #full chat history
        messages_for_agent = chat_history + [HumanMessage(content=query)]
        
        # .stream instead of invoke to view all steams
        stream_events = agent.stream(
            {"messages": messages_for_agent},
            stream_mode="values",
        )

        final_answer = ""
        source_documents = []

        # We loop through all the steps the agent takes
        for event in stream_events:
            last_message = event["messages"][-1]
            
        
            
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

        # updating chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=final_answer))


if __name__ == "__main__":
    main()