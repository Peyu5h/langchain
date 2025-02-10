import os
import shutil
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "Dracula.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

def read_text_file(file_path):
    """Read text file with different encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not read file with any of the encodings: {encodings}")

def initialize_vector_store():
    """Initialize the vector store"""
    print("Creating new vector store...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find Dracula text file at {file_path}")
    
    try:
        text_content = read_text_file(file_path)
        
        documents = [Document(page_content=text_content)]
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents)
        
        print(f"Split document into {len(docs)} chunks")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        db = Chroma.from_documents(
            docs, 
            embeddings, 
            persist_directory=persistent_directory
        )
        print("Vector store created successfully!")
        return db
        
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise

def query_database(db, query):
    """Query the database and get response"""
    try:
        retriever = db.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 5,            # Retrieve 5 documents
                "fetch_k": 8,      # Fetch 8 documents initially
                "lambda_mult": 0.7  # Diversity vs relevance trade-off
            }
        )
        relevant_docs = retriever.invoke(query)
        
        print("\n--- Relevant Documents ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content.strip())
        
        context = "\n\n".join([doc.page_content.strip() for doc in relevant_docs])
        combined_input = f"""
        Question: {query}
        
        Context from documents:
        {context}
        
        Please answer the question based only on the provided context. 
        If the answer isn't clear from the context, say "I'm not sure."
        Provide a concise answer with relevant quotes if available.
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        messages = [
            SystemMessage(content="You are a helpful assistant specialized in analyzing literature."),
            HumanMessage(content=combined_input)
        ]
        
        result = model.invoke(messages)
        return result.content
        
    except Exception as e:
        return f"Error querying database: {str(e)}"

def main():
    try:
        # removing existing vector store if it exists
        if os.path.exists(persistent_directory):
            print("Removing existing vector store...")
            shutil.rmtree(persistent_directory)
        

        db = initialize_vector_store()
        
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            
            if query.lower() == 'quit':
                break
            
            print(f"\nQuery: {query}")
            response = query_database(db, query)
            print("\n--- Generated Response ---")
            print(response)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        print("\nThank you for using the Dracula QA system!")

if __name__ == "__main__":
    main()