# with LLM_MODEL = "llama3.2:3b


import os
import fitz # PyMuPDF
import json
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import re
import pandas as pd # Optional: for nice table output
import time

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
PDF_FILE_PATH = r"E:\work\Omran\Transpotation\paper\Navid\3- AI-enabled Criteria Extraction for Multi-Criteria Decision Analysis Using Large Language Models\Article 2\1st test\5-energies-18-01437.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
FAISS_INDEX_DIR = "./faiss_indexes" # Directory to store saved indexes

# --- Ensure Ollama is Running ---
# ...

# --- Functions (load_and_split_pdf, create_rag_chain, parse_llm_output remain the same) ---
# Modify create_vector_store to handle loading/saving

def load_and_split_pdf(pdf_path):
    """Loads PDF content using PyMuPDF and splits it into chunks."""
    print(f"Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") # Extract plain text
        doc.close()

        if not full_text.strip():
            print("Warning: PDF seems empty or text extraction failed.")
            return []

        documents = [Document(page_content=full_text, metadata={"source": pdf_path})]

        print(f"Splitting document into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error loading/splitting PDF {pdf_path}: {e}")
        return []

# create_vector_store function is removed as logic is moved to main block

def create_rag_chain(llm_model_name, vectorstore):
    """Creates the RAG chain with a specific prompt."""
    # --- Use the LATEST refined prompt from the previous message ---
    if vectorstore is None:
        print("Vector store is not available, cannot create RAG chain.")
        return None

    try:
        print(f"Initializing LLM: {llm_model_name}")
        llm = Ollama(
            model=llm_model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )

        # --- Use k=8 or experiment ---
        retriever = vectorstore.as_retriever(search_kwargs={'k': 8, 'fetch_k': 20})

        # --- Further Refined General Prompt Template ---
        prompt_template = ChatPromptTemplate.from_messages([
             ("system", """You are an expert assistant highly skilled at identifying evaluation criteria in academic papers.
Your ONLY task is to carefully read the provided text context and identify the **explicit list and they are using specific named 'criteria'** to evaluate their subjects (e.g., scenarios, technologies, alternatives). Look for phrases like "The criteria used were:", "based on X criteria:", etc., followed by a list.

**Instructions:**
1.  Locate the specific list explicitly label the items as **'criteria'**.
2.  List ONLY the names of these items identified *as criteria*, exactly as they appear in the text. Include labels like (C1), (C2) *only if* they are written directly next to the name in the source text.
3.  Present the identified items as a simple list, separated by newlines or commas.
Focus solely on extracting the names from the list explicitly identified as 'criteria' in the context.
"""),
            # User message remains the same
            ("user", """Based only on the context below, identify the explicit list of criteria used and list ONLY their names, separated by newlines or commas.

Context:
{context}

Criteria list:""")
        ])

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print("RAG chain created successfully.")
        return retrieval_chain

    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        if "connection refused" in str(e).lower():
            print("Hint: Is the Ollama service running and the LLM model available?")
        return None

# --- Revert to a Basic Parser ---
def parse_llm_output(llm_response_text):
    """ Extracts the main text answer, removing potential filler. """
    print("\n--- Raw LLM Output ---")
    raw_text = llm_response_text.strip()
    print(raw_text)
    print("--- End Raw LLM Output ---\n")

    if "No specific evaluation criteria list found" in raw_text:
         print("LLM indicated no criteria list was found.")
         return None # Or return an empty list [] if you prefer

    # Very basic extraction: return the raw text if it's not the "not found" message
    # You might add more sophisticated cleaning later if needed
    return raw_text



# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.perf_counter()

    if PDF_FILE_PATH == "path/to/your/article.pdf":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update the 'PDF_FILE_PATH' variable !!!")
        print("!!! in the script with the actual path to your PDF.    !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # 1. Load and Split PDF
        pdf_chunks = load_and_split_pdf(PDF_FILE_PATH)

        if pdf_chunks:
            # --- Vector Store Handling (Load or Create/Save) ---

            # Create a unique name for the index based on the PDF filename
            pdf_filename = os.path.basename(PDF_FILE_PATH)
            # Simple naming: replace extension. For more robustness, consider hashing filename or path
            index_name = pdf_filename.replace('.pdf', '').replace('.', '_').replace(' ', '_') + "_faiss"
            index_folder_path = os.path.join(FAISS_INDEX_DIR, index_name)

            vector_db = None # Initialize vector_db

            # Ensure the directory for indexes exists
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

            # Instantiate embeddings - needed for both loading and creation
            try:
                 print(f"Initializing embedding model: {EMBEDDING_MODEL}")
                 embeddings = OllamaEmbeddings(
                     model=EMBEDDING_MODEL,
                     base_url=OLLAMA_BASE_URL
                 )
            except Exception as e:
                 print(f"Error initializing embedding model: {e}")
                 embeddings = None # Ensure embeddings is None if initialization fails

            if embeddings is not None:
                 # Check if index exists
                 faiss_index_file = os.path.join(index_folder_path, "index.faiss")
                 faiss_pkl_file = os.path.join(index_folder_path, "index.pkl")

                 if os.path.exists(index_folder_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
                     # Load existing index
                     print(f"Loading existing FAISS index from: {index_folder_path}")
                     try:
                         vector_db = FAISS.load_local(
                             folder_path=FAISS_INDEX_DIR, # Pass the base directory
                             embeddings=embeddings,
                             index_name=index_name, # Pass the specific index name
                             allow_dangerous_deserialization=True # Required for loading pickled data
                         )
                         print("FAISS index loaded successfully.")
                     except Exception as e:
                         print(f"Error loading FAISS index: {e}. Will attempt to recreate.")
                         vector_db = None # Reset on error
                 else:
                     # Create and save new index
                     print("No existing FAISS index found or index is incomplete.")
                     print("Creating FAISS vector store (this might take a while)...")
                     try:
                         vector_db = FAISS.from_documents(pdf_chunks, embeddings)
                         print("FAISS vector store created successfully.")
                         # Save the newly created index
                         print(f"Saving FAISS index to: {index_folder_path}")
                         vector_db.save_local(folder_path=FAISS_INDEX_DIR, index_name=index_name)
                         print("FAISS index saved successfully.")
                     except Exception as e:
                         print(f"Error creating/saving FAISS vector store: {e}")
                         if "connection refused" in str(e).lower():
                             print("Hint: Is the Ollama service running?")
                         vector_db = None # Reset on error

            # --- Proceed only if vector_db is valid ---
            if vector_db:
                # 3. Create RAG Chain
                rag_chain = create_rag_chain(LLM_MODEL, vector_db)

                if rag_chain:
                    # --- ADD THIS BLOCK TO INSPECT RETRIEVED DOCUMENTS ---
                    print("\n--- Running Retriever Separately to Inspect Context ---")
                    temp_retriever = vector_db.as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': 8, 'fetch_k': 20} # Match the settings above
                    )
                    temp_query = "What specific, named factors or dimensions are explicitly listed in the paper's methodology or evaluation sections as the basis for assessing the energy transition scenarios?" # Keep the query
                    
                    try:
                        retrieved_docs = temp_retriever.invoke(temp_query)
                        print(f"Retrieved {len(retrieved_docs)} documents for the LLM:")
                        full_context_for_llm = "\n\n".join([doc.page_content for doc in retrieved_docs]) # Combine like LangChain does

                        # --- Print the context that WILL BE sent to the LLM ---
                        print("\n=== START: Context Sent to LLM ===")
                        print(full_context_for_llm)
                        print("=== END: Context Sent to LLM ===")
                        # --- END OF INSPECTION BLOCK ---

                    except Exception as e:
                        print(f"Error during manual retrieval inspection: {e}")
                    # --------------------------------------------------------

                    # 4. Invoke Chain and Get Results (This will run normally using the context above)
                    print("\nInvoking RAG chain... (This might take a moment)")
                    input_query = temp_query # Ensure the chain uses the same query

                   # --- START OF THE TRY BLOCK TO FIX ---
                    try:
                       response = rag_chain.invoke({"input": input_query})

                       # 5. Parse and Display Output (Simplified)
                       if response and 'answer' in response:
                           # Use the simplified parser which returns raw text or None
                           extracted_text = parse_llm_output(response['answer'])

                           if extracted_text:
                               print("\n--- Extracted Criteria List (Raw Text) ---")
                               print(extracted_text) # Just print the raw list string
                           else:
                               # This covers cases where the parser returned None (e.g., "No list found")
                               print("\nNo criteria list extracted or found by the parser.")
                       else:
                           print("Failed to get a valid response from the RAG chain.")

                   # --- ENSURE THIS EXCEPT BLOCK IS PRESENT ---
                    except Exception as e:
                       # This catches errors during invoke() or parse_llm_output()
                       print(f"\nAn error occurred during RAG chain invocation or parsing: {e}")
                       # Optional: Add more specific error details if needed
                       # import traceback
                       # traceback.print_exc()
                   # --- END OF THE TRY...EXCEPT BLOCK ---

                else: # Corresponds to 'if rag_chain:'
                   print("\nSkipping RAG chain invocation due to setup errors.")
            else: # Corresponds to 'if vector_db:'
                print("\nSkipping RAG chain creation due to vector store errors.")
        else: # Corresponds to 'if pdf_chunks:'
           print("\nSkipping vector store creation due to loading/splitting errors.")

   # --- Calculate and Print Runtime ---
   # ... (runtime calculation) ...
end_time = time.perf_counter()
duration = end_time - start_time
print(f"\n--- Total Runtime: {duration:.2f} seconds ---")

print("\n--- Script Finished ---")
