# -*- coding: utf-8 -*-
"""
Created on Sun May  4 06:20:24 2025

@author: asus
"""

import os
import fitz # PyMuPDF
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
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
OLLAMA_BASE_URL = "http://localhost:11434" # Adjust if Ollama runs elsewhere
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
PDF_FILE_PATH = r"E:\work\Omran\Transpotation\paper\Navid\3- AI-enabled Criteria Extraction for Multi-Criteria Decision Analysis Using Large Language Models\Article 2\1st test\5-energies-18-01437.pdf" # <--- IMPORTANT: SET PDF PATH HERE

CHUNK_SIZE = 1500 # Experiment with this size
CHUNK_OVERLAP = 250 # Experiment with overlap

# --- Ensure Ollama is Running ---
# Make sure Ollama service is active and models are pulled:
# ollama run llama3:8b
# ollama run nomic-embed-text
# ---


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

        # Wrap the text in Langchain Document structure for the splitter
        # Using one large document initially, splitter will handle chunking
        documents = [Document(page_content=full_text, metadata={"source": pdf_path})]

        print(f"Splitting document into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True, # Helps in tracing back chunks if needed
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error loading/splitting PDF {pdf_path}: {e}")
        return []

def create_vector_store(chunks, embedding_model_name):
    """Creates embeddings and FAISS vector store."""
    if not chunks:
        print("No chunks to process for vector store creation.")
        return None
    try:
        print(f"Initializing embedding model: {embedding_model_name}")
        embeddings = OllamaEmbeddings(
            model=embedding_model_name,
            base_url=OLLAMA_BASE_URL
        )
        print("Creating FAISS vector store...")
        # This can take some time depending on the number of chunks
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("FAISS vector store created successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Add more specific error handling for Ollama connection issues if needed
        if "connection refused" in str(e).lower():
            print("Hint: Is the Ollama service running?")
        return None

def create_rag_chain(llm_model_name, vectorstore):
    """Creates the RAG chain with a specific prompt."""
    if vectorstore is None:
        print("Vector store is not available, cannot create RAG chain.")
        return None

    try:
        print(f"Initializing LLM: {llm_model_name}")
        llm = Ollama(
            model=llm_model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1 # Lower temperature for more deterministic extraction
        )

        # Define the retriever (how to fetch relevant chunks)
        # k=10 : Retrieve top 10 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        # --- Tailored Prompt Template ---
        # This is the most critical part for precision
        # --- Revised General Prompt Template ---
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant specialized in extracting specific evaluation information from academic research papers.

Your primary task is to:
**Identify the distinct set of factors, metrics, or dimensions explicitly defined and used by the authors within their evaluation framework to assess, compare, or rank specific subjects like policies, scenarios, technologies, or alternatives.**

**Clarification:** You are looking for the ***topics*** or ***aspects*** being evaluated (e.g., economic impact, environmental effect, technical feasibility, social considerations), NOT the mathematical methods, algorithms, or procedural steps used to perform the evaluation (e.g., calculating distances, applying fuzzy logic, weighting).

**Context & Location:**
*   These factors/metrics/dimensions are often presented as a specific list within the **Methodology, Experimental Setup, or Evaluation sections** of the provided text context. Focus your search there (e.g., Section 3.3 in some papers).

**Exclusions (CRITICAL - DO NOT EXTRACT THESE):**
*   **General methodological approaches** (e.g., 'fuzzy logic', 'MCDM', 'TOPSIS', 'PROMETHEE', 'SAW', 'Z-numbers', 'statistical analysis', 'scenario approach').
*   **Steps or calculations *within* an evaluation method** (e.g., 'distance calculation', 'pairwise comparison', 'weighting', 'normalization', 'ranking', 'similarity measure', 'Z-reasoning').
*   **Broad categories mentioned only abstractly** (e.g., in the introduction/abstract) unless explicitly part of the specific list of factors/metrics/dimensions used in the evaluation framework.
*   **The overall paper topic, research question, or general goals** (e.g., 'energy transition', 'sustainability') unless listed as a specific evaluation factor *used to assess the scenarios/alternatives*.
*   **References to criteria used in *other* papers** mentioned in literature reviews.
*   **Input variables or data sources** unless explicitly stated they are also used as evaluation criteria for the output/results.

**Output Format:**
*   Format the output clearly for **each** distinct factor/metric/dimension found using EXACTLY this structure, with each item on a new line:
    *   **Criteria Name:** [The specific name given in the text]
    *   **Sub-Criteria:** [Any specific sub-components mentioned, or N/A]
    *   **Description:** [Brief description from the text, or N/A]

**Desired Output Example (Structure Only):**
    Criteria Name: [Name of Factor/Metric 1 Found in Text]
    Sub-Criteria: [Sub-components Found, or N/A]
    Description: [Description Found, or N/A]
    Criteria Name: [Name of Factor/Metric 2 Found in Text]
    Sub-Criteria: [Sub-components Found, or N/A]
    Description: [Description Found, or N/A]

**Handling No Matches:**
*   If no specific factors/metrics/dimensions matching these detailed instructions are found in the provided context, respond ONLY with the exact phrase: "No specific evaluation criteria found matching the instructions in the provided text sections."

**Conciseness:**
*   Do not add any introductory, concluding, or explanatory sentences outside of the requested formatted output or the "No Matches" response. Stick strictly to the defined format.
"""),
            ("user", """Based *only* on the context below, extract the evaluation factors/metrics/dimensions according to the detailed instructions provided in the system message.

Context:
{context}

Extracted Criteria:""")
        ])


        # Create a chain to process retrieved documents (stuff them into the prompt)
        document_chain = create_stuff_documents_chain(llm, prompt_template)

        # Create the final retrieval chain
        # This chain first retrieves documents, then passes them to the document_chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        print("RAG chain created successfully.")
        return retrieval_chain

    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        if "connection refused" in str(e).lower():
            print("Hint: Is the Ollama service running and the LLM model available?")
        return None

def parse_llm_output(llm_response_text):
    """
    Parses the structured text output from the LLM into a list of dictionaries.
    This is a basic parser and might need adjustments based on LLM output variations.
    """
    print("\n--- Raw LLM Output ---")
    print(llm_response_text)
    print("--- End Raw LLM Output ---\n")

    if "No specific evaluation criteria found" in llm_response_text:
        return []

    criteria_list = []
    # Use regex to find blocks starting with "Criteria Name:"
    # This regex assumes each criterion block starts clearly and might need refinement
    pattern = re.compile(
        r"Criteria Name:\s*(.*?)\s*\n"
        r"Sub-Criteria:\s*(.*?)\s*\n"
        r"Description:\s*(.*?)(?=\n\s*\n|\nCriteria Name:|$)",
        re.DOTALL | re.IGNORECASE
    )

    matches = pattern.findall(llm_response_text)

    if not matches:
         # Fallback: Try parsing line by line if regex fails
        lines = llm_response_text.strip().split('\n')
        current_criterion = {}
        for line in lines:
            line = line.strip()
            if line.lower().startswith("criteria name:"):
                if current_criterion: # Save previous one if exists
                    criteria_list.append(current_criterion)
                current_criterion = {"Criteria Name": line.split(":", 1)[1].strip()}
            elif line.lower().startswith("sub-criteria:") and current_criterion:
                current_criterion["Sub-Criteria"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("description:") and current_criterion:
                current_criterion["Description"] = line.split(":", 1)[1].strip()
            elif current_criterion and "Description" in current_criterion: # Append to description if it continues
                 current_criterion["Description"] += " " + line
        if current_criterion: # Add the last one
             criteria_list.append(current_criterion)

    else: # Regex worked
        for match in matches:
            criteria_list.append({
                "Criteria Name": match[0].strip(),
                "Sub-Criteria": match[1].strip(),
                "Description": match[2].strip()
            })

    # Clean up potential 'N/A' inconsistencies
    for item in criteria_list:
        for key in item:
            if item[key].lower() == 'n/a':
                item[key] = 'N/A'

    return criteria_list


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.perf_counter() # <--- RECORD START TIME

    if PDF_FILE_PATH == "path/to/your/article.pdf":
         # ... (error message for path)
         pass # Added pass to make the block valid if the condition is met
    else:
        # 1. Load and Split PDF
        pdf_chunks = load_and_split_pdf(PDF_FILE_PATH)

        if pdf_chunks:
            # 2. Create Vector Store
            vector_db = create_vector_store(pdf_chunks, EMBEDDING_MODEL)

            if vector_db:
                # 3. Create RAG Chain
                # --- Use the REVISED GENERAL Prompt Template from the previous message here ---
                rag_chain = create_rag_chain(LLM_MODEL, vector_db) # Ensure create_rag_chain uses the right prompt

                if rag_chain:
                    # 4. Invoke Chain and Get Results
                    print("\nInvoking RAG chain... (This might take a moment)")
                    # --- Use the REVISED GENERAL input_query from the previous message ---
                    input_query = "What specific, named factors or dimensions are explicitly listed in the paper's methodology or evaluation sections as the basis for assessing the energy transition scenarios?"
                    try:
                        response = rag_chain.invoke({"input": input_query})

                        # 5. Parse and Display Output
                        if response and 'answer' in response:
                            parsed_criteria = parse_llm_output(response['answer'])

                            if parsed_criteria:
                                print("\n--- Extracted Evaluation Criteria ---")
                                try:
                                    df = pd.DataFrame(parsed_criteria)
                                    for col in ["Criteria Name", "Sub-Criteria", "Description"]:
                                        if col not in df.columns:
                                            df[col] = 'N/A'
                                    df = df[["Criteria Name", "Sub-Criteria", "Description"]]
                                    print(df.to_string(index=False))
                                except ImportError:
                                     print("Pandas not installed. Displaying as list of dictionaries:")
                                     for item in parsed_criteria:
                                         print(f"- Criteria: {item.get('Criteria Name', 'N/A')}")
                                         print(f"  Sub: {item.get('Sub-Criteria', 'N/A')}")
                                         print(f"  Desc: {item.get('Description', 'N/A')}")
                            else:
                                print("\nNo structured criteria could be parsed from the LLM response, or none were found.")
                                if "No specific evaluation criteria found" not in response['answer']:
                                     print("Check the 'Raw LLM Output' above for details.")
                        else:
                            print("Failed to get a valid response from the RAG chain.")

                    except Exception as e:
                        print(f"\nAn error occurred during RAG chain invocation: {e}")
                        # ... (error handling)

                # ... (rest of the error handling/skipping logic) ...
            # ...
        # ...

    # --- Calculate and Print Runtime ---
    end_time = time.perf_counter() # <--- RECORD END TIME
    duration = end_time - start_time # <--- CALCULATE DURATION
    print(f"\n--- Total Runtime: {duration:.2f} seconds ---") # <--- PRINT DURATION
    


    print("\n--- Script Finished ---")
    