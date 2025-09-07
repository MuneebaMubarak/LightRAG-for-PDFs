import os
import asyncio
import logging
import logging.config
from pathlib import Path
from typing import List
import PyPDF2
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

WORKING_DIR = "./dickens"
PDF_FOLDER = "./pdfs"  # Folder containing PDF files


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1} of {pdf_path}: {e}")
                    continue
        
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return ""
            
        return text
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return ""


def get_pdf_files(folder_path: str) -> List[str]:
    """Get all PDF files from the specified folder"""
    pdf_folder = Path(folder_path)
    if not pdf_folder.exists():
        print(f"PDF folder '{folder_path}' does not exist!")
        return []
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return []
    
    return [str(pdf_file) for pdf_file in pdf_files]


async def process_multiple_pdfs(rag: LightRAG, pdf_files: List[str]) -> None:
    """Process multiple PDF files and insert them into RAG"""
    total_files = len(pdf_files)
    print(f"\nProcessing {total_files} PDF files...")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{total_files}: {os.path.basename(pdf_path)}")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if pdf_text:
            # Add filename as header for better context
            document_text = f"=== Document: {os.path.basename(pdf_path)} ===\n\n{pdf_text}"
            
            # Insert the document into RAG
            try:
                await rag.ainsert(document_text)
                print(f"✓ Successfully processed: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(pdf_path)}: {e}")
        else:
            print(f"✗ Skipped {os.path.basename(pdf_path)} (no text extracted)")


async def initialize_rag():
    """Initialize the RAG instance"""
    

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file in the same directory with:")
        print("OPENAI_API_KEY=your-openai-api-key")
        print("\nOr set the environment variable by running:")
        print("export OPENAI_API_KEY='your-openai-api-key'")
        return

    # Create necessary directories
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    
    if not os.path.exists(PDF_FOLDER):
        print(f"Creating PDF folder: {PDF_FOLDER}")
        os.mkdir(PDF_FOLDER)
        print(f"Please add your PDF files to the '{PDF_FOLDER}' folder and run the script again.")
        return

    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        print("Cleaning up old data files...")
        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✓ Deleted: {file}")

        # Initialize RAG instance
        print("\nInitializing RAG instance...")
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test text: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}")

        # Get PDF files and process them
        pdf_files = get_pdf_files(PDF_FOLDER)
        if not pdf_files:
            return

        # Process all PDF files
        await process_multiple_pdfs(rag, pdf_files)

        print("\n" + "="*50)
        print("PDF PROCESSING COMPLETED")
        print("="*50)

        # Perform different types of queries
        queries = [
            ("explain the overview of income tax in Australia?", "naive"),
            ("Exempt income refers to what?", "local"),
            ("give a summary about the framework of Australian tax law ", "global"),
            ("Brief about capital gains tax (CGT)?", "hybrid")
        ]

        for question, mode in queries:
            print(f"\n{'='*25}")
            print(f"Query mode: {mode}")
            print(f"Question: {question}")
            print('='*25)
            try:
                query_params = QueryParam(
                    mode=mode,
                    top_k=5,
                    enable_rerank=False
                )
                result = await rag.aquery(question, param=query_params)
                print(result)
            except Exception as e:
                print(f"Error in {mode} query: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'rag' in locals():
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")