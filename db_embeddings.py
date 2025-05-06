import os
import time
import docx
import numpy as np
import random
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vector_db")
Base = declarative_base()


# Define the embedding model
class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer)
    text = Column(Text)
    embedding = Column(Vector(3072))  # Updated to 3072 dimensions for Gemini embeddings

    def __repr__(self):
        return f"<DocumentEmbedding(id={self.id}, chunk_id={self.chunk_id})>"


def extract_text_from_docx(file_path):
    """Extract text and tables from a Word document"""
    doc = docx.Document(file_path)
    full_text = []
    for element in doc.element.body:
        if element.tag.endswith('p'):
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                full_text.append(para.text)
        elif element.tag.endswith('tbl'):
            table = docx.table.Table(element, doc)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append("\t".join(cells))
            full_text.append("\n".join(rows))
    return "\n".join(full_text)


def split_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def generate_embedding_with_retry(client, model, text, max_retries=5, base_delay=5):
    """Generate embeddings with retry logic for rate limits"""
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(model=model, contents=text)
            return response.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Much more aggressive exponential backoff with jitter
                wait_time = (base_delay * (2 ** attempt)) + (random.random() * 10)
                print(f"Rate limited. Waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            else:
                print(f"Unexpected error: {e}")
                raise  # Re-raise other exceptions

    # If we get here, all retries failed
    raise Exception(f"Failed to generate embedding after {max_retries} retries")


def get_last_processed_chunk(session):
    """Get the last processed chunk ID from the database"""
    last_chunk = session.query(DocumentEmbedding).order_by(DocumentEmbedding.chunk_id.desc()).first()
    return last_chunk.chunk_id if last_chunk else 0


def setup_database_and_embeddings(document_path):
    """Create tables and generate embeddings for document"""
    # Initialize API client
    api_key = os.getenv("GEMINI_KEY")
    client = genai.Client(api_key=api_key)
    model = "gemini-embedding-exp-03-07"

    # Create database engine
    engine = create_engine(DATABASE_URL)

    # Drop the table if it exists (because we need to change the vector dimension)
    Base.metadata.drop_all(engine)

    # Create tables with new schema
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    db_session = Session()

    # Extract and process text
    print(f"Extracting text from {document_path}...")
    text = extract_text_from_docx(document_path)
    chunks = split_text(text)
    print(f"Document split into {len(chunks)} chunks.")

    # Generate and store embeddings
    print("Generating embeddings (this may take a while)...")

    # More conservative processing - start with a longer delay
    time_between_calls = 10  # seconds between API calls

    for i in range(1, len(chunks) + 1):
        chunk = chunks[i - 1]
        print(f"Processing chunk {i}/{len(chunks)}")

        try:
            embedding_vector = generate_embedding_with_retry(client, model, chunk, max_retries=5, base_delay=10)

            # Check embedding dimension
            print(f"Embedding dimension: {len(embedding_vector)}")

            embedding = DocumentEmbedding(
                chunk_id=i,
                text=chunk,
                embedding=embedding_vector
            )
            db_session.add(embedding)

            # Commit after each chunk
            db_session.commit()
            print(f"Successfully processed chunk {i}")

            # Sleep between API calls to avoid rate limits
            sleep_time = time_between_calls + (random.random() * 4)  # Add jitter
            print(f"Waiting {sleep_time:.2f} seconds before next call")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error on chunk {i}: {e}")
            # If we hit too many errors, increase delay time
            time_between_calls += 2
            print(f"Increased delay to {time_between_calls} seconds")
            db_session.rollback()  # Roll back on error
            time.sleep(15)  # Longer cooldown period

    db_session.close()
    print(f"Finished processing embeddings.")


if __name__ == "__main__":
    document_path = "Definition and Purpose of Rhinoplasty.docx"
    setup_database_and_embeddings(document_path)