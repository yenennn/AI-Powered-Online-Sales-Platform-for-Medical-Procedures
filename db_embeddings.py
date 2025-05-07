import os
import docx
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:yenenn@localhost:5432/vector_db")
Base = declarative_base()


# Define the embedding model with 768 dimensions for standard SentenceTransformer models
class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer)
    text = Column(Text)
    embedding = Column(Vector(768))  # Updated to 768 dimensions for SentenceTransformer models

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


def setup_database_and_embeddings(document_path, model_name='all-mpnet-base-v2'):
    """Create tables and generate embeddings for document using SentenceTransformer"""
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create database engine
    engine = create_engine(DATABASE_URL)

    # Drop existing tables and create new ones
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    db_session = Session()

    # Extract and process text
    print(f"Extracting text from {document_path}...")
    text = extract_text_from_docx(document_path)
    chunks = split_text(text)
    print(f"Document split into {len(chunks)} chunks.")

    # Generate and store embeddings
    print("Generating embeddings with local model...")

    batch_size = 8  # Process chunks in batches for efficiency
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        # Generate embeddings for the batch
        embeddings = model.encode(batch, convert_to_numpy=True)

        # Store each embedding in the database
        for j, embedding_vector in enumerate(embeddings):
            chunk_index = i + j + 1
            if chunk_index <= len(chunks):
                # Verify the embedding dimension
                if len(embedding_vector) != 768:
                    print(f"Warning: Expected embedding dimension 768, got {len(embedding_vector)}")

                embedding = DocumentEmbedding(
                    chunk_id=chunk_index,
                    text=chunks[chunk_index - 1],
                    embedding=embedding_vector
                )
                db_session.add(embedding)

        # Commit after each batch
        db_session.commit()
        print(f"Successfully processed {len(batch)} chunks (total: {min(i + batch_size, len(chunks))}/{len(chunks)})")

    db_session.close()
    print(f"Finished processing all embeddings without any API limits!")





if __name__ == "__main__":
    document_path = "Definition and Purpose of Rhinoplasty.docx"
    # You can choose from several models:
    # - 'all-mpnet-base-v2' (768 dimensions, best quality)
    # - 'all-MiniLM-L12-v2' (384 dimensions, good quality, faster)
    # - 'all-MiniLM-L6-v2' (384 dimensions, faster but less accurate)
    setup_database_and_embeddings(document_path, model_name='all-mpnet-base-v2')