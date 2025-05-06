import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def test_connection():
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")

    print(f"Testing connection to: {DATABASE_URL}")

    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"Successfully connected to PostgreSQL: {version}")

            # Check if pgvector is installed
            result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector';"))
            if result.fetchone():
                print("pgvector extension is installed correctly!")
            else:
                print("ERROR: pgvector extension is not installed!")
        return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False


if __name__ == "__main__":
    test_connection()