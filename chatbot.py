import os
import textwrap
import time
import ast
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transitions import Machine

load_dotenv()


class Application:
    # Define negotiation states
    states = ['greeting', 'operation_selection', 'doctor_selection', 'negotiation', 'bye_bye', 'intervention']

    def __init__(self):

        # Initialize the state machine
        self.machine = Machine(model=self, states=Application.states, initial='greeting')
        self.machine.add_transition(trigger='greeted', source='greeting', dest='operation_selection')
        self.machine.add_transition(trigger='operation_selected', source='operation_selection', dest='doctor_selection')
        self.machine.add_transition(trigger='doctor_selected', source='doctor_selection', dest='negotiation')
        self.machine.add_transition(trigger='negotiation_concluded', source='negotiation', dest='bye_bye')

        self.machine.add_transition(trigger='human_intervention_triggered', source='greeting', dest='intervention')
        self.machine.add_transition(trigger='human_intervention_triggered', source='operation_selection', dest='intervention')
        self.machine.add_transition(trigger='human_intervention_triggered', source='doctor_selection', dest='intervention')
        self.machine.add_transition(trigger='human_intervention_triggered', source='negotiation', dest='intervention')
        self.machine.add_transition(trigger='human_intervention_concluded', source='intervention', dest='bye_bye')

class Doctor:
    def __init__(self, name, surname, specialty, operations_list):
        self.name = name
        self.surname = surname
        self.specialty = specialty
        self.operations_list = operations_list

class Operations:
    def __init__(self, name):
        self.name = name

class RAGChatbot:
    def __init__(self, document_path="Definition and Purpose of Rhinoplasty.docx"):
        """
        Initialize the RAG Chatbot with a specific document

        Args:
            document_path (str): Path to the Word document to use as knowledge base
        """
        # Retrieve the Gemini key
        self.gemini_key = os.getenv('GEMINI_KEY')
        if not self.gemini_key:
            raise ValueError("Please set GEMINI_KEY in your .env file")

        self.model = "gemini-embedding-exp-03-07"
        self.client = genai.Client(api_key=self.gemini_key)

        self.document_path = document_path
        self.embeddings_df = self._generate_embeddings()

    def _extract_text_and_tables_in_order(self):
        """Extracts text and tables from a .docx file while preserving their order."""
        doc = docx.Document(self.document_path)
        full_text = []

        for element in doc.element.body:
            if element.tag.endswith('p'):  # Paragraphs
                para = docx.text.paragraph.Paragraph(element, doc)
                if para.text.strip():
                    full_text.append(para.text)

            elif element.tag.endswith('tbl'):  # Tables
                table = docx.table.Table(element, doc)
                table_text = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_text.append("\t".join(row_text))  # Use tabs for readability
                full_text.append("\n".join(table_text))  # Add full table as text

        return "\n".join(full_text)  # Preserve spacing for better readability

    def _split_text(self, text, chunk_size=1000, overlap=200):
        """Splits text into chunks with overlap to maintain context."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        return splitter.split_text(text)

    def _generate_embedding(self, text):
        """Generate embedding for a given text"""
        response = self.client.models.embed_content(model=self.model, contents=text)
        time.sleep(10)  # Sleep to avoid rate limiting
        return response.embeddings[0].values

    def _generate_embeddings(self):
        """Generate or load embeddings for the document"""
        embeddings_file = "embeddings.csv"

        if os.path.exists(embeddings_file):
            df = pd.read_csv(embeddings_file)
            df['Embeddings'] = df['Embeddings'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
            return df

        # Extract text from document
        text = self._extract_text_and_tables_in_order()

        # Split text into chunks
        chunks = self._split_text(text)

        # Create DataFrame
        df = pd.DataFrame({"chunk_id": range(1, len(chunks) + 1), "text": chunks})
        df['Embeddings'] = df.apply(lambda row: self._generate_embedding(row['text']), axis=1)

        # Save embeddings
        df.to_csv(embeddings_file, index=False, encoding="utf-8")
        print(f"Split document into {len(chunks)} chunks.")

        return df

    def find_best_passage(self, query):
        """Find the most relevant passage for a given query"""
        query_embedding = self.client.models.embed_content(model=self.model, contents=query)
        dot_products = np.dot(np.stack(self.embeddings_df['Embeddings']), query_embedding.embeddings[0].values)
        best_idx = np.argmax(dot_products)

        # Get previous, best, and next indices
        indices = [i for i in [best_idx - 1, best_idx, best_idx + 1] if 0 <= i < len(dot_products)]
        selected_texts = self.embeddings_df.iloc[indices]['text'].values

        return ' '.join(selected_texts)

    def generate_response(self, query, operation, background):
        """Generate a conversational response to the query"""
        relevant_passage = self.find_best_passage(query)
        prompt = textwrap.dedent(f"""You are a helpful and informative bot that answers questions about '{operation}' for your patients using text from the reference passage. 
        Also bear in mind that the patient has a background of '{background}'. Respond in a complete, conversational sentence. Explain complicated concepts simply. 

        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

        ANSWER:
        """)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.1
            )
        )
        return response.text


def main():
    chatbot = RAGChatbot()

    operation1 = Operations("Rhinoplasty")
    doctor1 = Doctor("Selcuk", "Coskun", "Plastic Surgeon",{operation1})

    application = Application()

    while True:
        if application.state == 'greeting':
            print(
                "Hello, I am an AI powered medical procedure assistant. Before starting provide me with your health information.")
            background = input("You: ")
            application.greeted()

        if application.state == 'operation_selection':
            print("Which operation do you want to get informed about?")
            print("Available operation(s) are listed below:")

            print(operation1.name)

            operation_selection = ""
            while  operation_selection != operation1.name:
                operation_selection = input("You: ")

            query = ""
            print(
                "Thank you for your response, now you can ask any question you have in your mind about " + operation_selection + ".")
            print(
                "Or if you want to proceed with selecting one of our available doctors for " + operation_selection + " operation please type proceed.")
            while True:
                query = input("You: ")
                if query.lower() == 'proceed':
                    break

                response = chatbot.generate_response(query, operation_selection, background)
                print(response, "\n")

            application.operation_selected()

        if application.state == 'doctor_selection':
            print("Our available doctors for your desired operation are listed below, please select one of them to proceed.")
            print("Dr. " + doctor1.name + " " + doctor1.surname)

            doctor_selection = ""
            while doctor_selection != doctor1.name:
                doctor_selection = input("You: ")

            application.doctor_selected()

        if application.state == 'negotiation':
            print("Negotiation")
            application.negotiation_concluded()

        if application.state == 'bye bye':
            break


if __name__ == "__main__":
    main()