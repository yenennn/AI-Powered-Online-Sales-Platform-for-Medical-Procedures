import os
import textwrap
import time
import ast
import numpy as np
import pandas as pd
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transitions import Machine

load_dotenv()

def llm_analyze_sentiment(text):
    gemini_key = os.getenv('GEMINI_KEY')
    if not gemini_key:
        raise ValueError("Please set GEMINI_KEY in your .env file")
    client = genai.Client(api_key=gemini_key)
    prompt = f"Analyze the sentiment of the following text and respond with a single word (positive, neutral, or negative):\n\n\"{text}\""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=10, temperature=0.0)
    )
    sentiment = response.text.strip().lower()
    if sentiment not in ['positive', 'neutral', 'negative']:
        sentiment = 'neutral'
    return sentiment

class Negotiation:
    def __init__(self, operation):
        self.operation = operation
        self.history = []
    def record_round(self, round_number, user_offer, counter_offer):
        timestamp = datetime.datetime.now().isoformat()
        self.history.append((round_number, user_offer, counter_offer, timestamp))
    def get_history(self):
        return self.history

class Application:
    states = ['greeting', 'operation_selection', 'doctor_selection', 'negotiation', 'bye_bye', 'intervention']
    def __init__(self):
        self.machine = Machine(model=self, states=Application.states, initial='greeting')
        self.machine.add_transition(trigger='greeted', source='greeting', dest='operation_selection')
        self.machine.add_transition(trigger='operation_selected', source='operation_selection', dest='doctor_selection')
        self.machine.add_transition(trigger='doctor_selected', source='doctor_selection', dest='negotiation')
        self.machine.add_transition(trigger='negotiation_concluded', source='negotiation', dest='bye_bye')
        self.machine.add_transition(trigger='human_intervention_triggered', source=['greeting', 'operation_selection', 'doctor_selection', 'negotiation'], dest='intervention')
        self.machine.add_transition(trigger='human_intervention_concluded', source='intervention', dest='bye_bye')

class Doctor:
    def __init__(self, name, surname, specialty, operations_list):
        self.name = name
        self.surname = surname
        self.specialty = specialty
        self.operations_list = operations_list

class Operations:
    def __init__(self, name, base_price, negotiation_min, negotiation_max):
        self.name = name
        self.base_price = base_price
        self.negotiation_min = negotiation_min
        self.negotiation_max = negotiation_max

class RAGChatbot:
    def __init__(self, document_path="Definition and Purpose of Rhinoplasty.docx"):
        self.gemini_key = os.getenv('GEMINI_KEY')
        if not self.gemini_key:
            raise ValueError("Please set GEMINI_KEY in your .env file")
        self.model = "gemini-embedding-exp-03-07"
        self.client = genai.Client(api_key=self.gemini_key)
        self.document_path = document_path
        self.embeddings_df = self._generate_embeddings()
    def _extract_text_and_tables_in_order(self):
        doc = docx.Document(self.document_path)
        full_text = []
        for element in doc.element.body:
            if element.tag.endswith('p'):
                para = docx.text.paragraph.Paragraph(element, doc)
                if para.text.strip():
                    full_text.append(para.text)
            elif element.tag.endswith('tbl'):
                table = docx.table.Table(element, doc)
                table_text = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_text.append("\t".join(row_text))
                full_text.append("\n".join(table_text))
        return "\n".join(full_text)
    def _split_text(self, text, chunk_size=1000, overlap=200):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ".", " "])
        return splitter.split_text(text)
    def _generate_embedding(self, text):
        response = self.client.models.embed_content(model=self.model, contents=text)
        time.sleep(10)
        return response.embeddings[0].values
    def _generate_embeddings(self):
        embeddings_file = "embeddings.csv"
        if os.path.exists(embeddings_file):
            df = pd.read_csv(embeddings_file)
            df['Embeddings'] = df['Embeddings'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
            return df
        text = self._extract_text_and_tables_in_order()
        chunks = self._split_text(text)
        df = pd.DataFrame({"chunk_id": range(1, len(chunks) + 1), "text": chunks})
        df['Embeddings'] = df.apply(lambda row: self._generate_embedding(row['text']), axis=1)
        df.to_csv(embeddings_file, index=False, encoding="utf-8")
        print(f"Split document into {len(chunks)} chunks.")
        return df
    def find_best_passage(self, query):
        query_embedding = self.client.models.embed_content(model=self.model, contents=query)
        dot_products = np.dot(np.stack(self.embeddings_df['Embeddings']), query_embedding.embeddings[0].values)
        best_idx = np.argmax(dot_products)
        indices = [i for i in [best_idx - 1, best_idx, best_idx + 1] if 0 <= i < len(dot_products)]
        selected_texts = self.embeddings_df.iloc[indices]['text'].values
        return ' '.join(selected_texts)
    def generate_response(self, query, operation, background):
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
            config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1)
        )
        return response.text

def main():
    operation1 = Operations("Rhinoplasty", base_price=5000, negotiation_min=4000, negotiation_max=5500)
    doctor1 = Doctor("Selcuk", "Coskun", "Plastic Surgeon", {operation1})
    application = Application()
    chatbot = RAGChatbot()
    if application.state == 'greeting':
        print("Hello, I am an AI powered medical procedure assistant. Please provide me with your health information.")
        background = input("You: ")
        application.greeted()
    if application.state == 'operation_selection':
        print("Which operation do you want to get informed about?")
        print("Available operation(s):")
        print(operation1.name)
        operation_selection = ""
        while operation_selection != operation1.name:
            operation_selection = input("You: ")
        print("Thank you for your response. Now, you can ask any questions you have about " + operation_selection + ".")
        print("Or type 'proceed' to select a doctor for this operation.")
        while True:
            query = input("You: ")
            if query.lower() == 'proceed':
                break
            response = chatbot.generate_response(query, operation_selection, background)
            print(response, "\n")
        application.operation_selected()
    if application.state == 'doctor_selection':
        print("Our available doctor for your desired operation is:")
        print("Dr. " + doctor1.name + " " + doctor1.surname)
        doctor_selection = ""
        while doctor_selection != doctor1.name:
            doctor_selection = input("Please type the doctor's first name to select: ")
        application.doctor_selected()
    if application.state == 'negotiation':
        print(f"Starting price negotiation for the operation: {operation1.name}")
        current_offer = operation1.base_price
        print(f"Initial price is ${current_offer:.2f}.")
        negotiation_session = Negotiation(operation1)
        accepted = False
        negotiation_round = 1
        while not accepted:
            user_input = input("Enter your counter-offer or type 'accept' to accept the current offer:")
            sentiment = llm_analyze_sentiment(user_input)
            print(f"(LLM Sentiment: {sentiment})")
            if user_input.lower() == "accept":
                print(f"You accepted the final price of ${current_offer:.2f}.")
                accepted = True
            else:
                try:
                    user_offer = float(user_input)
                except ValueError:
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", user_input)
                    if not match:
                        print("Please enter a valid number.")
                        continue
                    else:
                        user_offer = float(match.group())
                if user_offer < operation1.negotiation_min:
                    print("This price option is not suitable, try typing a different price.")
                    continue
                if user_offer > current_offer:
                    print("Your offer is higher than the current offer. Please try negotiating for a lower price.")
                    continue
                else:
                    midpoint = (current_offer + user_offer) / 2.0
                    if sentiment == "negative":
                        if negotiation_round == 1:
                            reduction_ratio = 0.27
                        else:
                            reduction_ratio = 0.27
                        counter_offer = midpoint - (midpoint - operation1.negotiation_min) * reduction_ratio
                    else:
                        counter_offer = midpoint
                    counter_offer = max(counter_offer, user_offer)
                    negotiation_session.record_round(negotiation_round, user_offer, counter_offer)
                    if negotiation_round > 1 and abs(counter_offer - user_offer) <= 0.05 * counter_offer:
                        print(f"Great! Your offer of ${user_offer:.2f} is acceptable.")
                        current_offer = user_offer
                        accepted = True
                    else:
                        print(f"I can offer it at ${counter_offer:.2f}.")
                        current_offer = counter_offer
                    negotiation_round += 1
        print("Negotiation concluded.")
        print("\nNegotiation History:")
        history = negotiation_session.get_history()
        for row in history:
            print(f"Round: {row[0]}, Your Offer: {row[1]}, Counter-Offer: {row[2]}, Timestamp: {row[3]}")
        application.negotiation_concluded()
    if application.state == 'bye_bye':
        print("Thank you for using our service. Goodbye!")

if __name__ == "__main__":
    main()
