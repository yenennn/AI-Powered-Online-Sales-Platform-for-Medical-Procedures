import os
import time
import ast
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from transitions import Machine
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer, util
from database import Doctor, Operation, DoctorOperation, Base, Patient
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
    embedding = Column(Vector(768))  # Adjust dimension based on your embedding model

    def __repr__(self):
        return f"<DocumentEmbedding(id={self.id}, chunk_id={self.chunk_id})>"


# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class LLMClient:
    def __init__(self, api_key=None, model="gemini-2.0-flash-lite"):
        self.api_key = api_key or os.getenv('GEMINI_KEY')
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def analyze_sentiment(self, text):
        prompt = (
            "Sen özel bir hasta-duygu sınıflayıcıysan. "
            "Mesajı oku ve yalnızca positive, neutral veya negative etiketlerinden biriyle yanıt ver. "
            "Pozitif: hasta hevesli, heyecanlı veya hazır olduğunu ifade ediyorsa; "
            "Nötr: sadece bilgi amaçlı sorular soruyorsa; "
            "Negatif: Risklerden korktuğunu belli ediyorsa, tereddüt veya çekingenlik gösteriyorsa. "
            f"Mesaj: \"{text}\""
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=10, temperature=0.0)
        )
        label = response.text.strip().lower()
        return label if label in ['positive','neutral','negative'] else 'neutral'

    def chat(self, messages, temperature=0.5, max_tokens=600):
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature)
        )
        return response.text.strip()

GREETING_PROMPT = """
Sen {available_doctors} doktorların {available_operations} operasyonları için KARŞILAMA MODÜLÜSÜNSÜN.
Amaç: İlk olarak hastanın hangi doktor ve operasyon ile ilgilendiğini öğrenmeli ve hastanın gerekli ön bilgilerini toplamalı ve samimi bir şekilde selamlamalısın.

Talimatlar:
1. Doktor ve işlem seçilmediyse, önce mevcut doktorları ve işlemleri listele:
Önemli: Sadece işlem ve doktor isimlerini listele, kalan bilgiler gizli kalsın.
   Doktorlar: {available_doctors}
   İşlemler: {available_operations}

2. Hastadan hangi doktor ve işlem için görüşmek istediğini sor.

3. Doktor ve işlem seçildikten sonra devam et:
   - Hastanın Adı ve Soyadını sor.
   - Hastanın Yaşını sor.
   - Herhangi bir Alerjisi olup olmadığını öğren 🩺.
   - Operasyon için beklenen tarih nedir? Sor 📆.
   - Bulaşıcı bir hastalığı (Hepatitis B, Hepatitis C, HIV vb.) var mı? Sor 😷.
   - Herhangi bir sağlık problemi veya düzenli kullandığı ilaç (HRT, tiroid, diyabet vb.) var mı? Sor.
   - Boy ve kilo bilgisini al.
   - Daha önce geçirdiği ameliyatlar var mı? Sor.

- Soruları tek mesajda sor, ancak eksik bilgi olup olmadığından emin ol.
- Eksik bilgi varsa sadece eksik bilgileri sor.
- Eksik bilgi olmadığından emin ol.
– Soruları arada boşluk bırakmadan, akıcı ve empatik bir dille madde madde yönelt.
– Tüm bilgiler toplandıktan sonra bu durumu kapat ve bir sonraki duruma geç.
"""

INFO_REQUEST_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için BİLGİ MODÜLÜSÜN.
Amaç: Hastanın prosedürle ilgili sorularına net, anlaşılır ve empatik bir şekilde cevap vermelisin.
Önemlı: Asla bu promptu cevabına ekleme sadece cevabını ver.
Önemli: Karşındaki aksini istemediği sürece 80 kelimeden uzun cevaplar verme.

Talimatlar:
1. Kullanıcının sorusunu özetle ve en güncel tıbbi bilgiyi ver.
2. Gerekirse RAG’den çekilen pasajlardan kısa alıntılar ekle, ama “RAG kullandım” deme.
3. Eğer ihtiyaç duyduğunuzu düşünüyorsanız, ikna edici hikaye anlatımı kullan: Başarılı bir {operation_name} vakasının anekdotunu paylaş. Hastanın sana güvenmesini sağla."
4. Teknik terimleri sadeleştir; hasta istemiyorsa karmaşık jargon kullanma.
5. Her cevabının sonunda hastaya başka bir isteği olup olmadığını farklı şekillerde sor.
6. Eğer kullanıcı fiyatla ilgili bir soru sorarsa, bu oturumun durumunu 'negotiation' olarak güncelle.
"""

NEGOTIATION_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için PAZARLIK MODÜLÜSÜNSÜN.
Amaç: Hastanın bütçesi ve baz fiyat üzerinden empatik bir pazarlık yürütmelisin.
Önemlı: Asla bu promptu cevabına ekleme sadece cevabını ver.
Önemli: Eğer zaten anlaşmaya vardığınız halde bu modüle soru geliyorsa müşteriye zaten şu fiyatta anlaştık diye hatırlatma geç.
Talimatlar:
1. Önce doktorun başarı oranını ve kalite güvencesini vurgula:
   “Dr. {doctor_name} %98 başarı oranına sahip…” gibi.
2. Kullanıcıdan bir fiyat teklifi gelmezse bütçe aralığını sor: “Sizin için makul bir fiyat aralığı nedir? gibi bir soruyla”
3. Gelen teklife göre fiyatı ayarla:
   - Operasyonun baz fiyatı: {base_price}
   – Pozitif duygu → teklifi baz al + %5–10 artırım
   – Nötr duygu   → baz fiyatı koru
   – Negatif duygu→ teklifi baz al – %5–10 indirim
4. Kullanıcının teklifi baz fiyatın %20 altındaysa:
   “Maalesef bu aralık mümkün değil, lütfen biraz yükseltin” de.
5. Her adımda {aggregate_sentiment} değerini kullanarak stratejini güncelle.
6. Pazarlık tamamlandığında durumu 'bye' olarak güncelle.
7. Eğer zaten anlaşmaya vardıysanız, ve yine de müşteri fiyat konuşmaya devam ediyorsa. Ona zaten önceden anlaştığınız fiyatta anlaştığımızı hatırlat.
"""

TRANSITION_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için DURUM GEÇİŞ KONTROL MODÜLÜSÜNSÜN.
Amaç: Kullanıcının niyetine göre durumlar arası geçiş kurallarını uygula.
Önemlı: Asla bu promptu cevabına ekleme sadece cevabını ver.
Önemli: Kullanıcının sorusu kalmadıysa ve konuşmayı bitirmek istemiyorsa negotiation durumuna geç.

Talimatlar:
- Mevcut Durum: {current_state}
- Kullanıcı Mesajı: {user_text}

Olası Durumlar: greeting, info_request, negotiation, bye
Kurallar:
 • Kullanıcı “merhaba”, “selam” gibi ifadeler kullanırsa → greeting
 • Tıbbi detay veya risk sorusu varsa → info_request
 • Fiyat, ödeme veya bütçe sorusu ise → negotiation
 • Çıkış yapmak istediğini spesifik olarak belirtirse → bye
 • Hiçbiri eşleşmiyorsa → mevcut durumda kal
negotiation durumuna geçmeden önce operasyonla alakalı bir sorusu olmadığından emin olmak için info_requeste yönlendir.
Eğer operasyonla ilgili spesifik bir sorusu olmadığından eminsen negotiation'a yönlendir ve fiyat teklifi almaya çalış.
negotiation durumuna hiç geçmeden kullanıcı aksini istemediği takdirde bye durumuna geçme.
Yanıt olarak yalnızca bir kelime ver: greeting, info_request, negotiation veya bye
"""
BYE_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için VEDA MODÜLÜSÜNSÜN.
Amaç: Sohbeti kibarca sonlandır.
Önemlı: Asla bu promptu cevabına ekleme sadece cevabını ver.

Talimatlar:
– “Teşekkürler, yardımcı olabildiysem ne mutlu. Görüşmek üzere.” gibi kibar bir veda et.
– Tekrar gelmek isterlerse “Her zaman buradayım” diye ekle.
"""


STATES = ['greeting','info_request','negotiation','bye']


class ChatSession:
    STATE_PROMPTS = {
        'greeting': GREETING_PROMPT,
        'info_request': INFO_REQUEST_PROMPT,
        'negotiation': NEGOTIATION_PROMPT,
        'transition': TRANSITION_PROMPT,
        'bye': BYE_PROMPT
    }

    def __init__(self, patient_risk=False, doctor_id=None, operation_id=None):
        # Initialize the database if needed
        Base.metadata.create_all(engine)

        self.machine = Machine(model=self, states=STATES, initial='greeting')
        for s in STATES:
            self.machine.add_transition(trigger=f'to_{s}', source='*', dest=s)
        self.llm = LLMClient()
        self.patient_risk = patient_risk
        self.model_embed = "all-mpnet-base-v2"
        self.client_embed = genai.Client(api_key=self.llm.api_key)
        self._generate_embeddings()

        # Load available doctors and operations from database
        self.available_doctors, self.available_operations = self._get_available_options()

        # Set doctor and operation if provided
        if doctor_id and operation_id:
            self._set_doctor_and_operation(doctor_id, operation_id)
        else:
            self.doctor = None
            self.operation = None
            self.base_price = None

        self.last_offer = None
        self.base_temperature = 0.5
        self.visit_counts = {s: 0 for s in STATES}
        self.chat_history = []
        self.memory_summary = None
        self.state = 'greeting'
        self.patient_info = {
            'name': None,
            'surname': None,
            'age': None,
            'allergies': None,
            'expected_date': None,
            'infectious_diseases': None,
            'infectious_diseases_details': None,
            'health_problems': None,
            'medications': None,
            'height': None,
            'weight': None,
            'previous_surgeries': None
        }
        self.agreed_price = None

        # Add first system message
        self._add_message('system', self._get_greeting_prompt())

    def save_patient_to_db(self):
        """Save patient information to database"""
        if not self.doctor or not self.operation:
            print("Cannot save patient: Doctor or operation not selected")
            return False

        # Convert any string values that should be numbers
        if isinstance(self.patient_info.get('age'), str) and self.patient_info.get('age').isdigit():
            self.patient_info['age'] = int(self.patient_info['age'])

        if isinstance(self.patient_info.get('height'), str) and self.patient_info.get('height').isdigit():
            self.patient_info['height'] = int(self.patient_info['height'])

        if isinstance(self.patient_info.get('weight'), str) and self.patient_info.get('weight').isdigit():
            self.patient_info['weight'] = int(self.patient_info['weight'])

        # Try to parse expected_date if it's a string
        expected_date = None
        if self.patient_info.get('expected_date'):
            try:
                from datetime import datetime
                date_text = self.patient_info.get('expected_date')
                # Try different date formats
                for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d.%m.%Y'):
                    try:
                        expected_date = datetime.strptime(date_text, fmt).date()
                        break
                    except ValueError:
                        continue
            except Exception as e:
                print(f"Error parsing date: {e}")

        db_session = Session()

        # Check if patient with same name/surname already exists
        existing_patient = None
        if self.patient_info.get('name') and self.patient_info.get('surname'):
            existing_patient = db_session.query(Patient).filter_by(
                name=self.patient_info.get('name'),
                surname=self.patient_info.get('surname'),
                doctor_id=self.doctor.id,
                operation_id=self.operation.id
            ).first()

        if existing_patient:
            # Update existing patient
            patient = existing_patient
            # Only update fields that have values
            for key, value in self.patient_info.items():
                if value is not None:
                    setattr(patient, key, value)

            if self.agreed_price:
                patient.agreed_price = self.agreed_price
                patient.negotiation_completed = True
        else:
            # Create new patient
            patient = Patient(
                name=self.patient_info.get('name'),
                surname=self.patient_info.get('surname'),
                age=self.patient_info.get('age'),
                allergies=self.patient_info.get('allergies'),
                expected_date=expected_date,
                infectious_diseases=bool(self.patient_info.get('infectious_diseases')),
                infectious_diseases_details=self.patient_info.get('infectious_diseases_details'),
                health_problems=self.patient_info.get('health_problems'),
                medications=self.patient_info.get('medications'),
                height=self.patient_info.get('height'),
                weight=self.patient_info.get('weight'),
                previous_surgeries=self.patient_info.get('previous_surgeries'),
                doctor_id=self.doctor.id,
                operation_id=self.operation.id,
                agreed_price=self.agreed_price,
                negotiation_completed=(self.agreed_price is not None)
            )
            db_session.add(patient)

        try:
            db_session.commit()
            print(f"Patient {'updated' if existing_patient else 'saved'} to database")
            result = True
        except Exception as e:
            db_session.rollback()
            print(f"Error saving patient to database: {e}")
            result = False
        finally:
            db_session.close()

        return result

    def _extract_patient_info(self, user_text):
        """Extract patient information from conversation and update patient_info"""
        # Create a summary prompt to extract structured patient info
        prompt = f"""
        Extract patient information from this conversation. 
        User: {user_text}

        Return ONLY a valid JSON object with these fields (use null for missing numeric values and empty string for missing text):
        {{
          "name": "",
          "surname": "",
          "age": null,
          "allergies": "",
          "expected_date": "",
          "infectious_diseases": "",
          "health_problems": "",
          "medications": "",
          "height": null,
          "weight": null,
          "previous_surgeries": ""
        }}
        """

        response = self.llm.chat([{'role': 'system', 'content': prompt}], temperature=0.0)

        try:
            # Extract JSON part from the response
            json_text = response.strip()
            # Remove code block markers if present
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]

            # Parse the JSON
            import json
            extracted_info = json.loads(json_text)

            # Update patient info with any new information
            for key, value in extracted_info.items():
                if value and not self.patient_info.get(key):
                    self.patient_info[key] = value

            print(f"Extracted patient info: {self.patient_info}")

        except Exception as e:
            print(f"Error extracting patient information: {e}")
            print(f"Raw response: {response}")

    def _detect_price_agreement(self, user_text, assistant_text):
        """Detect if a price agreement has been reached"""
        if self.state != 'negotiation' or self.agreed_price is not None:
            return

        prompt = f"""
        Analyze this conversation for a price agreement:
        User: {user_text}
        Assistant: {assistant_text}

        If the user has agreed to a specific price, extract ONLY the price number.
        If there's no clear agreement, return "NO_AGREEMENT"
        """

        response = self.llm.chat([{'role': 'system', 'content': prompt}], temperature=0.0)
        response_text = response.strip().lower()

        if "no_agreement" not in response_text:
            try:
                # Try to extract a number
                price_matches = re.findall(r'(\d+(?:\.\d+)?)', response_text)
                if price_matches:
                    self.agreed_price = float(price_matches[0])
                    print(f"Detected price agreement: {self.agreed_price}")
                    # Save to database when agreement is reached
                    self.save_patient_to_db()
                    return True

            except Exception as e:
                print(f"Error detecting price agreement: {e}")

        return False

    def _get_available_options(self):
        """Fetch all available doctors and operations from database"""
        db_session = Session()

        doctors = db_session.query(Doctor).all()
        operations = db_session.query(Operation).all()

        doctor_list = [f"{d.name} {d.surname} ({d.specialty}, ID: {d.id})" for d in doctors]
        operation_list = [f"{op.name} (Base Price: {op.base_price}₺, ID: {op.id})" for op in operations]

        db_session.close()
        return doctor_list, operation_list

    def _set_doctor_and_operation(self, doctor_id, operation_id):
        """Set doctor and operation based on IDs and calculate price"""
        db_session = Session()

        # Get doctor-operation relationship with price information
        doctor_op = db_session.query(DoctorOperation).join(Doctor).join(Operation) \
            .filter(DoctorOperation.doctor_id == doctor_id,
                    DoctorOperation.operation_id == operation_id).first()

        if not doctor_op:
            db_session.close()
            raise ValueError("Doctor-operation combination not found in database")

        self.doctor = doctor_op.doctor
        self.operation = doctor_op.operation

        # Set pricing based on database and risk factor
        modifier = 1.5 if self.patient_risk else 1.0
        self.base_price = doctor_op.price * modifier
        self.min_price = (doctor_op.min_price or doctor_op.price * 0.9) * modifier
        self.max_price = (doctor_op.max_price or doctor_op.price * 1.2) * modifier

        db_session.close()

    def _get_greeting_prompt(self):
        """Get customized greeting prompt based on available options"""
        doctor_name = f"{self.doctor.name} {self.doctor.surname}" if self.doctor else "Seçilecek doktor"
        operation_name = self.operation.name if self.operation else "Seçilecek operasyon"

        return self.STATE_PROMPTS['greeting'].format(
            doctor_name=doctor_name,
            operation_name=operation_name,
            available_doctors=", ".join(self.available_doctors),
            available_operations=", ".join(self.available_operations)
        )

    def _parse_selection(self, user_text):
        """Parse doctor and operation selection from user input"""
        # Simple regex pattern to find doctor ID and operation ID
        doctor_pattern = r"doktor[^\d]*(\d+)"
        operation_pattern = r"(?:i[şs]lem|operasyon)[^\d]*(\d+)"

        doctor_match = re.search(doctor_pattern, user_text.lower())
        operation_match = re.search(operation_pattern, user_text.lower())

        doctor_id = int(doctor_match.group(1)) if doctor_match else None
        operation_id = int(operation_match.group(1)) if operation_match else None

        # If direct ID not found, try to match by name
        if not doctor_id or not operation_id:
            db_session = Session()
            doctors = db_session.query(Doctor).all()
            operations = db_session.query(Operation).all()

            # Check for doctor name matches
            for doctor in doctors:
                if doctor.name.lower() in user_text.lower() or doctor.surname.lower() in user_text.lower():
                    doctor_id = doctor.id
                    break

            # Check for operation name matches
            for operation in operations:
                if operation.name.lower() in user_text.lower():
                    operation_id = operation.id
                    break

            db_session.close()

        return doctor_id, operation_id

    def process_user(self, user_text):
        self._add_message('user', user_text)

        # If doctor and operation not selected yet, try to parse from input
        if not self.doctor or not self.operation:
            doctor_id, operation_id = self._parse_selection(user_text)
            if doctor_id and operation_id:
                try:
                    self._set_doctor_and_operation(doctor_id, operation_id)
                    # Update greeting prompt with selected doctor and operation
                    self._add_message('system', self._get_greeting_prompt())
                except ValueError as e:
                    return f"Üzgünüm, seçtiğiniz doktor ve operasyon kombinasyonu bulunamadı. Lütfen tekrar deneyin."

        # Extract patient information if in greeting state and it's the 3rd or later user message
        user_message_count = sum(1 for m in self.chat_history if m['role'] == 'user')
        if (self.state == 'info_request' or self.state == 'greeting') and user_message_count == 3:
            self._extract_patient_info(user_text)

        # Continue with regular processing
        self.visit_counts[self.state] += 1
        temp = min(0.9, self.base_temperature + 0.1 * (self.visit_counts[self.state] - 1))

        if self.state == 'info_request':
            ref = self.find_best_passage(user_text)
        else:
            ref = user_text

        prompt, agg_sentiment = self._build_prompt(user_text)

        if self.state == 'negotiation':
            agg = f"aggregate_sentiment: positive={agg_sentiment['positive']}, neutral={agg_sentiment['neutral']}, negative={agg_sentiment['negative']}"
            system_content = f"{prompt}\n{agg}\nREFERENCE: {ref}"
        else:
            system_content = f"{prompt}\nREFERENCE: {ref}"

        msgs = [{'role': 'system', 'content': system_content}]
        if self.memory_summary:
            msgs.append({'role': 'system', 'content': f"memory_summary: {self.memory_summary}"})
        for m in self.chat_history:
            msgs.append({'role': m['role'], 'content': m['content']})

        reply = self.llm.chat(msgs, temperature=temp)
        self._add_message('assistant', reply)

        # Check for price agreement if in negotiation state
        if self.state == 'negotiation':
            self._detect_price_agreement(user_text, reply)

        # When transitioning to bye state, save patient info if not already saved
        if self.state == 'bye' and (self.doctor and self.operation):
            self.save_patient_to_db()

        return reply

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
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append("\t".join(cells))
                full_text.append("\n".join(rows))
        return "\n".join(full_text)

    def _split_text(self, text, chunk_size=1000, overlap=200):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)

    def _generate_embedding(self, text):
        response = self.client_embed.models.embed_content(model=self.model_embed, contents=text)
        time.sleep(10)
        return response.embeddings[0].values

    def _generate_embeddings(self):
        """Check if embeddings exist in the database"""
        db_session = Session()

        # Check if we already have embeddings in the database
        count = db_session.query(DocumentEmbedding).count()
        if count == 0:
            db_session.close()
            raise Exception(
                "No embeddings found in database. Please run db_embeddings.py first to generate embeddings."
            )
        else:
            print(f"Found {count} existing embeddings in database.")

        db_session.close()

    def find_best_passage(self, query):
        """Find the most relevant passage using vector similarity with local embeddings"""
        # Load SentenceTransformer model if not already loaded
        if not hasattr(self, 'local_model'):
            print("Loading local embedding model...")
            self.local_model = SentenceTransformer('all-mpnet-base-v2')  # Must match db_embeddings.py

        # Encode the query using local model
        query_embedding = self.local_model.encode(query, convert_to_numpy=True)

        db_session = Session()

        # Convert numpy array to list for pgvector
        query_vec_str = str(query_embedding.tolist())

        # Use SQLAlchemy's text() function for the query
        sql = text("""
            SELECT id, chunk_id, text, embedding <=> :query_vector AS distance
            FROM document_embeddings
            ORDER BY distance ASC
            LIMIT 3
        """).bindparams(query_vector=query_vec_str)

        result = db_session.execute(sql).fetchall()
        db_session.close()

        # Combine the text from the top 3 results
        passages = [row[2] for row in result]  # row[2] is the text column
        return " ".join(passages)

    def _add_message(self, role, content):
        sentiment = self.llm.analyze_sentiment(content)
        self.chat_history.append({'role': role, 'content': content, 'sentiment': sentiment, 'state': self.state})

    def _last_user_message(self):
        for msg in reversed(self.chat_history):
            if msg['role'] == 'user':
                return msg['content']
        return ''

    def _parse_next_state(self, response_text):
        candidate = response_text.strip().lower()
        return candidate if candidate in STATES else self.state

    def _build_prompt(self, user_text):
        # If doctor and operation not selected yet
        if not self.doctor or not self.operation:
            return self.STATE_PROMPTS['greeting'].format(
                doctor_name="Seçilecek doktor",
                operation_name="Seçilecek operasyon",
                available_doctors=", ".join(self.available_doctors),
                available_operations=", ".join(self.available_operations)
            ), {'positive': 0, 'neutral': 1, 'negative': 0}

        # Proceed with transition handling for selected doctor/operation
        transition = self.STATE_PROMPTS['transition'].format(
            doctor_name=f"{self.doctor.name} {self.doctor.surname}",
            operation_name=self.operation.name,
            current_state=self.state,
            user_text=user_text
        )
        trans_resp = self.llm.chat([
            {'role': 'system', 'content': transition},
            {'role': 'user', 'content': user_text}
        ], temperature=0.0, max_tokens=5)
        next_state = self._parse_next_state(trans_resp)
        if next_state != self.state:
            getattr(self, f'to_{next_state}')()
            self.state = next_state

        # Calculate aggregate sentiment
        sents = [m['sentiment'] for m in self.chat_history if m['role'] == 'user']
        agg_sentiment = {
            'positive': sents.count('positive'),
            'neutral': sents.count('neutral'),
            'negative': sents.count('negative')
        }

        # Format prompt based on state
        if self.state == 'negotiation':
            prompt = self.STATE_PROMPTS[self.state].format(
                doctor_name=f"{self.doctor.name} {self.doctor.surname}",
                operation_name=self.operation.name,
                base_price=self.base_price,
                aggregate_sentiment=f"positive={agg_sentiment['positive']}, neutral={agg_sentiment['neutral']}, negative={agg_sentiment['negative']}"
            )
        else:
            prompt = self.STATE_PROMPTS[self.state].format(
                doctor_name=f"{self.doctor.name} {self.doctor.surname}",
                operation_name=self.operation.name
            )

        return prompt, agg_sentiment


    def debug_embeddings(self):
        """Debug the first embedding in the database"""
        db_session = Session()
        first_doc = db_session.query(DocumentEmbedding).first()
        db_session.close()

        if first_doc:
            print(f"Embedding type: {type(first_doc.embedding)}")
            print(f"Embedding sample: {str(first_doc.embedding)[:100]}...")
        else:
            print("No embeddings found in database")

if __name__ == "__main__":
    load_dotenv()
    session = ChatSession(
        patient_risk=False
    )
    while session.state != 'bye':
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        print(session.process_user(user_input))