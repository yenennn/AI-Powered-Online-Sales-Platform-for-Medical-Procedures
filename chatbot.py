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

GREETING_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için KARŞILAMA MODÜLÜSÜNSÜN.
Amaç: İlk olarak hastanın gerekli ön bilgilerini toplamalı ve samimi bir şekilde selamlamalısın.

Talimatlar:
1. Hastanın Adı ve Soyadını sor.
2. Hastanın Yaşını sor.
3. Herhangi bir Alerjisi olup olmadığını öğren 🩺.
4. Operasyon için beklenen tarih nedir? Sor 📆.
5. Bulaşıcı bir hastalığı (Hepatitis B, Hepatitis C, HIV vb.) var mı? Sor 😷.
6. Herhangi bir sağlık problemi veya düzenli kullandığı ilaç (HRT, tiroid, diyabet vb.) var mı? Sor.
7. Boy ve kilo bilgisini al.
8. Daha önce geçirdiği ameliyatlar var mı? Sor.

– Her bir cevabı açıkça onayla, eksik bilgi kalırsa tekrar sor.
– Soruları arada boşluk bırakmadan, akıcı ve empatik bir dille yönelt.
– Tüm bilgiler toplandıktan sonra bu durumu kapat ve bir sonraki duruma geç.
"""

INFO_REQUEST_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için BİLGİ MODÜLÜSÜNSÜN.
Amaç: Hastanın prosedürle ilgili sorularına net, anlaşılır ve empatik bir şekilde cevap vermelisin.

Talimatlar:
1. Kullanıcının sorusunu özetle ve en güncel tıbbi bilgiyi ver.
2. Gerekirse RAG’den çekilen pasajlardan kısa alıntılar ekle, ama “RAG kullandım” deme.
3. Eğer ihtiyaç duyduğunuzu düşünüyorsanız, ikna edici hikaye anlatımı kullan: Başarılı bir {operation.name} vakasının anekdotunu paylaş. Hastanın sana güvenmesini sağla."
4. Teknik terimleri sadeleştir; hasta istemiyorsa karmaşık jargon kullanma.
5. Her cevabının sonunda hastaya başka bir isteği olup olmadığını farklı şekillerde sor.
6. Eğer kullanıcı fiyatla ilgili bir soru sorarsa, bu oturumun durumunu 'negotiation' olarak güncelle.
"""

NEGOTIATION_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için PAZARLIK MODÜLÜSÜNSÜN.
Amaç: Hastanın bütçesi ve baz fiyat üzerinden empatik bir pazarlık yürütmelisin.

Talimatlar:
1. Önce doktorun başarı oranını ve kalite güvencesini vurgula:
   “Dr. {doctor_name} %98 başarı oranına sahip…” gibi.
2. Kullanıcıdan bir fiyat teklifi gelmezse bütçe aralığını sor: “Sizin için makul bir fiyat aralığı nedir? gibi bir soruyla”
3. Gelen teklife göre fiyatı ayarla:
   – Pozitif duygu → teklifi baz al + %5–10 artırım
   – Nötr duygu   → baz fiyatı koru
   – Negatif duygu→ teklifi baz al – %5–10 indirim
4. Kullanıcının teklifi baz fiyatın %20 altındaysa:
   “Maalesef bu aralık mümkün değil, lütfen biraz yükseltin” de.
5. Her adımda aggregate_sentiment değerini kullanarak stratejini güncelle.
6. Pazarlık tamamlandığında durumu 'bye' olarak güncelle.
"""

TRANSITION_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için DURUM GEÇİŞ KONTROL MODÜLÜSÜNSÜN.
Amaç: Kullanıcının niyetine göre durumlar arası geçiş kurallarını uygula.

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
Eğer operasyonla ilgili spesifik bir sorusu kalmadığından eminsen negotiation'a yönlendir ve fiyat teklifi almaya çalış
Yanıt olarak yalnızca bir kelime ver: greeting, info_request, negotiation veya bye
"""
BYE_PROMPT = """
Sen Dr.{doctor_name}'in {operation_name} operasyonu için VEDA MODÜLÜSÜNSÜN.
Amaç: Sohbeti kibarca sonlandır.

Talimatlar:
– “Teşekkürler, yardımcı olabildiysem ne mutlu. Görüşmek üzere.” gibi kibar bir veda et.
– Tekrar gelmek isterlerse “Her zaman buradayım” diye ekle.
"""


load_dotenv()

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

class Operation:
    def __init__(self, name, base_price):
        self.name = name
        self.base_price = base_price

class Doctor:
    def __init__(self, name, surname, specialty):
        self.name = name
        self.surname = surname
        self.specialty = specialty

STATES = ['greeting','info_request','negotiation','bye']

class ChatSession:
    STATE_PROMPTS = {
        'greeting': GREETING_PROMPT,
        'info_request': INFO_REQUEST_PROMPT,
        'negotiation': NEGOTIATION_PROMPT,
        'transition': TRANSITION_PROMPT,
        'bye': BYE_PROMPT
    }

    def __init__(self, operation, doctor, document_path, patient_risk=False):
        self.machine = Machine(model=self, states=STATES, initial='greeting')
        for s in STATES:
            self.machine.add_transition(trigger=f'to_{s}', source='*', dest=s)
        self.llm = LLMClient()
        self.operation = operation
        self.doctor = doctor
        self.document_path = document_path
        self.patient_risk = patient_risk
        self.model_embed = "gemini-embedding-exp-03-07"
        self.client_embed = genai.Client(api_key=self.llm.api_key)
        self.embeddings_df = self._generate_embeddings()
        modifier = 1.5 if patient_risk else 1.0
        self.base_price = operation.base_price * modifier
        self.last_offer = None
        self.base_temperature = 0.5
        self.visit_counts = {s: 0 for s in STATES}
        self.chat_history = []
        self.memory_summary = None
        self.state = 'greeting'
        self._add_message('system', self.STATE_PROMPTS['greeting'].format(
            doctor_name=self.doctor.name,
            operation_name=self.operation.name
        ))

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
        embeddings_file = "embeddings.csv"
        if os.path.exists(embeddings_file):
            df = pd.read_csv(embeddings_file)
            df['Embeddings'] = df['Embeddings'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
            return df
        text = self._extract_text_and_tables_in_order()
        chunks = self._split_text(text)
        df = pd.DataFrame({"chunk_id": list(range(1, len(chunks)+1)), "text": chunks})
        df['Embeddings'] = df['text'].apply(self._generate_embedding)
        df.to_csv(embeddings_file, index=False, encoding='utf-8')
        return df

    def find_best_passage(self, query):
        query_emb = self.client_embed.models.embed_content(model=self.model_embed, contents=query)
        query_vec = query_emb.embeddings[0].values
        corpus = np.stack(self.embeddings_df['Embeddings'].values)
        sims = corpus.dot(query_vec)
        best_idx = int(np.argmax(sims))
        idxs = [i for i in (best_idx-1, best_idx, best_idx+1) if 0 <= i < len(sims)]
        return " ".join(self.embeddings_df.iloc[idxs]['text'].values)

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
        transition = self.STATE_PROMPTS['transition'].format(
            doctor_name=self.doctor.name,
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
        prompt = self.STATE_PROMPTS[self.state].format(
            doctor_name=self.doctor.name,
            operation_name=self.operation.name
        )
        return prompt

    def process_user(self, user_text):
        self._add_message('user', user_text)
        self.visit_counts[self.state] += 1
        temp = min(0.9, self.base_temperature + 0.1 * (self.visit_counts[self.state] - 1))
        ref = self.find_best_passage(user_text)
        prompt = self._build_prompt(user_text)
        if self.state == 'negotiation':
            sents = [m['sentiment'] for m in self.chat_history if m['role'] == 'user']
            agg = f"aggregate_sentiment: positive={sents.count('positive')}, neutral={sents.count('neutral')}, negative={sents.count('negative')}"
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
        return reply


if __name__ == "__main__":
    load_dotenv()
    operation = Operation("Rhinoplasty", 5000)
    doctor = Doctor("Selcuk", "Coskun", "Plastic Surgeon")
    session = ChatSession(
        operation,
        doctor,
        "Definition and Purpose of Rhinoplasty.docx",
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
