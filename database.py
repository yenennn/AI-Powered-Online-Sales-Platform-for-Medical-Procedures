from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

Base = declarative_base()

from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Date, Boolean, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker


# Add this to your database.py file
class Patient(Base):
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    age = Column(Integer)
    allergies = Column(Text)
    expected_date = Column(Date)
    infectious_diseases = Column(Boolean, default=False)
    infectious_diseases_details = Column(Text)
    health_problems = Column(Text)
    medications = Column(Text)
    height = Column(Integer)  # in cm
    weight = Column(Integer)  # in kg
    previous_surgeries = Column(Text)

    # Foreign keys
    doctor_id = Column(Integer, ForeignKey('doctors.id'))
    operation_id = Column(Integer, ForeignKey('operations.id'))

    # Price information
    agreed_price = Column(Float)
    negotiation_completed = Column(Boolean, default=False)

    # Relationships
    doctor = relationship("Doctor", back_populates="patients")
    operation = relationship("Operation", back_populates="patients")

    def __repr__(self):
        return f"<Patient(id={self.id}, name={self.name} {self.surname})>"


class Doctor(Base):
    __tablename__ = 'doctors'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    specialty = Column(String(100))
    description = Column(Text)

    # Relationship
    operations = relationship("DoctorOperation", back_populates="doctor")
    patients = relationship("Patient", back_populates="doctor")

    def __repr__(self):
        return f"<Doctor(id={self.id}, name={self.name} {self.surname}, specialty={self.specialty})>"


class Operation(Base):
    __tablename__ = 'operations'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    base_price = Column(Float, nullable=False)

    # Relationship
    doctors = relationship("DoctorOperation", back_populates="operation")
    patients = relationship("Patient", back_populates="operation")
    def __repr__(self):
        return f"<Operation(id={self.id}, name={self.name}, base_price={self.base_price})>"


class DoctorOperation(Base):
    __tablename__ = 'doctor_operations'

    id = Column(Integer, primary_key=True)
    doctor_id = Column(Integer, ForeignKey('doctors.id'), nullable=False)
    operation_id = Column(Integer, ForeignKey('operations.id'), nullable=False)
    price = Column(Float, nullable=False)  # Doctor's specific price for this operation
    min_price = Column(Float)  # Minimum acceptable price (for negotiation)
    max_price = Column(Float)  # Maximum price (for premium service)

    # Relationships
    doctor = relationship("Doctor", back_populates="operations")
    operation = relationship("Operation", back_populates="doctors")

    def __repr__(self):
        return f"<DoctorOperation(doctor_id={self.doctor_id}, operation_id={self.operation_id}, price={self.price})>"


def create_tables():
    """Create the database tables"""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print("Database tables created successfully")


def populate_sample_data():
    """Add sample data to the database"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Add doctors
    doctor1 = Doctor(name="Selcuk", surname="Coskun", specialty="Plastic Surgeon",
                     description="Specialist in rhinoplasty with over 15 years of experience")
    doctor2 = Doctor(name="Ayse", surname="Demir", specialty="Aesthetic Surgeon",
                     description="Expert in facial reconstruction and aesthetics")

    # Add operations
    operation1 = Operation(name="Rhinoplasty", base_price=5000,
                           description="Surgical procedure that changes the shape of the nose")
    operation2 = Operation(name="Facelift", base_price=7500,
                           description="Procedure to reduce visible signs of aging in the face and neck")

    session.add_all([doctor1, doctor2, operation1, operation2])
    session.commit()

    # Link doctors with operations and set prices
    doc_op1 = DoctorOperation(doctor_id=doctor1.id, operation_id=operation1.id,
                              price=5500, min_price=5000, max_price=6500)
    doc_op2 = DoctorOperation(doctor_id=doctor1.id, operation_id=operation2.id,
                              price=8000, min_price=7500, max_price=9000)
    doc_op3 = DoctorOperation(doctor_id=doctor2.id, operation_id=operation1.id,
                              price=5200, min_price=4800, max_price=6000)

    session.add_all([doc_op1, doc_op2, doc_op3])
    session.commit()

    print("Sample data added to database")
    session.close()


if __name__ == "__main__":
    DATABASE_URL = "postgresql://postgres:yenenn@localhost:5432/vector_db"
    create_tables()
    populate_sample_data()