from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CustomerData(Base):
    __tablename__ = "customer_data"

    id = Column(Integer, primary_key=True, index=True)
    review = Column(String, index=True)
    label = Column(Integer)
