from sqlalchemy import Integer, Column, String, create_engine, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('postgresql://user:password@localhost:5432/mydb')
Base = declarative_base()
session = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Article(Base):
    __tablename__ = 'article'

    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    content = Column(String(20000))
    published_date = Column(Date)
    pages = Column(Integer)
