import torch
import uuid
import qdrant_client

import pandas as pd
import numpy as np

from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from transformers import AutoTokenizer, AutoModel, pipeline

# read data
df = pd.read_csv('data/mle_screening_dataset.csv')

# drop missing values rows
df = df[~df['answer'].isna()].copy()

# vector database class
class VectorDatabase:
    def __init__(self, collection_name="answers_collection", embed_model="sentence-transformers/all-distilroberta-v1"):
        """Vector database and embedding model initialization."""
        self.collection_name = collection_name
        self.client = qdrant_client.QdrantClient(":memory:")
    
        # embedding model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.model = AutoModel.from_pretrained(embed_model)

        # create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # we are using cosine similarity
        )

    def get_embedding(self, text):
        """Create embeddings for provided text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def split_text(self, text, chunk_size=500, overlap=100):
        """Divides text into smaller chunks with overlapping."""
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    def insert_answers(self, answers):
        """Insert answers into the Qdrant vector database."""
        points = []
        for answer in answers:
            chunks = self.split_text(answer)
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),  # unique id
                        vector=embedding.tolist(),  # embedding of the chunk
                        payload={"text": chunk, "full_text": answer}  # store original text (both chunk and full)
                    )
                )
        self.client.upsert(self.collection_name, points)

    def search_answer(self, query, top_k=3):
        """Search for the most relevant answers in the collection."""
        query_embedding = self.get_embedding(query)
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        return search_results

# question answering model class
class QuestionAnsweringModel:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        """Initialize the Hugging Face QA model."""
        self.qa_pipeline = pipeline("question-answering", model=model_name)

    def get_best_result(self, query, search_results, min_score=0.3):
        """
        Select the best among search results using hugging face question-answering model.
        """
        # here we create dummy best result
        # it has default score equal to 0.3 since below that value usually answers are unreliable 
        # in that case we return info, that answer could not be found
        # we can change it so that it always returns "something"
        best_result = {
            'score': min_score,
            'start': 0,
            'end': 33,
            'answer': "Sorry, I couldn't find an answer.",
            'chunk': "Sorry, I couldn't find an answer.",
            'full_text': "Sorry, I couldn't find an answer."
        }

        # for each search result we look for good answer
        for hit in search_results:
            chunk = hit.payload['text']
            full_text = hit.payload['full_text']
            
            result = self.qa_pipeline(question=query, context=chunk)
            
            result['chunk'] = chunk
            result['full_text'] = full_text

            # if the answer is the best to this moment we save it with the chunk and full text
            if result['score'] > best_result['score']:
                best_result = result.copy()

        return best_result

# model and collection name
EMB_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
COLLECTION_NAME = "answers_collection"

# define object of database
vector_database = VectorDatabase(collection_name=COLLECTION_NAME, embed_model=EMB_MODEL_NAME)

# QA model name
QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# define object of question answering model
question_answering_model = QuestionAnsweringModel(QA_MODEL_NAME)

# put whole dataset - uncomment for "prod"
# vector_database.insert_answers(df['answer'].unique().tolist())

# insert part of dataset - uncomment for tests
print('Inserting documents into database.')
vector_database.insert_answers(df['answer'].unique().tolist()[:100])

print('\nQUESTION:')
# search_query = "What are clinical trials?"
# search_query = "Where are aneurysms usually located?"
search_query = "Can loss of taste that occurs with aging be prevented?"
print(search_query)

print('Searching for the best documents.')
search_results = vector_database.search_answer(search_query)

print('Answering the question.')
result = question_answering_model.get_best_result(search_query, search_results)

print('\nBEST ANSWER')
print('Score: ', result['score'])
print('Answer: ', result['answer'])
print('Full text: ', result['full_text'])

