from langchain.embeddings.openai import OpenAIEmbeddings
from util.constants import *


def embed_to_pinecone(embedding_model, dataset, batch_size, index):
    for i in range(0, len(dataset), batch_size):
        i_end = min(len(dataset), i + batch_size)
        batch = dataset.iloc[i:i_end]
        ids = [str(i) for i in range(len(batch))]
        metadata = [{'source': "O.Y.", 'title': "CA as Synthetic data"} for _ in batch.iterrows()]
        embeddings = embedding_model.embed_documents(batch)
        index.upsert(vectors=zip(ids, embeddings, metadata))
        index.describe_index_stats()

