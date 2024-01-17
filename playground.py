import pandas as pd
import os
from chunking_and_embedding.chunking_util import chunk, shallow_chunk
from chunking_and_embedding.get_pinecone_index import get_pinecone_index
from chunking_and_embedding.embedding_util import embed_to_pinecone
from langchain.vectorstores import Pinecone
from apis.api_key_setup import set_open_ai_api_key_to_environment
from apis.keys import GPT_MODEL_ID, OPEN_AI_API_KEY_DICT_ID
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from query_util.query_augmentation import get_augmented_prompt
from util.constants import *
from langchain.schema import (
    HumanMessage
)
from langchain.chat_models import ChatOpenAI
import sys
import logging


def rag_pipeline_for_thesis_data(query, database_choice=CHROMADB, should_embed_to_pinecone=False):
    set_open_ai_api_key_to_environment()
    # read thesis text, raw, currently not perfect
    thesis_raw = open(THESIS_DATA_PATH).read()
    chunks_list = shallow_chunk(thesis_raw)
    chunks_doc = chunk(thesis_raw)
    df = pd.DataFrame(chunks_list, columns=[COLUMN_THESIS_RAW])
    # embedding
    embedding_model = OpenAIEmbeddings(model=ADA_002_EMBEDDING)
    index = None
    if should_embed_to_pinecone:
        # get pinecone index on GCP server
        index = get_pinecone_index()
        # embed to pinecone
        embedding_model = embed_to_pinecone(embedding_model=embedding_model, dataset=df, batch_size=4, index=index)
    chosen_relevant_information = None
    if database_choice == PINECONE:
        vectorstore = Pinecone(
            index, embedding_model, COLUMN_THESIS_RAW
        )
        chosen_relevant_information = vectorstore.similarity_search(query=query)
    elif database_choice == CHROMADB:
        db = Chroma.from_documents(chunks_doc, embedding_model)
        chosen_relevant_information = db.similarity_search(query)
    augmented_prompt = get_augmented_prompt(source_knowledge=chosen_relevant_information,
                                            query=query)
    final_prompt = HumanMessage(
        content=augmented_prompt
    )
    messages = [final_prompt]
    chat = ChatOpenAI(
        openai_api_key=os.environ[OPEN_AI_API_KEY_DICT_ID],
        model=GPT_MODEL_ID
    )
    response = chat(messages)
    return response


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    query_input = sys.argv[1:]
    if query_input[0] == QUERY_ARG:
        print(PROCESSING_QUERY)
        resp = rag_pipeline_for_thesis_data(query_input[1])
        print(resp.content)
    else:
        print(UNEXPECTED_ARG_FORMAT_MSG)

