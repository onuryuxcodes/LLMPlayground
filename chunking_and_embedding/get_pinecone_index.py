from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from apis.keys import *
from util.constants import *


def get_pinecone_index():
    api_key = PINECONE_API_KEY
    pc = Pinecone(api_key=api_key)

    spec = ServerlessSpec(
        cloud=GCP, region=REGION_US_CENTRAL
    )
    index_name = PINECONE_INDEX_NAME_PORTFOLIO_PROJECT
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]
    if index_name not in existing_indexes:
        pc.create_index(
            index_name,
            dimension=1536,
            metric=METRIC_DOT_PRODUCT,
            spec=spec
        )
        while not pc.describe_index(index_name).status[STATUS_READY]:
            time.sleep(1)
    index = pc.Index(index_name)
    time.sleep(1)
    print(index.describe_index_stats())
    return index

