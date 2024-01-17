from langchain.text_splitter import CharacterTextSplitter
from util.constants import SEPARATOR


def chunk(text_to_be_chunked, chunk_size=1500, chunk_overlap=20):
    text_splitter = CharacterTextSplitter(
        separator=SEPARATOR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.create_documents([text_to_be_chunked])


def shallow_chunk(text_to_be_chunked):
    return text_to_be_chunked.split(SEPARATOR)
