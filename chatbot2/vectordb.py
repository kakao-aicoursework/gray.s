import os

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIR = 'chroma-persist'
CHROMA_COLLECTION_NAME = 'dosu-bot'


def upload_embedding_from_file(file_path):
    documents = TextLoader(file_path).load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def upload_embedding_from_dir(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                upload_embedding_from_file(file_path)
                print("SUCCESS: ", file_path)
            except Exception as e:
                print("FAILED: ", file_path + f"by({e})")


if __name__ == '__main__':
    upload_embedding_from_dir('raw')