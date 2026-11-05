from abc import ABC, abstractmethod


class LiteratureSource(ABC):
    source = None

    @abstractmethod
    def search(self, query, limit=20, embedding_model="nomic-local"):
        raise NotImplementedError

    @abstractmethod
    def get_article(self, identifier, embedding_model="nomic-local"):
        raise NotImplementedError
