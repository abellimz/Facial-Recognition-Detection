import abc

class EmbeddingDAO(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getAllEmbeddings(self):
        """ Returns a 2-tuple of embeddings and corresponding labels"""
        pass
