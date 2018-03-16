import glob
import json

import os

from common.constants import JSON_KEY_EMBEDDINGS, JSON_KEY_LABEL
from data.embedding_dao import EmbeddingDAO

class JsonEmbeddingDao(EmbeddingDAO):

    def __init__(self, json_dir):
        self.embeddings = []
        self.labels = []
        json_basenames = glob.glob1(json_dir, "*embeddings*.json")
        for json_basename in json_basenames:
            json_path = os.path.join(json_dir, json_basename)
            data = json.load(open(json_path))
            for data_point in data:
                self.embeddings.append(data_point[JSON_KEY_EMBEDDINGS])
                self.labels.append(data_point[JSON_KEY_LABEL])

    def getAllEmbeddings(self):
        """ Returns a 2-tuple of embeddings and corresponding labels"""
        return self.embeddings, self.labels