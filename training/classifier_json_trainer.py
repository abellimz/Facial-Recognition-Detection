from training.classifier_trainer import ClassifierTrainer
from classifier.mlp_keras_classifier import MLPKerasClassifier
from data.json_embedding_dao import JsonEmbeddingDao

class ClassifierJsonTrainer(ClassifierTrainer):
    def __init__(self, json_data_dir, model_dir, model_basename):
        self.model_dir = model_dir
        self.model_basename = model_basename
        self.embedding_dao = JsonEmbeddingDao(json_data_dir)
        self.clf = MLPKerasClassifier()

    def train(self):
        """
        Trains classifier on given data set
        """
        embeddings, labels = self.embedding_dao.getAllEmbeddings()
        self.clf.new_model(list(set(labels)), (len(embeddings[0]),))
        self.clf.train(embeddings, labels)

    def save_classifier(self):
        """
        Saves classifier
        """
        self.clf.save_model(self.model_dir, self.model_basename)