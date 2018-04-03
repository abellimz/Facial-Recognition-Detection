import unittest

from classifier.mlp_keras_classifier import MLPKerasClassifier

class MLPKerasTest(unittest.TestCase):
    def test_train_infer(self):
        nn_clf = MLPKerasClassifier()
        nn_clf.new_model(["tokyo", "paris", "amsterdam"], (4,))
        nn_clf.train([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [0, 0, 0, 0],
                      [0.1, 0.01, 0.02, 0.1], [-2.0, -3.0, -4.0, -5.0], [1.1, 1.2, 1.0, 1.1]],
                     ["tokyo", "paris", "amsterdam", "amsterdam", "paris", "tokyo"])
        test_features = [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [0, 0, 0, 0]]
        all_probs = nn_clf.infer(test_features)
        self.assertEqual(all_probs[0].index(max(all_probs[0])), 0)
        self.assertEqual(all_probs[1].index(max(all_probs[1])), 1)
        self.assertEqual(all_probs[2].index(max(all_probs[2])), 2)

    def test_save_load(self):
        # Trains model
        nn_clf = MLPKerasClassifier()
        nn_clf.new_model(["tokyo", "paris", "amsterdam"], (4,))
        nn_clf.train([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [0, 0, 0, 0],
                      [0.1, 0.01, 0.02, 0.1], [-2.0, -3.0, -4.0, -5.0], [1.1, 1.2, 1.0, 1.1]],
                     ["tokyo", "paris", "amsterdam", "amsterdam", "paris", "tokyo"])

        # Get prediction scores after training
        test_features = [[-1.0, 0.0, 0.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [0, 0, 0, 0]]
        expected_probs = nn_clf.infer(test_features)

        # Saves existing model and load that model
        nn_clf.save_model("/tmp/test_mlp", "test_mlp")
        nn_clf = MLPKerasClassifier()
        nn_clf.load_model("/tmp/test_mlp", "test_mlp")

        # Run test on loaded model
        new_probs = nn_clf.infer(test_features)
        self.assertEqual(expected_probs, new_probs)

if __name__ == '__main__':
    unittest.main()
