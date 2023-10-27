from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100)
        self.params = {
            'model__max_depth': [3, 5],
            'model__min_samples_split': [2, 5],
            'model__max_features': ['sqrt', 6] # 6=(features.shape[1] - len(meta))**(2/3)
        }
