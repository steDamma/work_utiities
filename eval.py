import json
import os
import argparse
import pickle
import random
import joblib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from classifiers.classifier import Classifier
from analysis.data_op import Data

class Eval:
    def __init__(self,
                 raw_data_name,
                 n_trials,
                 classifier_name,
                 period,
                 n_test,
                 n_validate,
                 leave_one_out,
                 plot,
                 verbose):

        self.results = None
        self.mistakes = None
        self.period = period
        self.raw_data_name = raw_data_name
        self.leave_one_out = leave_one_out
        self.n_trials = n_trials
        self.n_test = n_test
        self.n_validate = n_validate
        self.plot = plot
        self.verbose = verbose

        self.data = Data(self.raw_data_name, period=period, leave_one_out=leave_one_out)
        self.features = self.data.features
        self.classifier = Classifier(classifier_name, self.verbose)

    def run(self):
        print(f'\nTesting {self.classifier.classifier_name} in {self.leave_one_out} scenario on {self.n_test} test sets'
              f' and {self.n_validate} val sets averaging results over {self.n_trials} trials')

        self.results = self.evaluate()
        self.plot_mistakes() if self.plot else None
        self.save_results()

    def evaluate(self):
        results = {}
        train_idx, test_idx = self.data.monte_carlo_leave_1_out(self.features, n_samples=self.n_test)

        directory = f'../results/{self.classifier.classifier_name}/'
        file_path = f'{directory}pred_loo_{self.leave_one_out}_period_12h_marco.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                results = pickle.load(file)
        else:
            #for id_t, (train_idx, test_idx) in tqdm(zip(loo_test), total=len(loo_test), desc='Testing'):
            for id_t in tqdm(range(self.n_test), total=self.n_test, desc='Testing'):
                current_result = self.classifier.train(train_data=self.features.loc[train_idx, :],
                                                       test_data=self.features.loc[test_idx, :],
                                                       n_validate=self.n_validate,
                                                       leave_one_out=self.leave_one_out)

                for key in current_result:
                    if key in results:
                        results[key] = np.concatenate([results[key], current_result[key]])
                    else:
                        results[key] = current_result[key]

        results = self.compute_rec_performance(results)
        return results

    def compute_rec_performance(self, predictions):
        eval_metrics = {
            'accuracy': [],
            'b_accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'err_ratio': -1
        }

        y_true = predictions['true_labels']
        y_pred = predictions['pred_labels']

        n_samples = len(y_true) * 5 // 100

        for _ in tqdm(range(self.n_trials), total=self.n_trials, desc='Running predictions MC: '):
            idxes = random.sample(range(len(y_true)), n_samples)
            p_y_true = y_true[idxes]
            p_y_pred = y_pred[idxes]
            eval_metrics['accuracy'].append(accuracy_score(p_y_true, p_y_pred))
            eval_metrics['b_accuracy'].append(balanced_accuracy_score(p_y_true, p_y_pred))
            eval_metrics['precision'].append(precision_score(p_y_true, p_y_pred))
            eval_metrics['recall'].append(recall_score(p_y_true, p_y_pred))
            eval_metrics['f1_score'].append(f1_score(p_y_true, p_y_pred))

        eval_metrics['err_ratio'] = self.compute_error(p_y_true, p_y_pred)
        results = eval_metrics
        results['true_labels'] = predictions['true_labels']
        results['pred_labels'] = predictions['pred_labels']

        return results

    def compute_error(self, y_true, y_pred):
        self.mistakes = np.where(y_true != y_pred)[0]
        err_ratio = len(self.mistakes) / len(y_true)
        return err_ratio

    def save_results(self):
        directory = f'../results/{self.classifier.classifier_name}/'
        filename = f'{directory}pred_loo_{self.leave_one_out}_period_{self.period}h_{self.raw_data_name}.pkl'
        os.makedirs(directory, exist_ok=True)

        with open(filename, 'wb') as file:
            pickle.dump(self.results, file)

        for metric_name, metric_score in self.results.items():
            if 'labels' not in metric_name:
                if 'err_ratio' in metric_name:
                    print(f'\t{metric_name}: {metric_score:.2%}')
                else:
                    print(f'\t{metric_name}: {np.mean(metric_score):.2%} +/- {np.std(metric_score):.2%}')


if __name__ == "__main__":
    os.chdir('analysis/data')
    parser = argparse.ArgumentParser(description='disk predictions')
    parser.add_argument("--config", type=str, default="config.json", help="config file path")
    args = parser.parse_args()

    with open(f'../{args.config}', 'r') as config_file:
        config = json.load(config_file)

    clf = Eval(**config)
    clf.run()
    print('Analysis complete!!!')
