import os
import logging
from time import time
import pprint

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.externals import joblib

from .descriptors import DenseLBPDescriptor
from .base_classifier import BaseClassifier
from .utils import write_csv_result, plot_confusion_matrix, get_specific_channel

from IPython import embed
logger = logging.getLogger(__name__)


class BagOfWord(BaseClassifier):

    def __init__(self, ds_root_path, model_result_folder, channel_type, lbp_points, lbp_radius):
        super().__init__()

        self.ds_root_path = ds_root_path
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.model_result_folder = model_result_folder
        self.channel_type = channel_type

        self.num_neg_data = 221
        self.num_pos_class_data = 125
        self.lbp_feature_shape = (361, self.lbp_points + 1)
        self.dataset = None
        self.dataset_label = None
        self.cf_dataset = None
        self.train_dataset = None
        self.train_dataset_label = None
        self.test_dataset = None
        self.test_dataset_label = None

        self.result_folder = os.path.join(model_result_folder, "lbp_p{}_r{}".format(lbp_points, lbp_radius))

        os.makedirs(self.result_folder, exist_ok=True)
        self.clf_folder = None
        self.num_cluster = None

        self.lbp_descriptor = DenseLBPDescriptor(self.lbp_points, self.lbp_radius)

    def _get_clf_folder(self, clf):
        clf_folder = os.path.join(self.result_folder, str(clf.n_clusters))
        logger.info("CLF Folder = {}".format(clf_folder))
        return clf_folder

    def load_dataset(self):
        t0 = time()
        logger.info("[START] Load data set ")
        nopl_count = 0
        nopl_data = np.empty((self.num_pos_class_data, self.lbp_feature_shape[0], self.lbp_feature_shape[1]),
                             dtype=np.float64)
        nopl_label = np.empty((self.num_pos_class_data,), dtype=np.int)

        pl_count = 0
        pl_data = np.empty((self.num_pos_class_data, self.lbp_feature_shape[0], self.lbp_feature_shape[1]),
                           dtype=np.float64)
        pl_label = np.empty((self.num_pos_class_data,), dtype=np.int)

        neg_count = 0
        neg_data = np.empty((self.num_neg_data, self.lbp_feature_shape[0], self.lbp_feature_shape[1]),
                            dtype=np.float64)
        neg_label = np.empty((self.num_neg_data,), dtype=np.int)

        for dirpath, dirnames, files in os.walk(self.ds_root_path):
            for file in files:
                if file.endswith(".bmp"):
                    label_name = os.path.basename(dirpath)
                    image = cv2.imread(os.path.join(dirpath, file))
                    gray = get_specific_channel(image, self.channel_type)

                    lbp_feature = self.lbp_descriptor.describe(gray)
                    if label_name == "neg_320":
                        neg_data[neg_count] = lbp_feature
                        neg_label[neg_count] = 0
                        neg_count += 1
                    elif label_name == "nopl_320":
                        nopl_data[nopl_count] = lbp_feature
                        nopl_label[nopl_count] = 1
                        nopl_count += 1
                    elif label_name == "pl_320":
                        pl_data[pl_count] = lbp_feature
                        pl_label[pl_count] = 1
                        pl_count += 1

        self.dataset = np.concatenate((pl_data, nopl_data, neg_data))
        self.dataset_label = np.concatenate((pl_label, nopl_label, neg_label))

        (self.train_dataset, self.test_dataset, self.train_dataset_label, self.test_dataset_label) = train_test_split(
            np.array(self.dataset), self.dataset_label, test_size=0.2, random_state=42
        )

        self.cf_dataset = np.vstack(self.dataset)
        # self.cf_dataset = np.vstack(self.train_dataset)

        logger.info("done in %0.3fs" % (time() - t0))
        logger.info("[START] Finish loading data set ( cluster formation = {}, train = {}, testing = {})".format(
            self.cf_dataset.shape, self.train_dataset.shape, self.test_dataset.shape
        ))

    def clustering(self, num_cluster):
        # cluster formation
        start_time = time()
        logger.info("[START] Kmeans Cluster with num cluster = {}".format(num_cluster))

        # Parallel method
        # cluster_formation = KMeans(n_clusters=num_cluster, n_jobs=-1)

        cluster_formation = KMeans(n_clusters=num_cluster)
        cluster_formation.fit(self.cf_dataset)
        clf_folder = os.path.join(self._get_clf_folder(cluster_formation), "kmeans")
        os.makedirs(clf_folder, exist_ok=True)

        joblib.dump(cluster_formation, os.path.join(clf_folder, "bow_kmeans.pkl"))

        end_time = time() - start_time
        logger.info("done in %0.3fs" % end_time)
        logger.info("[Finish] Kmeans Cluster with num cluster = {}".format(cluster_formation.n_clusters))
        kmeans_csv_header = ["num_cluster", "timetaken_second"]
        kmeans_csv_path = os.path.join(self._get_clf_folder(cluster_formation), "kmeans_time.csv")
        kmeans_time_data = {
            "num_cluster": num_cluster,
            "timetaken_second": end_time
        }

        write_csv_result(kmeans_csv_path, kmeans_csv_header, kmeans_time_data)
        return cluster_formation

    def prepare_training_data(self, clf, split=True):
        t0 = time()
        logger.info("[Start] Prepare SVM training data")
        if split:
            train_features = np.empty((self.train_dataset.shape[0], clf.n_clusters))
            train_features_count = 0
            # learn from cluster data
            for dataset in self.train_dataset:
                lbp_cluster_hist = np.empty((clf.n_clusters,))
                labels = clf.predict(dataset)
                for x in range(0, clf.n_clusters):
                    lbp_cluster_hist[x] = labels.tolist().count(x)
                train_features[train_features_count] = lbp_cluster_hist
                train_features_count += 1

            test_features = np.empty((self.test_dataset.shape[0], clf.n_clusters))
            test_features_count = 0
            for dataset in self.test_dataset:
                lbp_cluster_hist = np.empty((clf.n_clusters,))
                labels = clf.predict(dataset)
                for x in range(0, clf.n_clusters):
                    lbp_cluster_hist[x] = labels.tolist().count(x)
                test_features[test_features_count] = lbp_cluster_hist
                test_features_count += 1

            logger.info("done in %0.3fs" % (time() - t0))
            logger.info("[Finish] Prepare SVM training data")
            return train_features, test_features
        else:
            dataset_features = np.empty((self.dataset.shape[0], clf.n_clusters))
            dataset_features_count = 0
            for dataset in self.dataset:
                lbp_cluster_hist = np.empty((clf.n_clusters,))
                labels = clf.predict(dataset)
                for x in range(0, clf.n_clusters):
                    lbp_cluster_hist[x] = labels.tolist().count(x)
                dataset_features[dataset_features_count] = lbp_cluster_hist
                dataset_features_count += 1
            logger.info("done in %0.3fs" % (time() - t0))
            logger.info("[Finish] Prepare SVM training data")
            return dataset_features

    def training(self, cluster_formation, kernel_type, kfold_num, train_features, test_features):
        ###############################################################################

        perf_data = {'lbp_points': self.lbp_points, 'lbp_radius': self.lbp_radius,
                     'km_num_clust': cluster_formation.n_clusters, 'kvm_kernel_type': kernel_type,
                     'cv_kfold_num': kfold_num,
                     'prec_neg': None, 'prec_pos': None, 'prec_weight_avg': None,
                     'recall_neg': None, 'recall_pos': None, 'recall_weight_avg': None,
                     'f1_neg': None, 'f1_pos': None, 'f1_weight_avg': None,
                     'test_data_pos': None, 'test_data_neg': None, 'test_data_tot': None}

        perf_header = [
            "lbp_points", "lbp_radius", "km_num_clust", "kvm_kernel_type", "cv_kfold_num",
            "prec_neg", "prec_pos", "prec_weight_avg",
            "recall_neg", "recall_pos", "recall_weight_avg",
            "f1_neg", "f1_pos", "f1_weight_avg",
            "test_data_pos", "test_data_neg", "test_data_tot"
        ]

        logger.info('[Start] Train SVM with "{}" kernel'.format(kernel_type))
        start_time = time()
        # Train a SVM classification model
        print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

        cv = cross_validation.StratifiedKFold(self.train_dataset_label, n_folds=kfold_num,
                                              shuffle=False, random_state=None)

        svm_classification = GridSearchCV(SVC(kernel=kernel_type, class_weight='balanced'), param_grid, cv=cv)

        svm_classification = svm_classification.fit(train_features, self.train_dataset_label)

        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(svm_classification.best_estimator_)

        ###############################################################################
        # Quantitative evaluation of the model quality on the test set

        print("Predicting Cattle Eye on the test set")
        t0 = time()
        y_pred = svm_classification.predict(test_features)
        print("done in %0.3fs" % (time() - t0))

        labels = unique_labels(self.test_dataset_label, y_pred)
        p, r, f1, s = metrics.precision_recall_fscore_support(self.test_dataset_label, y_pred, labels=labels,
                                                              average=None, sample_weight=None)
        perf_data["prec_neg"] = p[0]
        perf_data["prec_pos"] = p[1]
        perf_data["prec_weight_avg"] = ((p[0] * s[0]) + (p[1] * s[1])) / np.sum(s)
        perf_data["recall_neg"] = r[0]
        perf_data["recall_pos"] = r[1]
        perf_data["recall_weight_avg"] = ((r[0] * s[0]) + (r[1] * s[1])) / np.sum(s)
        perf_data["f1_neg"] = f1[0]
        perf_data["f1_pos"] = f1[1]
        perf_data["f1_weight_avg"] = ((f1[0] * s[0]) + (f1[1] * s[1])) / np.sum(s)
        perf_data["test_data_neg"] = s[0]
        perf_data["test_data_pos"] = s[1]
        perf_data["test_data_tot"] = np.sum(s)

        logger.info("Classification result")
        logger.info(pprint.pformat(perf_data))

        write_csv_result(os.path.join(self.result_folder, "report.csv"), perf_header, perf_data)

        cm = metrics.confusion_matrix(self.test_dataset_label, y_pred, labels=range(2))

        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)

        plot_confusion_matrix(self.result_folder, cm_normalized, "cf_lbp{}x{}_kmeans{}_svm_{}".format(
            self.lbp_points, self.lbp_radius, cluster_formation.n_clusters, kernel_type
        ), title='Normalized confusion matrix')

        svm_folder = os.path.join(self._get_clf_folder(cluster_formation), kernel_type)
        os.makedirs(svm_folder, exist_ok=True)
        joblib.dump(svm_classification, os.path.join(svm_folder, "bow_svm.pkl"))
        print("done in %0.3fs" % (time() - start_time))
        logger.info('[END] Train SVM with "{}" kernel'.format(kernel_type))