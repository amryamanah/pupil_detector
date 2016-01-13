import os

from sklearn.externals import joblib

from livestockwatch.bow import BagOfWord

num_cluster = 140
channel_type = "gray"
model_name = "model_dense_lbp_gray"

# for lbp_point in [8, 16, 24]:
#     for lbp_radius in range(2, 7):
#         bow = BagOfWord("dataset_oct", model_name, channel_type, lbp_point, lbp_radius)
#         bow.load_dataset()
#
#         for num_cluster in range(100, 210, 20):
#             cluster_formation_path = os.path.join(model_name, "lbp_p{}_r{}".format(lbp_point, lbp_radius),
#                                       str(num_cluster), "kmeans", "bow_kmeans.pkl")
#
#             if not os.path.exists(cluster_formation_path):
#                 clf = bow.clustering(num_cluster=num_cluster)
#             else:
#                 clf = joblib.load(cluster_formation_path)
#             train_features, test_features = bow.prepare_training_data(clf)
#             bow.training(clf, "rbf", 5, train_features, test_features)

for lbp_point in [24]:
    for lbp_radius in range(5, 7):
        bow = BagOfWord("dataset_oct", model_name, channel_type, lbp_point, lbp_radius)
        bow.load_dataset()

        for num_cluster in range(100, 210, 20):
            cluster_formation_path = os.path.join(model_name, "lbp_p{}_r{}".format(lbp_point, lbp_radius),
                                      str(num_cluster), "kmeans", "bow_kmeans.pkl")

            if not os.path.exists(cluster_formation_path):
                clf = bow.clustering(num_cluster=num_cluster)
            else:
                clf = joblib.load(cluster_formation_path)
            train_features, test_features = bow.prepare_training_data(clf)
            bow.training(clf, "rbf", 5, train_features, test_features)
