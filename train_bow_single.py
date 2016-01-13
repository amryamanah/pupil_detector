import os

from sklearn.externals import joblib

from livestockwatch.bow import BagOfWord

lbp_point = 24
lbp_radius = 4
num_cluster = 140
channel_type = "blue"
model_name = "model_dense_lbp_blue"
bow = BagOfWord("dataset_oct", model_name, channel_type, lbp_point, lbp_radius)

bow.load_dataset()

cluster_formation_path = os.path.join(model_name, "lbp_p{}_r{}".format(lbp_point, lbp_radius),
                                      str(num_cluster), "kmeans", "bow_kmeans.pkl")

if not os.path.exists(cluster_formation_path):
    clf = bow.clustering(num_cluster=num_cluster)
else:
    clf = joblib.load(cluster_formation_path)
train_features, test_features = bow.prepare_training_data(clf)
bow.training(clf, "rbf", 5, train_features, test_features)

