import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set1.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


def predictor(csv_path, dist_metric = "euclidean", linkage_type = "single"):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    # select features
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # X = feature_extraction(X)
    # k-means with K = number of unique VIDs of set1
    lowest_var = float("inf")
    model_with_lowest_var = None
    k_with_lowest_var = None
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    for K in range(6,30):
        model = AgglomerativeClustering(n_clusters=K, metric=dist_metric, linkage=linkage_type)
        # model = KMeans(n_clusters=K, random_state=123, n_init='auto')
        # predict cluster numbers of each sample
        labels_pred = model.fit_predict(X)
        var_sil = evaluate_K(X, labels_pred, K)
        if var_sil< lowest_var:
            lowest_var = var_sil
            model_with_lowest_var = model
            k_with_lowest_var = K
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print("K: ", K, "; var: ", var_sil, "; score: ", rand_index_score)
    print("Best K: ", k_with_lowest_var)
    return model_with_lowest_var.fit_predict(X)

def hyper_parameter_tuning(csv_path):
    linkage_options = ['single','complete','average','ward']
    metric_options = ['euclidean','manhattan','cosine']
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    max_score = -1*float('inf')
    best_params = ('single','euclidean')
    for linkage in linkage_options:
        for metric in metric_options:
            if linkage == 'ward' and metric !='euclidean':
                continue
            labels_pred = predictor(csv_path,dist_metric=metric,linkage_type=linkage)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)
            if rand_index_score> max_score:
                max_score = rand_index_score
                best_params = (linkage,metric)
    print(f'best linkage: {best_params[0]} best metric: {best_params[1]}')
    labels_pred = predictor(csv_path,dist_metric=best_params[1],linkage_type=best_params[0])
    print(f'Adjusted Rand Index Score of {csv_path}: {adjusted_rand_score(labels_true,labels_pred):.4f}')
    return best_params[0], best_params[1]

def evaluate_K(X, labels, K):
    indices_all = np.argsort(labels)
    classes = np.zeros((K), dtype=int)
    for label in labels:
        classes[label] += 1
    indices_label =[]
    index=0
    for class_len in classes:
        indices_label.append(indices_all[index:index+class_len])
        index+=class_len
    silhouette_values = silhouette_samples(X,labels)
    sil_means = np.zeros((K))
    for cluster_index in range(len(indices_label)):
        cluster = indices_label[cluster_index]
        cluster_sil = np.take(silhouette_values,cluster)
        sil_means[cluster_index] = np.mean(cluster_sil)
    # print("indices labels:", indices_label)
    # print("sil means:", sil_means)
    return np.var(sil_means)


def feature_extraction(X):
    pca = PCA(n_components=4)
    pca.fit(X)
    return pca.transform(X)




if __name__=="__main__":
    get_baseline_score()
    hyper_parameter_tuning('./Data/set2.csv')
    evaluate()
    


