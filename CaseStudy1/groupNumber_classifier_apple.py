import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def logistic_regression(train_data, train_label, test_data, test_label, reg_type, C):
    lr = LogisticRegression(penalty=reg_type, C=C, solver='saga', max_iter=1000, random_state=1)
    lr.fit(train_data, train_label)
    return lr.score(test_data, test_label)


def SVM(train_data, train_label, test_data, test_label, kernel, C):
    svm = SVC(kernel=kernel, C=C, random_state=1)
    svm.fit(train_data, train_label)
    return svm.score(test_data, test_label)

def decision_tree(train_data, train_label, test_data, test_label, criterion, max_depth):
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dt.fit(train_data, train_label)
    return dt.score(test_data, test_label)


path = './Data/train.csv'
df = pd.read_csv(path)
labels = df.pop(df.columns[len(df.columns) - 1])
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)

stdsc = StandardScaler()
stdsc.fit(df.values)
data = stdsc.transform(df.values)
# KFold(n_splits=2, random_state=None, shuffle=False)

kf = KFold(n_splits=5, random_state=None, shuffle=False)
reg_types = ['l1', 'l2']
logistic_regression_highest_accuracy = 0
best_reg = None
lr_C = None
kernel_types = ['poly', 'rbf', 'sigmoid']
svc_highest_accuracy = 0
best_kernel = None
SVM_C = None
criterion = ['gini', 'entropy', 'log_loss']
decision_tree_accuracy = 0
best_criteria = None
best_max_depth = None

# for i, (train_index, test_index) in enumerate(kf.split(data)):
#     train_data = np.take(data, train_index, axis=0)
#     train_label = np.take(labels, train_index)
#     test_data = np.take(data, test_index, axis=0)
#     test_label = np.take(labels, test_index)
#     # logistic regression
#     for i in reg_types:
#         for exponent in range(-3, 4):
#             accuracy = logistic_regression(train_data, train_label, test_data, test_label, i, 10 ** exponent)
#             error = len(test_data) * (1 - accuracy)     # Error rate = 1 - accuracy
#             print(len(test_data), ";",error)
#             # print("Logistic Regression: regularization type", i, ", C:", 10 ** exponent, ", Accuracy:", accuracy)
#             # if accuracy > logistic_regression_highest_accuracy:
#             #     logistic_regression_highest_accuracy = accuracy
#             #     best_reg = i
#             #     lr_C = 10 ** exponent
#
#     # SVC
#     for i in kernel_types:
#         for exponent in range(-3, 4):
#             accuracy = SVM(train_data, train_label, test_data, test_label, i, 10 ** exponent)
#             # print("SVC: kernel type", i, ", C:", 10 ** exponent, ", Accuracy:", accuracy)
#             # if accuracy > svc_highest_accuracy:
#             #     svc_highest_accuracy = accuracy
#             #     best_kernel = i
#             #     SVM_C = 10 ** exponent
#
#     # decision tree
#     for i in criterion:
#         for max_depth in range(5, 30, 5):
#             accuracy = decision_tree(train_data, train_label, test_data, test_label, i, max_depth)
#             # print("Decision Tree: criteria", i, ", max depth:", max_depth, ", Accuracy:", accuracy)
#             # if accuracy > decision_tree_accuracy:
#             #     decision_tree_accuracy = accuracy
#             #     criteria = i
#             #     best_max_depth = max_depth

totalData = len(data)

# logistic regression
for reg in reg_types:
    for exponent in range(-3, 4):
        errorSum = 0
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = np.take(data, train_index, axis=0)
            train_label = np.take(labels, train_index)
            test_data = np.take(data, test_index, axis=0)
            test_label = np.take(labels, test_index)
            accuracy = logistic_regression(train_data, train_label, test_data, test_label, reg, 10 ** exponent)
            error = len(test_data) * (1 - accuracy)  # Error rate = 1 - accuracy
            errorSum += error
        errorRate = errorSum / totalData
        accuracyRate = 1 - errorRate
        print("Logistic Regression: regularization:", reg, ", C:", 10 ** exponent, ", Error Rate:", errorRate,
              ", Accuracy:", accuracyRate)
        if accuracyRate > logistic_regression_highest_accuracy:
            logistic_regression_highest_accuracy = accuracyRate
            best_reg = reg
            lr_C = 10 ** exponent

# SVM
for kernel in kernel_types:
    for exponent in range(-3, 4):
        errorSum = 0
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = np.take(data, train_index, axis=0)
            train_label = np.take(labels, train_index)
            test_data = np.take(data, test_index, axis=0)
            test_label = np.take(labels, test_index)
            accuracy = SVM(train_data, train_label, test_data, test_label, kernel, 10 ** exponent)
            error = len(test_data) * (1 - accuracy)  # Error rate = 1 - accuracy
            errorSum += error
        errorRate = errorSum / totalData
        accuracyRate = 1 - errorRate
        print("SVM: kernel:", kernel, ", C:", 10 ** exponent, ", Error Rate:", errorRate, ", Accuracy:", accuracyRate)
        if accuracyRate > svc_highest_accuracy:
            svc_highest_accuracy = accuracyRate
            best_kernel = kernel
            SVM_C = 10 ** exponent

# decision tree
for criteria in criterion:
    for max_depth in range(5, 30, 5):
        errorSum = 0
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = np.take(data, train_index, axis=0)
            train_label = np.take(labels, train_index)
            test_data = np.take(data, test_index, axis=0)
            test_label = np.take(labels, test_index)
            accuracy = decision_tree(train_data, train_label, test_data, test_label, criteria, max_depth)
            error = len(test_data) * (1 - accuracy)  # Error rate = 1 - accuracy
            errorSum += error
        errorRate = errorSum / totalData
        accuracyRate = 1 - errorRate
        print("Decision Tree: criteria:", criteria, ", max depth:", max_depth, ", Error Rate:", errorRate,
              ", Accuracy:", accuracyRate)
        if accuracyRate > decision_tree_accuracy:
            decision_tree_accuracy = accuracyRate
            best_criteria = criteria
            best_max_depth = max_depth

print("Best of Logistic Regression Accuracy: ", logistic_regression_highest_accuracy, "reg: ", best_reg, ", C: ", lr_C)
print("Best of SVM Accuracy: ", svc_highest_accuracy, " kernel: ", best_kernel, ", C: ", SVM_C)
print("Best of Decision Tree Accuracy: ", decision_tree_accuracy, "criteria: ", criteria, ", max depth: ", best_max_depth)
