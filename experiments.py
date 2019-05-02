import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import utils
import pandas as pn


def remove_features(classifier_factory, k, num_of_features=52, norm='guass'):
    features_to_remove = []
    curr_acc = evaluate_selected_features(classifier_factory, k, features_to_remove, norm=norm)

    for i in range(num_of_features):
        acc = evaluate_selected_features(classifier_factory, k, features_to_remove+[i], norm=norm)
        if acc >= curr_acc:
            features_to_remove.append(i)

            curr_acc = acc
    return features_to_remove

def evaluate_selected_features(classifier_factory, k, features_to_remove=[], print_stats=False, norm='guass'):
    #create k sets
    sub_trains = []
    sub_labels = []
    for i in range(k):
        data, label = load_data()
        # data = manipulate_examples(data)
        data = np.delete(data, features_to_remove, 1)
        if norm == 'guass':
            data = normalize_data_guass(data)
        else:
            data = normalize_data(data)
        sub_trains.append(data)
        sub_labels.append(label)

    #cross validation
    accuracies = []
    errors = []
    for i in range(k):
        validation_data = sub_trains[i]
        validation_true_label = sub_labels[i]
        validation_classifier_label = []
        train_data = []
        train_y_labels = []
        for index, data in enumerate(sub_trains):
            if index != i:
                for idx, example in enumerate(data):
                    train_data.append(example)
                    train_y_labels.append(sub_labels[index][idx])

        train_data = np.array(train_data)

        classifier = classifier_factory.train(train_data, train_y_labels)
        for voter in validation_data:
            validation_classifier_label.append(classifier.classify(voter))
        cur_accuracy = calc_accuracy(validation_true_label, validation_classifier_label,
                                                           print_stats)

        accuracies.append(cur_accuracy)
        errors.append(1-cur_accuracy)

    return sum(accuracies)/float(len(accuracies))

def load_data():

    ElectionData = pn.read_csv(r'ElectionsData.csv', header=0)
    clean_args = {'df': ElectionData, 'features_info_dict': None, 'drop_features': False, 'negative_to_mean': True,
                  'labels_to_unique_ints': False,
                  'nominal_to_bool_split': True, 'missing_values_fill': True, 'binary_to_numeric': True,
                  'normalization': False}
    ElectionData = utils.clean_data(*clean_args.values())
    train_features, train_labels = ElectionData.drop('Vote', axis=1).values, ElectionData['Vote'].values
    return train_features, train_labels

def normalize_data_guass(data):
    data = np.array(data)
    for i in range(data.shape[1]):
        if type(data[:, i][0]) is not str:
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            data[:, i] -= mean
            data[:, i] /= std
    return data

def normalize_data(data):
    data = np.array(data)
    for i in range(data.shape[1]):
        if type(data[:, i][0]) is not str:
            col_min = min(data[:, i])
            col_max = max(data[:, i])
            data[:, i] -= col_min
            data[:, i] /= (col_max - col_min)
    return data


def calc_accuracy(labels, evaluated_labels, print_stats=False): #TODO change
    count = 0
    for label, evaluated_label in zip(enumerate(labels), evaluated_labels):
        _,label = label
        evaluated_label = evaluated_label[0]
        if label == evaluated_label:
            count += 1
    return float(count)/float(len(labels))


def experiments():
    # clf = SVM_classifier_factory()
    # features_to_remove = remove_features(clf, 2, num_of_features=52)
    # acc_svm = evaluate_selected_features(clf, 2, features_to_remove)
    # print('final acc svm: ', acc_svm)
    # print('best feat for svm',features_to_remove)



    clf = AdaBoost_classifier_factory()
    features_to_remove = remove_features(clf, 2, num_of_features=52)
    acc_ada = evaluate_selected_features(clf, 2, features_to_remove)
    print('final acc ada: ', acc_ada)
    print('best feat for ada', features_to_remove)


    clf = ID3_classifier_factory()
    features_to_remove = remove_features(clf, 2, num_of_features=52)
    acc_id3 = evaluate_selected_features(clf, 2, features_to_remove, print_stats=True)
    print('final acc id3: ', acc_id3)
    print('best feat for id3', features_to_remove)


    clf = KNN_greedy_classifier_factory(10)
    features_to_remove = remove_features(clf, 2, num_of_features=52)
    acc = evaluate_selected_features(clf, 2, features_to_remove, True)
    print('final acc knn: ', acc)
    print('best feat for knn', features_to_remove)

    # clf = KNN_greedy_classifier_factory(100)
    # features_to_remove = remove_features(clf, 2, num_of_features=52)
    # print(type(clf), type(features_to_remove))
    # acc_knn, mistake_knn = evaluate_selected_features(clf, 2, features_to_remove)
    # print('final acc knn: ', acc_knn)
    # print('best feat for knn', features_to_remove)

class abstract_classifier_factory:
    '''
    an abstruct class for classifier factory
    '''
    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: abstruct_classifier object
        '''
        raise Exception('Not implemented')

class abstract_classifier:
    '''
        an abstruct class for classifier
    '''

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''
        raise Exception('Not implemented')

class ID3_classifier_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return ID3_classifier(data, labels)

class ID3_classifier(abstract_classifier):
    def __init__(self, examples, labels):
        self.classifier = DecisionTreeClassifier(criterion="entropy")
        self.classifier.fit(examples, labels)

    def classify(self, features):
        return self.classifier.predict(np.array(features).reshape(1, -1))


class KNN_greedy_classifier_factory(abstract_classifier_factory):

    def __init__(self, k=5):
        self.k = k

    def train(self, data, labels):
        #normalize the data
        data = normalize_data(data)
        return KNN_greedy_classifier(data, labels, self.k)


class KNN_greedy_classifier(abstract_classifier):
    def __init__(self, examples, labels, k):
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.classifier.fit(examples, labels)

    def classify(self, features):
        return self.classifier.predict(np.array(features).reshape(1, -1))

class AdaBoost_classifier_factory(abstract_classifier_factory):
    def train(self, data, labels):
        #normalize the data
        data = np.array(data)
        for i in range(data.shape[1]):
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            data[:, i] -= mean
            data[:, i] /= std

        return AdaBoost_classifier(data, labels)


class AdaBoost_classifier(abstract_classifier):
    def __init__(self, examples, labels):
        self.classifier = AdaBoostClassifier()
        self.classifier.fit(examples, labels)

    def classify(self, features):
        return self.classifier.predict(np.array(features).reshape(1, -1))



class SVM_classifier_factory(abstract_classifier_factory):
    def train(self, data, labels):
        #normalize the data
        data = normalize_data(data)
        return SVM_classifier(data, labels)


class SVM_classifier(abstract_classifier):
    def __init__(self, examples, labels):
        self.classifier = SVC(class_weight='balanced')
        self.classifier.fit(examples, labels)

    def classify(self, features):
        return self.classifier.predict(np.array(features).reshape(1, -1))

