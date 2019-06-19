#!/usr/bin/env python
# coding: utf-8

# Helping libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2


class MultiColumn_LabelEncoder:
    
    # Specify column names that needs to be encoded
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y):
        return self
    
    def transform(self, X, y=None):
        output = X.copy()
        if self.columns is not None:
            for column in self.columns:
                output[column] = LabelEncoder().fit_transform(output[column])
                output[column] = output[column].astype('category')
        else:
            for column_name, column in output.iteritems():
                output[column_name] = LabelEncoder().fit_transform(column)
        return output
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OneHotEncode:
    
    # Specify data and column_names
    def __init__(self, data):
        self.data = data
    
    def one_hot_encode(self):
        # Seperating Features and Labels
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        
        columns_to_encode = list(X.select_dtypes(include=['category', object]))
        if not columns_to_encode:
            print("No attributes to one hot encode.")
            X_new = X.copy()
        else:
            print("Attributes to one hot encode :", columns_to_encode)
            # One-hot encoding on Features with categorical values
            X_new = pd.get_dummies(X, drop_first=True, columns=columns_to_encode)
    
        return X_new, y


class Transform:
    def __init__(self, X, y, minmax=None, normalize=False):
        self.X = X
        self.y = y
        self.minmax = minmax
        self.normalize=normalize
    
    def min_max_normalize(self):
        features = np.array(self.X.iloc[:, :].values)
        labels = np.array(self.y.iloc[:].values)
        
        N, dim = features.shape
        
        # Rescaling data between minimum and maximum value
        if self.minmax is not None:
            min_max = MinMaxScaler(feature_range=self.minmax, copy=False)
            rescaled_features = min_max.fit_transform(features)
        
        # Normalizing data (L2 normalization)
        if self.normalize:
            normalizer = Normalizer(copy=False)
            rescaled_features = normalizer.fit_transform(rescaled_features)
        
        features = rescaled_features
        
        return features, labels


def preprocess_data(file_path, nan_values='?', minmax=None, normalize=False):
    data = pd.read_csv('{}'.format(file_path), na_values=nan_values)
    # Handling NAN values_training.csv', na_values=' ?')
    data = data.dropna()
    
    # Array of column names with data type as object (non integer or float)
    object_attributes = list(data.select_dtypes(include='object'))
    if not object_attributes:
        print("No attributes to label encode.")
        new_data = data
    else:
        print("Attributes for label encoding: ", object_attributes)
        label_encoder = MultiColumn_LabelEncoder(columns=object_attributes)
        new_data = label_encoder.fit_transform(data)
    
    X, y = OneHotEncode(data=new_data).one_hot_encode()
    print("\nColumn names after processing :\n", list(X.columns))
    print("\nTotal number of columns: ", len(list(X.columns)))
    
    # Numpy array for features and labels
    transform_data = Transform(X=X, y=y, minmax=minmax, normalize=normalize)
    features, labels = transform_data.min_max_normalize()
    
    # Check scores of each attribute for selecting best ones
    selector = SelectKBest(score_func=chi2, k='all')
    X_new = selector.fit_transform(features, labels)
    print("\nFeature scores based on chi2: ", selector.scores_)
    
    return features, labels


def main():
    file = 'datasets/adult_training.csv'
    preprocess_data(file_path=file, nan_values=' ?', minmax=(0,1), normalize=False)


if __name__ == '__main__':
    main()

