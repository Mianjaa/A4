import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class BagOfWordsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_features=5000, C=1.0):
        """
        Initialize the Bag-of-Words Classifier
        
        Parameters:
        max_features (int): Maximum number of features (words) to use
        C (float): Inverse of regularization strength
        """
        self.max_features = max_features
        self.C = C
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=self.max_features)),
            ('classifier', LogisticRegression(C=self.C, max_iter=1000))
        ])
    
    def fit(self, X, y):
        """
        Fit the classifier on the training data
        
        Parameters:
        X (list): Training text data
        y (array): Training labels
        
        Returns:
        self: Fitted classifier
        """
        # Fit the pipeline
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict labels for input data
        
        Parameters:
        X (list): Text data to predict
        
        Returns:
        array: Predicted labels
        """
        # Use pipeline to predict
        return self.pipeline.predict(X)
