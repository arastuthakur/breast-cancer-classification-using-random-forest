#!/usr/bin/env python3
        # -*- coding: utf-8 -*-

        """Machine Learning Project

        This module implements a complete machine learning pipeline including data preprocessing,
        model training, evaluation, and visualization.
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix,
            roc_curve, auc, precision_recall_curve
        )
        from sklearn.ensemble import RandomForestClassifier
        import logging
        from typing import Tuple, Dict, Any
        import warnings
        warnings.filterwarnings('ignore')

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import logging

class MLProject:
    def __init__(self, data_path):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
        except FileNotFoundError:
            self.logger.error("File not found")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")

    def explore_data(self):
        try:
            # EDA
            print(self.data.head())
            print(self.data.describe())
            
            # Visualizations
            plt.figure(figsize=(10,6))
            sns.scatterplot(x=self.data['mean_radius'], y=self.data['mean_texture'], hue=self.data['diagnosis'])
            plt.title('Scatter Plot of Mean Radius vs. Mean Texture')
            plt.xlabel('Mean Radius')
            plt.ylabel('Mean Texture')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error exploring data: {e}")

    def preprocess_data(self):
        try:
            # Split data
            self.X = self.data.drop('diagnosis', axis=1)
            self.y = self.data['diagnosis']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                            test_size=0.2, random_state=42)

            # Scale data
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")

    def train_model(self):
        try:
            # Specify parameter grid for GridSearchCV
            param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}

            # Initialize and train RandomForest model using GridSearchCV
            self.model = RandomForestClassifier()
            self.model.fit(self.X_train, self.y_train)
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")

    def evaluate_model(self):
        try:
            # Calculate metrics
            self.predictions = self.model.predict(self.X_test)
            print(classification_report(self.y_test, self.predictions))
            
            # Plot classification report
            plt.figure(figsize=(10,6))
            sns.heatmap(classification_report(self.y_test, self.predictions), 
                        annot=True, fmt='g')
            plt.title('Classification Report Heatmap')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")

def main():
    # Initialize the MLProject class
    try:
        project = MLProject()
    except Exception as e:
        print("Error initializing MLProject: {}".format(e))
        return

    # Run the complete pipeline
    try:
        project.run_pipeline()
    except Exception as e:
        print("Error running pipeline: {}".format(e))
        return

    # Print evaluation results
    try:
        print("Evaluation results:")
        for metric, value in project.evaluation_results.items():
            print("{}: {}".format(metric, value))
    except Exception as e:
        print("Error printing evaluation results: {}".format(e))

if __name__ == "__main__":
    main()