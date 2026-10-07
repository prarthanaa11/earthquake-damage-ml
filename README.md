# Application of Machine Learning Methods to Predict the Level of Building Damage in the Nepal Earthquake

## Overview

This project applies machine learning techniques to predict the level of structural damage suffered by buildings during the 2015 Nepal (Gorkha) Earthquake.

Assessing the damage level of a large number of buildings after an earthquake can be time-consuming when performed using traditional structural analysis methods. Machine learning provides an approach for automatically classifying buildings into different damage categories using information about their structural, ownership, usage, and household characteristics.

The project compares three supervised machine learning approaches:

* K-Nearest Neighbors (KNN)
* Backpropagation Neural Network (BPNN)
* Random Forest

The objective is to determine which machine learning method provides the most effective prediction of building damage levels.

---

## Problem Statement

The goal of this project is to develop machine learning models that can predict the damage level of a building after the 2015 Nepal Earthquake based on available building information.

Each building is classified into one of three damage categories:

| Label | Damage Level                |
| ----: | --------------------------- |
|     1 | Low damage                  |
|     2 | Medium damage               |
|     3 | Almost complete destruction |

The project treats building damage prediction as a multi-class classification problem.

---

## Objectives

The main objectives of this project are:

1. Analyze the Nepal Earthquake building dataset.
2. Preprocess the available building features for machine learning.
3. Develop multiple classification models.
4. Compare KNN, Backpropagation Neural Network, and Random Forest.
5. Evaluate model performance using the micro-averaged F1 score.
6. Identify the model that provides the best prediction performance.
7. Analyze the strengths and limitations of each approach.

---

## Dataset

The project uses building information collected following the 2015 Nepal Earthquake.

The dataset consists of three main files:

```text
train_values.csv
train_labels.csv
test_values.csv
```

### `train_values.csv`

Contains the input features for the training buildings.

* Number of examples: **260,601**
* Number of features: **38**
* Each row represents a building.
* Each building is identified by `building_id`.

The features contain information about characteristics such as:

* Number of floors
* Building age
* Foundation type
* Building use
* Ownership status
* Number of families
* Other structural and socio-economic characteristics

### `train_labels.csv`

Contains the damage labels corresponding to the buildings in `train_values.csv`.

The target variable contains three classes:

```text
1 → Low damage
2 → Medium damage
3 → Almost complete destruction
```

### `test_values.csv`

Contains feature information for buildings for which damage labels are not provided.

* Number of examples: **86,868**
* Contains the same feature space used for prediction.

---

## Dataset Directory

Place the dataset files inside the following directory:

```text
data/
├── train_values.csv
├── train_labels.csv
└── test_values.csv
```

> **Note:** The dataset files are not included in this repository if they are too large or subject to the dataset's distribution restrictions. Download them separately and place them inside the `data/` directory.

---

# Methodology

The overall workflow of the project is:

```text
                    Dataset
                       |
                       v
              Data Preprocessing
                       |
          +------------+------------+
          |            |            |
          v            v            v
         KNN          BPNN      Random Forest
          |            |            |
          v            v            v
      Evaluation    Evaluation   Evaluation
          |            |            |
          +------------+------------+
                       |
                       v
                Model Comparison
                       |
                       v
             Best Performing Model
```

---

# Machine Learning Models

## 1. K-Nearest Neighbors

K-Nearest Neighbors is a supervised learning algorithm that classifies an unknown sample based on the classes of its nearest training samples.

For this project:

* Feature selection is performed using `SelectKBest`.
* The 20 highest-scoring features are selected.
* `StandardScaler` is used to standardize the selected features.
* `KNeighborsClassifier` is used for classification.

The number of neighbors is set to:

```text
n_neighbors = 7
```

The KNN model is evaluated using the micro-averaged F1 score.

---

## 2. Backpropagation Neural Network

A two-layer regularized neural network is used for multi-class classification.

The network consists of:

```text
Input Layer
    ↓
Hidden Layer
    ↓
Output Layer
```

The project uses:

* 38 input features
* A hidden layer
* 3 output neurons corresponding to the three damage classes
* Sigmoid activation in the hidden layer
* Softmax activation in the output layer
* Gradient-descent-based training
* Backpropagation
* Regularization

The training data is standardized before being supplied to the neural network.

The target labels are converted into one-hot encoded vectors for neural network training.

The learning rate is tuned to avoid unstable or oscillating training behavior.

---

## 3. Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees and combines their predictions.

Unlike KNN and the neural network, Random Forest does not require feature scaling because it is a tree-based model.

The model uses all 38 original features.

Hyperparameter tuning is performed using `GridSearchCV`.

The parameter grid includes:

```python
{
    "min_samples_leaf": [1, 5],
    "n_estimators": [50, 100]
}
```

The best parameters identified in the reference implementation are:

```text
min_samples_leaf = 1
n_estimators = 100
```

---

# Preprocessing

The preprocessing procedure depends on the machine learning algorithm.

## KNN

KNN is sensitive to feature magnitude because it relies on distance calculations.

Therefore:

```text
Raw Features
     ↓
Feature Selection
     ↓
SelectKBest
     ↓
20 Selected Features
     ↓
StandardScaler
     ↓
Standardized Features
     ↓
KNN
```

## Neural Network

The neural network uses standardized features.

```text
Raw Features
     ↓
Feature Standardization
     ↓
One-Hot Encoded Labels
     ↓
Neural Network
```

## Random Forest

Random Forest operates directly on the original feature representation.

```text
Original 38 Features
        ↓
Random Forest
```

---

# Feature Selection

For KNN, `SelectKBest` with the `f_classif` scoring function is used to select the 20 most informative features.

This reduces the dimensionality of the input to KNN and reduces the computational cost of distance calculations.

The neural network and Random Forest use the full feature set according to the reference methodology.

---

# Model Evaluation

The primary evaluation metric used in this project is the:

## Micro-Averaged F1 Score

The F1 score combines precision and recall.

The F1 score is defined as:

```text
F1 = 2 × Precision × Recall
     -------------------------
       Precision + Recall
```

For the micro-averaged version, the contributions of all classes are aggregated before calculating precision and recall.

A higher F1 score indicates better overall classification performance.

---

# Experimental Results

The reference implementation reported the following results:

| Model          | Configuration                                | Micro F1 Score |
| -------------- | -------------------------------------------- | -------------: |
| KNN            | `n_neighbors = 7`                            |         0.6456 |
| Neural Network | Best reported run                            |         0.6639 |
| Random Forest  | `n_estimators = 100`, `min_samples_leaf = 1` |         0.7214 |

### Best Performing Model

The Random Forest model achieved the highest reported test micro-F1 score:

```text
Micro F1 = 0.7150
```

Therefore, Random Forest performed better than KNN and the neural network in the reported experiments.

---

# Overfitting Analysis

The Random Forest model achieved:

```text
Training Micro F1: 0.9843
Test Micro F1:     0.7150
```

The large difference between training and test performance indicates possible overfitting.

Potential approaches for reducing overfitting include:

* Limiting tree depth
* Increasing regularization
* Adjusting the number of trees
* Tuning `min_samples_leaf`
* Further hyperparameter optimization
* Using cross-validation during model selection

---

# Project Structure

A recommended repository structure is:

```text
nepal-earthquake-damage-prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── train_values.csv
│   ├── train_labels.csv
│   └── test_values.csv
│
├── notebooks/
│   └── earthquake_damage_prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── knn_model.py
│   ├── neural_network.py
│   ├── random_forest.py
│   └── evaluate.py
│
├── models/
│   ├── knn_model.pkl
│   ├── neural_network.pkl
│   └── random_forest.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── feature_scores.csv
│   └── figures/
│
└── predictions/
    └── predictions.csv
```

If your implementation uses a single notebook instead of separate Python files, the repository can be simplified accordingly.

---

# Requirements

The project requires Python 3.x and the following Python libraries:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
joblib
```

The exact versions can be pinned in `requirements.txt` after testing the project environment.

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd nepal-earthquake-damage-prediction
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Download the Nepal Earthquake dataset and place the required files inside:

```text
data/
```

The directory should contain:

```text
data/
├── train_values.csv
├── train_labels.csv
└── test_values.csv
```

Make sure the filenames match the filenames expected by the project code.

---

# How to Run the Project

There are two recommended ways to run the project.

## Option 1 — Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/earthquake_damage_prediction.ipynb
```

Run the notebook cells sequentially.

The notebook performs:

```text
Load Dataset
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Prediction
      ↓
Model Evaluation
      ↓
Model Comparison
```

---

# Option 2 — Python Scripts

If the project is implemented using separate Python scripts, run the preprocessing step first:

```bash
python src/preprocessing.py
```

Then train the KNN model:

```bash
python src/knn_model.py
```

Train the neural network:

```bash
python src/neural_network.py
```

Train and tune the Random Forest:

```bash
python src/random_forest.py
```

Finally, evaluate and compare the models:

```bash
python src/evaluate.py
```

> If your actual files have different names, replace the commands above with the corresponding filenames in your repository.

---

# Expected Output

After successful execution, the project should produce:

* Trained machine learning models
* Predictions for building damage classes
* Evaluation metrics
* Model comparison results
* Feature-selection results for KNN
* Performance visualizations, where implemented

The predicted damage classes are:

```text
1 → Low damage
2 → Medium damage
3 → Almost complete destruction
```

---

# Example Prediction

For a building represented by its feature vector:

```text
Building Features
        ↓
Preprocessing
        ↓
Trained ML Model
        ↓
Predicted Damage Class
```

Example:

```text
Predicted Damage Class: 2
Damage Level: Medium Damage
```

---

# Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* K-Nearest Neighbors
* Random Forest
* Backpropagation Neural Network

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Development

* Jupyter Notebook
* Git
* GitHub

---

# Results Summary

The reported experiments show that Random Forest achieved the strongest predictive performance among the three evaluated approaches.

```text
KNN             → 0.6488 Micro F1
Neural Network  → 0.6502 Micro F1
Random Forest   → 0.7150 Micro F1
```

Random Forest therefore produced the highest reported test performance.

However, the difference between its training and test F1 scores indicates potential overfitting, suggesting that further tuning could improve generalization.

---

# Limitations

The project has several limitations:

1. The Random Forest model shows evidence of overfitting.
2. Neural network performance depends on appropriate learning-rate and architecture tuning.
3. KNN can become computationally expensive with large datasets and high-dimensional feature spaces.
4. The project uses structured building information rather than direct image-based structural damage assessment.
5. The performance of the models depends on the quality and representativeness of the available earthquake dataset.

---

# Future Improvements

Possible future improvements include:

* More extensive hyperparameter optimization
* Cross-validation for model selection
* Better handling of class imbalance, if present
* Advanced ensemble methods
* Improved neural network architectures
* Dimensionality reduction
* Feature importance analysis
* Model explainability
* Deployment as a web-based prediction application
* Integration with post-disaster assessment systems

---

# Conclusion

This project demonstrates the application of supervised machine learning to earthquake building-damage classification.

Three approaches—KNN, Backpropagation Neural Network, and Random Forest—are evaluated on building information collected after the 2015 Nepal Earthquake.

Among the evaluated methods, Random Forest achieves the highest reported test micro-F1 score of **0.7150**, outperforming KNN and the neural network.

The project demonstrates how machine learning can support rapid classification of building damage and potentially assist post-disaster planning, rescue operations, reconstruction, and loss assessment.

---

# References

1. Nepal Earthquake Open Data Portal / DrivenData earthquake damage prediction dataset.
2. Scikit-learn documentation.
3. Relevant research literature on earthquake damage prediction and machine learning.
4. Yitian Liang, *Application of Machine Learning Methods to Predict the Level of Buildings Damage in the Nepal Earthquake*, CS229 Final Project Report, Stanford University.

---

# Authors

**Machine Learning Mini-Project**

Course:

**UE24CS352A – Machine Learning**

Team Members:

PRARTHANA ACHARYA- PES2UG24AM119
PRAGYA -PES2UG24AM115

Institution:

**PES University**

---

## License

This project is developed for academic and educational purposes as part of the Machine Learning mini-project.

Please refer to the original dataset's terms and conditions before redistributing the dataset files.
