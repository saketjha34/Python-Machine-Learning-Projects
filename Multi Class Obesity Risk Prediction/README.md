
# Multi Class Obesity Risk Prediction

The goal of this project is to develop a machine learning model to classify individuals into different obesity risk categories based on demographic, lifestyle, and medical information.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

### Introduction to Multi-Class Obesity Risk Prediction Project
Obesity is a global health issue with significant medical and socio-economic impacts. The Multi-Class Obesity Risk Prediction project aims to classify individuals into different obesity risk categories using machine learning techniques. By leveraging a dataset containing demographic, lifestyle, and medical information, the project seeks to identify key factors contributing to obesity and provide early risk assessments. This can help in implementing targeted interventions and preventive measures to combat obesity-related health problems.

Key Objectives:
- Data Collection and Preprocessing: Gather and clean data from the Kaggle competition, handle missing values, and prepare features.
- Model Development: Train and optimize various machine learning models to predict obesity risk categories.
- Model Evaluation: Use metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
- Deployment and Application: Deploy the model for practical use in predicting obesity risk and informing health interventions.

This project highlights the importance of predictive analytics in public health and aims to contribute valuable insights into obesity risk factors. You can access the dataset and additional information on the Kaggle competition page.

## Project Structure

```
.
├── csv
│   ├── obesitydataset.csv
│   ├── test.csv
│   ├── train.csv
│   ├── sample_submission.csv
├── jupyter notebooks
│   ├── DataPreprocessing.ipynb
│   ├── ModelTrainingNotebookOptuna.ipynb
│   ├── ObesityRiskPredictionLGBM.ipynb
│   ├── ObesityRiskPredictionXGBoost.ipynb
│   ├── ObesityRiskPredictionNeuralNetworks.ipynb
├── files
│   ├── Model Scores
│   ├── Optimization Plots
│   ├── Model Benchmarks/LGBM
│   ├── Model Benchmarks/XGB
│   ├── Model Benchmarks/NeuralNetworks
├── preprocessed data
│   ├── train
│   ├── val
│   ├── test
├── utils
│   ├── basic_utils.py
│   ├── neural_network_models.py
│   ├── neural_network_utils.py
│   ├── hyperparameter_utils.py
├── saved models
│   ├── LGBModel.joblib
│   ├── XGBModel.joblib
├── README.md
└── requirements.txt
```


## Dataset

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Obesity or CVD risk dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
dataset can be found in `csv` directory

```bash
cd csv
cd preprocessed data

cd csv/train.csv

cd preprocessed data/train/X.csv
cd preprocessed data/train/targets.csv
```

## Installation

Clone the repository and install the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/saketjha34/Python-Machine-Learning-Projects.git
    cd Python-Machine-Learning-Projects/Multi%20Class%20Obesity%20Risk%20Prediction
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To view model performances and benchmarks check out:

```bash
python Jupyter Notebooks/ObesityRiskPredictionNeuralNetworks.ipynb
python Jupyter Notebooks/ObesityRiskPredictionLGBM.ipynb
python Jupyter Notebooks/ObesityRiskPredictionXGBoost.ipynb
python Jupyter Notebooks/ModelTrainingNotebookOptuna.ipynbb
```

For more detailed instructions, refer to the `utils/` directory, which contains all the necessary scripts for data preprocessing, training, and evaluation.

```bash
python utils/basic_utils.py
python utils/nueral_network_utils.py
python utils/hyperparameter_utils.py
python utils/nueral_network_models.py
```

## Model Architecture

### XGBoost
- XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting library designed for efficiency and performance.
- It uses a gradient boosting framework to build decision trees in a sequential manner.
- Key features include regularization to avoid overfitting, parallel processing, and the ability to handle missing values.

### LightGBM
- LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses tree-based learning algorithms.
- It is designed for efficiency and scalability, especially for large datasets.
- Key features include histogram-based decision trees, leaf-wise growth, and support for parallel and GPU learning.

### Neural Networks
- Neural Networks consist of multiple layers of interconnected neurons that can learn complex patterns in data.
- They are particularly useful for modeling non-linear relationships.
- Key architectures include:
  - **Feedforward Neural Networks:** Basic structure where data flows in one direction from input to output.
  - **Convolutional Neural Networks (CNNs):** Specialized for image data, using convolutional layers to detect spatial hierarchies.
  - **Recurrent Neural Networks (RNNs):** Specialized for sequential data, using recurrent layers to maintain context across time steps.

## Results

|      Models     | Accuracy |
|-----------------|----------|
|    LightGBM     |   92%    |
|    XGBoost      |   90.6%  |
| Neural Network  |   87.5%  |

The `files/Model Benchmarks/` directory contains detailed performance metrics for each model:

Example of accessing benchmarks from the command line:
```bash
cd files/Model Benchmarks/LGBM
cd files/Model Benchmarks/XGB
cd files/Model Benchmarks/NeuralNetwork
```
To run the benchmarks of any particular models run:
```python
from utils.basic_utils import evalmodel , plot_model_importance , plot_confusion_matrix , plot_classification_report , save_model
from xgboost import XGBClassifier

def main():
    evalmodel(XGBClassifier , X_train , train_targets , X_val , val_targets)
    plot_model_importance(XGBClassifier, X_train, train_targets, X_val ,val_targets ,**xgb_params)
    plot_classification_report(XGBClassifier , X_train ,train_targets , X_val , val_targets ,labels=class_names,**xgb_params)
    plot_confusion_matrix(XGBClassifier , X_train,train_targets, X_val , val_targets, labels=class_names,**xgb_param)

if __name__ == "__main__":
    main()

```

## Deployment

The trained models are saved as `.joblib` files in the `saved models` directory. These files can be used for further deployment purposes. You can load the models using the following code:

```python
from utils.basic_utils import load_model

#load LGBM Model
filepath = '..\\saved models\\LGBModel.joblib'
loaded_model, loaded_model_params = load_model(filepath)

filepath = '..\\saved models\\XGBModel.joblib'
loaded_model, loaded_model_params = load_model(filepath)
```

## References

### Neural Networks
- **[Deep Learning](https://www.deeplearningbook.org/)**: Comprehensive book covering the fundamentals of neural networks, including feedforward, convolutional, and recurrent architectures.

### XGBoost
- **[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)**: Paper by Chen and Guestrin detailing the design, implementation, and applications of XGBoost.

### LightGBM
- **[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://www.microsoft.com/en-us/research/publication/lightgbm-a-highly-efficient-gradient-boosting-decision-tree/)**: Paper introducing LightGBM, focusing on its efficiency and scalability.

### SVM (Support Vector Machine)
- **[Support Vector Machines: Concepts and Applications](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)**: Guide by Hsu, Chang, and Lin providing an overview of SVM concepts, implementation, and applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have any improvements or new models to add, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.

## Contact

For any questions or suggestions, please open an issue or contact me @ saketjha0324@gmail.com. or [Linkedin](https://www.linkedin.com/in/saketjha34/)

---

Happy coding!

