# League of Legends Match Outcome Predictor

## Overview
This project implements a logistic regression model using PyTorch to predict the outcome of **League of Legends** matches based on in-game statistics. It demonstrates a complete machine learning pipeline, from data preprocessing to model evaluation and optimization.

The goal is to apply supervised learning techniques to a real-world esports dataset and gain insights into factors that influence match outcomes.

## Objectives
- Load and preprocess match data for model training
- Build a logistic regression model using PyTorch
- Train and optimize the model using gradient descent
- Evaluate performance using classification metrics
- Visualize results and interpret feature importance
- Save and reload trained models for reuse
- Perform hyperparameter tuning to improve performance

## Project Structure

### Data Loading & Preprocessing
Data is cleaned, split into training and testing sets, and standardized before being converted into PyTorch tensors.

### Model Implementation
A logistic regression model is implemented using `nn.Linear` and `nn.Sigmoid`.

### Training & Optimization
The model is trained using stochastic gradient descent with optional L2 regularization (weight decay).

### Evaluation
Performance is measured using accuracy, confusion matrices, and ROC curves.

### Visualization & Interpretation
Results are visualized to better understand prediction quality and feature impact.

### Model Persistence
The trained model can be saved and reloaded using PyTorch utilities.

### Hyperparameter Tuning
Different learning rates are tested to identify the optimal configuration.

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
1. Clone the repository  
2. Open the Jupyter Notebook  
3. Run all cells in order to train and evaluate the model  

No additional configuration is required beyond the listed dependencies.

## Results
The model successfully learns patterns from match statistics and achieves meaningful predictive performance. Evaluation metrics and visualizations provide insight into model reliability and feature influence.

## License
This project is released under the MIT License.
