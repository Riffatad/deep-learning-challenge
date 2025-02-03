# deep-learning-challenge
# AlphabetSoup Charity Neural Network Analysis

## Project Overview

This project applies deep learning techniques to analyze charity donation data and predict the success of funding requests. Using TensorFlow and Keras, we build and optimize a neural network model that processes categorical and numerical features to classify donation success.

### Files in This Repository

## Starter_Code.ipynb: This file contains the initial implementation of a simple neural network model. It includes:
 
Data loading and preprocessing (removal of unnecessary columns, encoding categorical variables, and scaling numerical data).

Splitting data into training and test sets.

Defining a basic neural network architecture with two hidden layers.

Training the model using binary cross-entropy loss and the Adam optimizer.

Evaluating the model’s accuracy on test data.

Saving the trained model in .keras format.

### AlphabetSoupCharity_Optimization.ipynb: This file improves upon the baseline model by implementing hyperparameter tuning. It includes:

Applying Keras Tuner for optimizing activation functions, neuron count per layer, and number of hidden layers.

Running a hyperparameter search using Hyperband.

Training the best-performing model for 150 epochs.

Evaluating and comparing performance against the baseline model.

Saving the optimized model as .h5 format.

charity_optimization_model.keras: The saved model from Starter_Code.ipynb, which is the initial implementation.

### AlphabetSoupCharity_Optimization_Tuned.h5: The saved optimized model from AlphabetSoupCharity_Optimization.ipynb, incorporating the best hyperparameters identified during tuning.

#### Instructions to Run the Project

#### 1. Run Starter_Code.ipynb

Loads the dataset and performs preprocessing.

Converts categorical variables into numeric format.

Scales the data using StandardScaler.

Builds and trains a simple neural network.

Saves the trained model in .keras format.

#### 2. Run AlphabetSoupCharity_Optimization.ipynb

Applies hyperparameter tuning using Keras Tuner.

Selects the best activation function and neuron count.

Trains the optimized model for 150 epochs.

Saves the final optimized model as .h5.

## Model Summary

Baseline Model (Starter Code)

### Architecture:

1st Hidden Layer: 80 neurons, ReLU

2nd Hidden Layer: 30 neurons, ReLU

Output Layer: 1 neuron, Sigmoid

### Performance:

Accuracy: ~73% (varies slightly per run)

Optimized Model

Hyperparameter Tuning Results:

Best activation function: (varied, based on Keras Tuner results)

Best neuron count: Selected dynamically

Best number of layers: Adjusted dynamically

### Performance:

Accuracy: Improved compared to baseline (~75%+)

### Key Findings

Feature encoding and scaling improve model performance.

Neural network structure significantly impacts accuracy.

Hyperparameter tuning leads to more robust predictions.

### Future Improvements

Experiment with other optimizers like Adamax or RMSprop.

Implement dropout layers to prevent overfitting.

Test alternative machine learning models such as Random Forest or XGBoost for comparison.

## Conclusion

This project demonstrates the impact of deep learning on charity donation predictions. The optimized model provides an improved accuracy rate through hyperparameter tuning, making it a valuable tool for donation classification tasks.

