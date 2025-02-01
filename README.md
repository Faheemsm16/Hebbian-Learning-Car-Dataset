# Car Evaluation using Hebbian Neural Network

## Dataset:
The Car Evaluation dataset, sourced from the UCI Machine Learning Repository, contains evaluations of cars based on different attributes. The goal is to classify cars into categories such as acceptable, good, very good, or unacceptable. The dataset consists of categorical features such as buying price, maintenance cost, number of doors, number of persons the car can carry, boot size, and safety level. The target variable is the class of the car, which can be one of four classes: unacc, acc, good, and vgood.

Here are the attributes of the dataset:

- Buying: The buying price of the car (vhigh, high, med, low).
- Maint: The maintenance cost of the car (vhigh, high, med, low).
- Doors: Number of doors in the car (2, 3, 4, 5more).
- Persons: Number of persons the car can carry (2, 4, more).
- Lug_boot: Size of the luggage boot (small, med, big).
- Safety: The safety level of the car (low, med, high).
- Class: The target class of the car (unacc, acc, good, vgood).

## Code Walkthrough:
This project implements a Hebbian Neural Network to classify cars based on the features in the Car Evaluation dataset. The following steps are followed:

1. Importing Required Libraries:
- We import numpy for numerical operations, pandas for data manipulation, and several functions from sklearn for preprocessing the data.

2. Activation and Loss Functions:
- The sigmoid activation function is used to map the output to a range between 0 and 1.
- The mean squared error (MSE) loss function computes the average of the squared differences between the true and predicted target values.

3. Hebbian Neural Network Class:
- The HebbianNN class defines a simple neural network using Hebbian learning. This network performs forward propagation using the sigmoid activation function and updates weights based on the Hebbian learning rule, which strengthens the connection between neurons that activate simultaneously.
- The train() method iterates through multiple epochs, updating the weights and printing the loss value at each epoch.

4. Data Preprocessing:
- One-hot encoding is used to convert the categorical features and target variable into numerical format suitable for the neural network.
- Standardization is performed on the features to ensure they have a mean of 0 and standard deviation of 1, helping the network learn more effectively.

5. Training the Neural Network:
- The neural network is trained for 15 epochs with a learning rate of 0.01. The loss is printed at each epoch to monitor the model's performance.

## Key Highlights:
1. Dataset: The Car Evaluation dataset contains categorical attributes like buying, maint, and safety which describe the features of a car, and the target is the class of the car.
2. Neural Network: A Hebbian Neural Network is used for this project, which follows the Hebbian learning rule where the connection between neurons is strengthened when they activate simultaneously.
3. Data Preprocessing:
- One-hot encoding is used for converting categorical data into numerical format.
- Standardization of features ensures the model converges faster and more effectively.
4. Model Evaluation: The loss function, based on Mean Squared Error (MSE), is used to evaluate the model’s performance.

## Interpretation of the Output:
The output of the Hebbian Neural Network training process is a series of loss values printed for each epoch. These loss values indicate how well the model is performing during training, with the goal of minimizing the loss over time. In this case, the loss function being used is the Mean Squared Error (MSE), which measures the difference between the predicted and actual target values.

### Key Observations:
1. Initial Loss (Epoch 1):
- The initial loss is 0.2501, indicating that the model starts with a relatively high error. This is expected in the first epoch as the model is initializing the weights randomly.
2. Loss Increase in Early Epochs:
- From Epoch 1 to Epoch 4, the loss increases significantly from 0.2501 to 0.4631. This could indicate that the weight updates are not yet optimally aligned with the data, or that the network is struggling to learn meaningful patterns early on.
- This could also happen due to high learning rates, which may lead to overshooting during weight updates.
3. Slow Convergence (Epochs 5-8):
- After Epoch 4, the loss stabilizes somewhat but remains relatively high, fluctuating between 0.4808 and 0.4879. This suggests that the network might not be learning efficiently, possibly due to the limitations of the Hebbian learning rule or the chosen learning rate.
4. Convergence and Stability (Epochs 9-15):
- From Epoch 9 onward, the loss seems to stabilize and decrease very slowly, ending at 0.4839 at Epoch 15. This could suggest that the model has reached a local minimum, and further training might not improve performance significantly. The model is struggling to reduce the loss further, likely due to the simplicity of the Hebbian learning rule and its inability to perform backpropagation-based weight adjustments.

## Possible Causes for Loss Behavior:
1. Learning Rate:
- The learning rate of 0.01 might be too high, especially given that the Hebbian learning rule is relatively simple and does not involve any gradient descent mechanism like other neural networks. This could cause the network to overshoot and prevent it from converging to a lower loss.
2. Nature of Hebbian Learning:
- Hebbian learning relies on the correlation between neuron activations to update the weights. This can lead to inefficient learning when the dataset has complex patterns. The model may not be able to capture more intricate relationships compared to more advanced learning algorithms like backpropagation.
3. Epoch Limitation:
- The number of epochs is limited to 15, which might not be enough for the model to converge. Increasing the epochs could allow the model to potentially reduce the loss further, although it might still be constrained by the learning rule.
4. Data Complexity:
- The dataset itself is categorical and might require more complex transformations and learning techniques. The Hebbian learning rule, being unsupervised and simple, may not be the best fit for this type of problem, especially for classification tasks with categorical data.

## Recommendations for Improvement:
1. Adjust Learning Rate:
- Try experimenting with a smaller learning rate (e.g., 0.001) to see if the model converges more smoothly and reduces the loss more effectively.
2. Increase Epochs:
- Increasing the number of epochs might allow the model to learn more about the data, although it may still face challenges due to the limitations of Hebbian learning.
3. Change Learning Algorithm:
- Consider using a more advanced neural network with backpropagation, such as a feed-forward neural network (FFNN) with gradient descent. This would provide better weight updates and more efficient learning, especially for classification tasks.
4. Evaluate Feature Engineering:
- Investigate further preprocessing of the dataset, such as using different encoding techniques or scaling methods, which might help the network learn better.

## Conclusion:
The model’s loss decreases slowly over epochs, showing some learning but not converging to a satisfactory level. This slow convergence is likely due to the inherent limitations of the Hebbian learning rule, the chosen learning rate, and the complexity of the dataset. Experimenting with different learning rates, increasing epochs, or transitioning to a more advanced learning algorithm may yield better results.
