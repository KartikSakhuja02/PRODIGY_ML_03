Task 01 - Implement a Support Vector Machine (SVM) model to classify images of cats and dogs using a given dataset.
Given Dataset: https://www.kaggle.com/c/dogs-vs-cats/data

The Python file named task03.py does all the tasks mentioned below.

To tackle the task of classifying images of cats and dogs using a Support Vector Machine (SVM) model, I followed a systematic approach. Here’s a step-by-step overview of how I completed this task:

1. Data Loading: I began by loading the training dataset from the specified directory using Python's os module. The dataset contained images labeled as either "cat" or "dog."

2. Feature and Target Definition: I specified the features for the model as the pixel values of the images. The target variable I aimed to predict was the class label: 0 for cats and 1 for dogs.

3. Data Preparation: I processed the images by resizing them to a consistent size (50x50 pixels), converting them to grayscale, flattening them into one-dimensional arrays, and normalizing the pixel values to a range between 0 and 1.

4. Data Splitting: To evaluate the model’s performance, I split the dataset into training and testing sets using an 80-20 split. This allowed me to train the model on one subset of the data and test its performance on another.

5. Feature Reduction: To improve the model's performance and reduce computational complexity, I applied Principal Component Analysis (PCA) to the training data, reducing the number of features to 100.

6. Model Training: I initialized an SVM model with a linear kernel and trained it on the PCA-transformed training data. The model learned the relationship between the features and the target variable.

7. Making Predictions: After training, I used the model to make predictions on the test dataset. This allowed me to assess how well the model performed.

8. Model Evaluation: I evaluated the model’s performance by calculating the classification report and confusion matrix. These metrics provided insights into the model's accuracy, precision, recall, and F1-score.

9. Visualization: To visualize the model’s performance, I plotted a confusion matrix using Seaborn. This helped in understanding the accuracy of the predictions visually.

By following these steps, I successfully implemented a Support Vector Machine (SVM) model to classify images of cats and dogs. The classification report shows the model's performance metrics, indicating how well the model was trained and evaluated.

