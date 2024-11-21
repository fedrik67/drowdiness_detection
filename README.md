# drowsiness_detection
# Driver State Detection

This project aims to detect the driver's state (eye state and yawning) using a Convolutional Neural Network (CNN) trained on the Yawn-Eye Dataset.

## Dataset

The project uses the "Yawn-Eye Dataset-New" available on Kaggle. You can download it from [here](kagglehub.dataset_download('serenaraju/yawn-eye-dataset-new')). The dataset contains images categorized into four classes:

- Open (eyes open)
- Closed (eyes closed)
- no_yawn (no yawning)
- yawn (yawning)

## Model

The project utilizes a CNN model built using Keras. The model architecture includes:

- Convolutional layers with increasing filter sizes and ReLU activation.
- Max pooling layers for downsampling.
- Flatten layer to convert feature maps into a vector.
- Dense layers for classification with ReLU activation.
- Output layer with softmax activation for multi-class classification.

## Usage

1.  **Import necessary libraries:**
2.  **Download the dataset:**
3.  **Preprocess the data:**
    -   Resize images to a consistent size.
    -   Organize images into separate lists based on eye state and yawning.
    -   Split data into training, validation, and testing sets.
4.  **Train the model:**
    -   Define the CNN model architecture.
    -   Compile the model with an optimizer, loss function, and metrics.
    -   Train the model using the training and validation data.
5.  **Evaluate the model:**
    -   Evaluate the model's performance on the test data.
    -   Visualize the results using metrics and plots.
6.  **Save the model:**
    -   Save the trained model for future use.


## Results

The model achieved an accuracy of approximately 95% on the test data. The confusion matrix and classification report provide detailed insights into the model's performance for each class.

## Conclusion

This project demonstrates the effectiveness of using CNNs for driver state detection. The trained model can be used to identify drowsy or distracted drivers, potentially contributing to safer driving conditions.

## Downloads
A copy of trained model is provided in the google drive adddress mentioned in links.txt. Here you will get one more file with alarm for detection. Both of them can be used in 'mediapipe_detection.py'
