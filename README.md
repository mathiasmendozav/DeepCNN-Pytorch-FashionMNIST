# FashionMNIST CNN with PyTorch

![FashionMNIST Example](https://ik.imagekit.io/r67xuhpwk/Fashion-MNIST-0000000040-4a13281a_m8bp4wm.webp?updatedAt=1729017936709)

This repository contains an implementation of a Convolutional Neural Network (CNN) for the FashionMNIST dataset using PyTorch. FashionMNIST is a dataset of Zalando's article images—consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image associated with a label from 10 classes.

## Project Structure

```bash
fashion-mnist-cnn/
│
├── src/
│   ├── dataset.py       # Data loading and preprocessing
│   ├── model.py         # CNN architecture definition
│   ├── train.py         # Training loop and logic
│   ├── evaluate.py      # Model evaluation script
│   └── utils.py         # Utility functions for training and evaluation
│
├── tests/
│   └── test_model.py    # Unit tests for the model
│
├── requirements.txt     # Dependencies for the project
├── README.md            # This file
└── LICENSE              # Project license
```

## CNN Architecture

The CNN model defined in `model.py` employs the following structure:

- **Input Layer**: Takes in a 1x28x28 image (grayscale).
- **Convolutional Layers**:
  - Two convolutional layers with ReLU activation:
    - First Conv2D: 32 filters, kernel size 3x3, padding 1
    - Second Conv2D: 64 filters, kernel size 3x3, padding 1
  - Each followed by:
    - Max Pooling: kernel size 2, stride 2
    - Dropout: for regularization, with p=0.25
- **Fully Connected Layers**:
  - Flatten layer
  - Dense layer with 128 units and ReLU activation
  - Dropout with p=0.5
  - Output layer with 10 units (one for each class), using softmax activation for classification.

## Training

The training script `train.py` uses:

- **Optimizer**: Adam with learning rate = 0.01
- **Loss Function**: Cross Entropy Loss
- **Batch Size**: 64
- **Epochs**: Configurable, default set to 20
- **Device**: Automatic selection between CUDA (if available) or CPU

### Implementation Details:
- **Data Augmentation**: Random rotation up to 10 degrees, random horizontal flips to increase dataset diversity.
- **Learning Rate Scheduler**: StepLR with step_size=7, gamma=0.1 to decrease learning rate over time.
- **Early Stopping**: Implemented to prevent overfitting, patience set to 5 epochs.

## Evaluation

The `evaluate.py` script provides:

- **Metrics**: Accuracy, Confusion Matrix, Classification Report
- **Visualization**: Examples of misclassified images, ROC curves for multi-class classification if extended.

## Setup Instructions

1. **Clone this repository**:
   ```bash
   git clone https://github.com/mathiasmendozav/DeepCNN-Pytorch-FashionMNIST
   cd fashion-mnist-cnn
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the FashionMNIST dataset**:
   This is automatically handled by `src/dataset.py` on the first run.

4. **Train the model**:
   ```bash
   python src/train.py
   ```

5. **Evaluate the model**:
   ```bash
   python src/evaluate.py
   ```

## Advanced Features

- **TensorBoard Integration**: For real-time tracking of training metrics.
- **Model Checkpointing**: Saving the best model based on validation loss.
- **Hyperparameter Tuning**: Script included for tuning using grid search or random search over defined parameters.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request. Ensure your code passes the existing unit tests and adds new tests if introducing new functionality.

## License

This project is licensed under the MIT License.