# Imagenette Image Classifier using EfficientNet-B0

## Project Description
This project implements an image classification web application using Streamlit and a pre-trained EfficientNet-B0 model. The model is trained on the Imagenette dataset, a smaller subset of ImageNet, designed for faster experimentation in deep learning. Users can upload an image, and the application will predict its class and display confidence scores for all possible classes.

## Features
*   **Image Upload:** Easily upload images in JPG, PNG, or JPEG formats.
*   **Real-time Prediction:** Get instant class predictions and confidence scores.
*   **EfficientNet-B0 Model:** Utilizes a powerful and efficient convolutional neural network for accurate classification.
*   **Interactive UI:** Built with Streamlit for a user-friendly web interface.

## Setup and Installation

### 1. Clone the Repository (if applicable)
```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### 2. Create a Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Download the Pre-trained Model Weights
Ensure you have the `best_model.pth` file in the root directory of your project. This file contains the trained weights for the EfficientNet-B0 model. If you don't have it, you would typically train the model yourself or download it from a specified source.

## How to Run the Application

1.  **Activate your virtual environment** (if you haven't already):
    ```bash
    source venv/bin/activate
    ```
2.  **Navigate to the project directory**:
    ```bash
    cd /Users/neharajbanshi/Desktop/project
    ```
3.  **Run the Streamlit application**:
    ```bash
    streamlit run GUI.py
    ```

    This command will open the application in your default web browser at `http://localhost:8501`.

## Model Details
The application uses `efficientnet_b0` from the `timm` library, pre-trained on a large dataset and fine-tuned for the Imagenette dataset (implied by the class labels). EfficientNet models are known for their efficiency and accuracy, scaling up CNNs in a more principled way.

## Class Labels
The model is trained to classify images into the following 10 categories:
*   0: 'tench'
*   1: 'English springer'
*   2: 'cassette player'
*   3: 'chain saw'
*   4: 'church'
*   5: 'French horn'
*   6: 'garbage truck'
*   7: 'gas pump'
*   8: 'golf ball'
*   9: 'parachute'

## Future Improvements
*   **More Robust Error Handling:** Implement more specific error messages for invalid file types or model loading issues.
*   **Support for More Models:** Allow users to select different pre-trained models.
*   **Dataset Information:** Provide more details about the Imagenette dataset within the application.
*   **Deployment:** Instructions for deploying the application to cloud platforms like Hugging Face Spaces, AWS, or Google Cloud.
*   **Interactive Model Explanation:** Integrate tools like Grad-CAM to visualize model predictions.