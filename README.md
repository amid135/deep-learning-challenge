Deep Learning Challenge: Alphabet Soup

Dataset
The dataset contains over 34,000 records of organizations that received funding, with features including:

- EIN and NAME: Identification columns (removed for analysis)
- APPLICATION_TYPE: Type of application submitted
- AFFILIATION: Sector of industry
- CLASSIFICATION: Government classification of the organization
- USE_CASE: Intended use of the funds
- ORGANIZATION: Type of organization
- STATUS: Active status of the organization
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Specific considerations for the application
- ASK_AMT: Amount of funding requested
- IS_SUCCESSFUL: Indicates effective use of funds

Project Structure
Preprocessing: Clean and prepare the dataset using Pandas and scikit-learn. This includes:
- Reading the CSV file into a DataFrame
- Identifying target and feature variables
- Dropping unnecessary columns
- Handling categorical variables using one-hot encoding
- Splitting the data into training and testing sets
- Scaling features with StandardScaler

Model Development: Design and evaluate a neural network using TensorFlow and Keras:
- Create the model architecture (input layer, hidden layers, output layer)
- Compile the model and fit it to the training data
- Evaluate model performance using accuracy metrics

Optimization: Enhance the model to achieve predictive accuracy above 75% by:
- Adjusting the number of neurons and layers
- Modifying activation functions
- Experimenting with different training parameters

Final Output: Save the trained model in both HDF5 and Keras formats for future use.
- Instructions to Run the Project

Setup:
- Create a new repository named deep-learning-challenge.
- Clone the repository to your local machine and create a directory for this project.
- Install Required Libraries: Make sure you have the necessary libraries installed:

- bash
- Copy code
- pip install pandas scikit-learn tensorflow

Run Jupyter Notebook:
- Open the provided Jupyter Notebook files and execute each cell in order after preprocessing the dataset.

Model Evaluation:
- Analyze the results and performance of the model, adjusting parameters as needed to optimize accuracy.
- Files Included
- charity_data.csv: The dataset used for analysis.
- AlphabetSoupCharity.ipynb: Jupyter Notebook for model development and evaluation.
- AlphabetSoupCharity_Optimization.ipynb: Jupyter Notebook for optimizing the model.
- AlphabetSoupCharity.h5: Saved model file in HDF5 format.
- AlphabetSoupCharity.keras: Saved model file in Keras format.


Report on the Neural Network Model

Data Preprocessing

Target Variable:
  - IS_SUCCESSFUL: Indicates whether the funding was used effectively.

Feature Variables:
  - APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.

Variables to Remove:
  - EIN and NAME were removed as they serve as identifiers and do not contribute to predictive capabilities.
  - Compiling, Training, and Evaluating the Model


Model Architecture:
  - Layers: The model consisted of 3 layers:
    - Input layer (number of input features)
    - Two hidden layers

Neurons: The first hidden layer had 128 neurons, and the second hidden layer had 64 neurons, chosen to balance model complexity and training efficiency.
Activation Functions:
  - ReLU (Rectified Linear Unit) for hidden layers to introduce non-linearity.
  - Sigmoid for the output layer to output a probability for binary classification.

Performance Achievement:
  - The model achieved an accuracy of over 75% during validation, meeting the target performance.

Steps to Increase Model Performance:
  - Adjusted the number of neurons and layers.
  - Modified the activation functions.
  - Increased training epochs and adjusted batch sizes.
