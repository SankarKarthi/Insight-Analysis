# Data Dumper

**Data Dumper** is a Streamlit-based web application for data analysis and machine learning. It allows users to upload a dataset, clean and preprocess the data, apply machine learning models, and extract important features for prediction. The app is designed for users with minimal coding experience, enabling them to analyze data easily and generate insights.

## Features
- **Data Preprocessing**: Upload datasets, check for null values, fill missing data, and perform basic data cleaning.
- **Data Visualization**: Display heatmaps for correlations, statistical summaries, and basic insights from the dataset.
- **Machine Learning**: Choose between Multiple Linear Regression, Random Forest Classifier, or Random Forest Regressor to train and evaluate models.
- **Feature Extraction**: Use Random Forest Regressor to determine feature importance in your dataset.
- **Download Processed Data**: After data preprocessing, users can download the cleaned dataset as a CSV file.
- **Model Saving**: Trained models are saved as `.pkl` files for future use.

## Requirements
To run the app locally, you'll need to install the following dependencies:

- Python 3.8 or higher
- Streamlit
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Joblib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/data-dumper.git
    cd data-dumper
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

### Usage

1. Open your browser and go to `http://localhost:8501/`.
2. Select a demo from the sidebar:
   - **Main**: Introduction to the app and contact information.
   - **Data PreProcessing**: Upload and clean your dataset.
   - **Regression and Prediction**: Apply machine learning models.
   - **Feature Extraction**: Analyze feature importance.
3. Upload a CSV file to start analyzing data.
4. View the output and download the cleaned dataset or model files.
