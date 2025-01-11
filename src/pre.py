from data_preprocessing import DataPreprocessor

# Define file paths
inference_logs_path = "data/inference_logs.csv"
processed_inference_path = "data/processed_inference_logs.csv"
preprocessor_load_path = "data/preprocessor.joblib"

# Initialize the DataPreprocessor without labels
preprocessor = DataPreprocessor(filepath=inference_logs_path, has_labels=False)

try:
    # Load the preprocessor
    preprocessor.load_preprocessor(path=preprocessor_load_path)

    # Load and preprocess the inference data
    preprocessor.preprocess()

    # Retrieve processed data
    X_inference, _, log_ids_inference = preprocessor.get_processed_data()

    # Combine processed features with LogID for traceability
    processed_inference_df = X_inference.copy()
    processed_inference_df['logid'] = log_ids_inference
    processed_inference_df.to_csv(processed_inference_path, index=False)
    print(f"\nProcessed inference data saved successfully at '{processed_inference_path}'.")
except Exception as e:
    print(f"\nAn error occurred during inference preprocessing: {e}")
