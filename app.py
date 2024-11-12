import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your pre-trained LSTM model
model = load_model('lstm_model.h5')  

# Function to create sequences
def make_sequence(df, sequence_length):
    X = [] 
    for i in range(len(df) - sequence_length):
        s = df.iloc[i : i + sequence_length].values
        X.append(s)
    return np.array(X)

# Streamlit app layout
st.title("Temperature Prediction with LSTM")
st.write("This app uses an LSTM model to predict temperature based on historical data.")

# Upload test data
uploaded_file = st.file_uploader("Choose a CSV file for testing", type="csv")
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Test Data:")
    st.dataframe(test_df)

    # Create sequences from the uploaded test data
    seq_length = 28  # Adjust as necessary
    ds_test = make_sequence(test_df, seq_length)

    # Make predictions
    ds_test = np.array(ds_test, dtype=np.float32)
    predictions = model.predict(ds_test)

    # Display predictions
    st.write("Predictions:")
    st.dataframe(predictions)

    # Optionally visualize the predictions
    if st.checkbox("Show Prediction Plot"):
        plt.figure(figsize=(12, 6))
        plt.plot(predictions, label='Predicted Temperature', color='orange')
        plt.title('Predicted Temperature Over Time')
        plt.xlabel('Samples')
        plt.ylabel('Mean Temperature')
        plt.legend()
        st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    st.write("Upload your CSV file to see the predictions.")
