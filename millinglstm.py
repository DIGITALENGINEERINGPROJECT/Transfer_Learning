import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess the dataset
data = pd.read_csv('milling_machine_data.csv')  # Replace 'milling_machine_data.csv' with your dataset file
X = data[['speed']]  # Assuming 'speed' is the input feature
y = data['power']    # Assuming 'power' is the target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the data into sequences for LSTM
sequence_length = 10  # Adjust the sequence length as needed
X_sequence = []
y_sequence = []
for i in range(len(X_scaled) - sequence_length):
    X_sequence.append(X_scaled[i:i+sequence_length])
    y_sequence.append(y[i+sequence_length])

X_sequence = np.array(X_sequence)
y_sequence = np.array(y_sequence)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequence, y_sequence, test_size=0.2, random_state=42)

# Define the LSTM neural network architecture
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks for early stopping and model checkpoint
callbacks = [
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('pretrained_milling_machine_model.keras', save_best_only=True)
]

# Train the neural network
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Evaluate the model on test data
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)
