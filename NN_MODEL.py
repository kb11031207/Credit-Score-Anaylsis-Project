from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
import time

# Load the cleaned data
cleanedData = pd.read_csv('train_transformed.csv')
X = cleanedData.drop('Credit_Score', axis=1)
y = cleanedData['Credit_Score']

# Split the data into training and temporary sets
grouped_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_idx, temp_idx in grouped_split.split(X, y, groups=X['Customer_ID']):
    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]

# Split the temporary set into validation and test sets (50% each)
grouped_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in grouped_split.split(X_temp, y_temp, 
                                              groups=X_temp['Customer_ID']):
    X_val, X_test = X_temp.iloc[val_idx], X_temp.iloc[test_idx]
    y_val, y_test = y_temp.iloc[val_idx], y_temp.iloc[test_idx]

# Drop the Customer_ID column
X_train = X_train.drop('Customer_ID', axis=1)
X_val = X_val.drop('Customer_ID', axis=1)
X_test = X_test.drop('Customer_ID', axis=1)
X_temp = X_temp.drop('Customer_ID', axis=1)

# Step 2: Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_temp = scaler.transform(X_temp)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 3: Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

# Step 4: Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Implement Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Step 6: Train the model
start = time.time()
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping])

# Print time taken
print(f'Time taken: {time.time() - start} seconds')

# Step 7: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_accuracy)
#accuracy = model.evaluate(X_temp, y_temp)