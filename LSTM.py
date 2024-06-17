import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Load and prepare data
with open('shifted_datasets_25_years.pkl', 'rb') as file:
    shifted_datasets = pickle.load(file)

data = shifted_datasets[-30]
data = data.sort_index()
data['Final_Target'] = data['Final_Target'].astype(int)
feature_names = data.drop(['Target', 'Quarter', 'Final_Target', 'Ticker'], axis=1).columns.tolist()

train_end = int(len(data) * 0.7)  
validation_start = train_end  
validation_end = len(data)  
scaler = MinMaxScaler()

scaler.fit(data.iloc[:train_end][feature_names])

data.loc[:train_end, feature_names] = scaler.transform(data.iloc[:train_end][feature_names])
data.loc[validation_start:validation_end, feature_names] = scaler.transform(data.iloc[validation_start:validation_end][feature_names])

# Compute class weights for handling class imbalance in the training set
class_weights = compute_class_weight('balanced', classes=np.unique(data['Final_Target'][:train_end]), y=data['Final_Target'][:train_end])
class_weight_dict = dict(enumerate(class_weights))

# Batch generator as defined
def batch_generator(feature_names,dataset, target, window, batch_size=4):
    total_size = len(dataset) - window
    feature_columns = feature_names  # Identify feature columns dynamically
    while True:
        for offset in range(0, total_size, batch_size):
            X, y = [], []
            for i in range(offset, min(offset + batch_size, total_size)):
                idx = i + window - 1  
                if idx < len(target):  
                    X.append(dataset.iloc[i:idx+1][feature_columns].values)  
                    y.append(target.iloc[idx])  
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int32)  
            yield X, y

# Create generators for training and validation
train_gen = batch_generator(feature_names,data.iloc[:train_end], data['Final_Target'][:train_end], window=120, batch_size=128)
val_gen = batch_generator(feature_names,data.iloc[validation_start:validation_end], data['Final_Target'][validation_start:validation_end], window=21, batch_size=128)

# Define and com
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Compile the model
num_features = len(feature_names)  

model = Sequential([
    # Convolutional Layer
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(120, num_features), padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    LSTM(64, return_sequences=True, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),

    LSTM(64, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),

    Dense(20, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', f1_m]
)

early_stopping = EarlyStopping(
    monitor='val_f1_m',
    mode='max',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_f1_m',
    mode='max',
    save_best_only=True,
    save_weights_only=True
)

model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch=(train_end - 120) // 128, 
    validation_data=val_gen,
    validation_steps=((validation_end - validation_start - 120) // 128), 
    epochs=10,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)