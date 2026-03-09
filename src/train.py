import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers


# Load dataset
df = pd.read_csv('data/train.csv')

# Split features and labels
x = df.drop('label', axis=1)
y = df['label']

# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert to numpy
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0


# Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28,1)),

    layers.Conv2D(32, (3,3), use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()


# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# Predict
predictions = model.predict(x_test)

print("Predicted:", np.argmax(predictions[0]))
print("Actual:", y_test.iloc[0])


# Show image
plt.imshow(np.squeeze(x_test[0]), cmap='gray')
plt.title(y_test.iloc[0])
plt.show()