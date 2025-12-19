import pandas as pd
import tensorflow as tf

# load data
data = pd.read_csv("synthetic_data/crush_dataset.csv")

X = data.drop("likes_you", axis=1)
y = data["likes_you"]

# simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(4,activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# train
model.fit(X, y, epochs=100, batch_size=16, verbose=1)

# save model
model.save("crush_model.keras")

print("Model saved as 'crush_model'")
