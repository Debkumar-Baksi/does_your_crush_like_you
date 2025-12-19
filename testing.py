import tensorflow as tf
import numpy as np

# load trained model
model = tf.keras.models.load_model("crush_model.keras")

# test sample: [looks, height, confidence, talk_frequency, shared_interests]
test_person = np.array([[8, 175, 7, 6, 9]])

# predict
prob = model.predict(test_person)[0][0]

print(f"Probability your crush likes you: {prob:.2f}")

# simple decision
if prob >= 0.5:
    print("Result: Likes you ❤️")
else:
    print("Result: Doesn't like you 😔")
