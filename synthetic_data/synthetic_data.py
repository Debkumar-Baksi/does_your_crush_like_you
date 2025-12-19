import numpy as np
import pandas as pd

np.random.seed(42)

n = 500  # number of samples

data = pd.DataFrame({
    "looks": np.random.randint(1, 11, n),
    "height": np.random.randint(150, 195, n),
    "confidence": np.random.randint(1, 11, n),
    "talk_frequency": np.random.randint(1, 11, n),
    "shared_interests": np.random.randint(1, 11, n),
})

# simple scoring rule
score = (
    0.3 * data["looks"] +
    0.1 * (data["height"] - 150) / 45 +  # normalize height
    0.25 * data["confidence"] +
    0.2 * data["talk_frequency"] +
    0.25 * data["shared_interests"]
)

data["likes_you"] = (score > 5.5).astype(int)

print(data.head())
data.to_csv("crush_dataset.csv", index=False)
print("Dataset saved")