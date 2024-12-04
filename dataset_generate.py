import numpy as np
import pandas as pd

# Setting the seed for reproducibility
np.random.seed(42)

# Creating the numerical data
data = np.random.uniform(-1, 1, size=(1200, 40000))

# Creating the labels
labels = np.random.randint(0, 2, size=(1200, 1))

# Combining the data and labels
full_data = np.hstack((data, labels))

# Creating the DataFrame
columns = [f'feature_{i}' for i in range(1, 40001)] + ['label']
df = pd.DataFrame(full_data, columns=columns)
df.to_csv("data/data.csv")
# Displaying the first few rows of the DataFrame
df.head()
