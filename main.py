import pandas as pd

data = pd.read_csv("Dataset\Text\sentences.csv", names=['lang', 'text'])

print(data)