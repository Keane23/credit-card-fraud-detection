import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

path = Path(__file__).resolve().parent.joinpath('dataset')
dataset = pd.read_csv(path.joinpath(Path('credit.csv')))

print(dataset.head())
print(dataset.info())