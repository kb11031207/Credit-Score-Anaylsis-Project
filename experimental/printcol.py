import pandas as pd


cleanedData = pd.read_csv('train_transformed.csv')
cleanedData.info()
#print(cleanedData.head())
print(cleanedData.head(20))


