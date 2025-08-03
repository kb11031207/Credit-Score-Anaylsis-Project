# credit_score_analysis.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

# Load the data from train.csv and test.csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Define the DataPreprocessing transformer
class DataPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        pass
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # Set feature number
        self.n_features = X.shape[1]
        self.features = X.columns.tolist()
        
        return self

        
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        print('made to data preprocessing')
        # Set feature number
        assert X.shape[1] == self.n_features
        check_is_fitted(self, 'n_features')
        # Drop columns
        X['Month'] = X['Month'].astype('category').cat.codes
        X['Month_sin'] = np.sin(2 * np.pi * X['Month'] + 1 / 12)
        X['Month_cos'] = np.cos(2 * np.pi * X['Month'] + 1 / 12)
        X.drop(columns=['Month'], inplace=True)
        X['Credit_Mix'] = X['Credit_Mix'].astype('category').cat.codes
        X.dropna(subset=['Monthly_Inhand_Salary', 'Num_of_Delayed_Payment', 'Amount_invested_monthly', 'Monthly_Balance'], inplace=True)
        #lets payyment  min amount to dummies encoding
        if 'Credit_Score' in X.columns:
             X['Credit_Score'] = X['Credit_Score'].astype('category').cat.codes 
        X['Payment_of_Min_Amount'] =  X['Payment_of_Min_Amount'].astype('category').cat.codes 
        #apply the one hot encoding on the occupation column using the onehot encoder object
        X = pd.concat([X, pd.DataFrame(self.onehot.fit_transform(X[['Occupation']]), columns=self.onehot.get_feature_names_out())], axis=1)
        #X.drop(columns = ['Occupation'], inplace = True)




       
        print(X.info()  )

        return X

# Define the NumericCleaner transformer
class NumericCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column] = pd.to_numeric(X[column].str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
        return X

# Define the CreditHistoryTransformer
class CreditHistoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
      


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].astype(str)
        years = X[self.column].str.extract(r'(\d+)\s*Years?')[0].fillna(0).astype(int)
        months = X[self.column].str.extract(r'(\d+)\s*Months?')[0].fillna(0).astype(int)
        X[self.column] = years * 12 + months
        return X

# Define the PaymentBehaviourTransformer
class PaymentBehaviourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.charge_mapping = {'Low_spent': 0, 'Medium_spent': 1, 'High_spent': 2}
        self.payment_mapping = {'Small_value_payments': 0, 'Medium_value_payments': 1, 'Large_value_payments': 2}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        charge, payment = X[self.column].apply(self.split_payment_behavior)
        X['Charge'] = charge
        X['Payment'] = payment
        X.drop(self.column, axis=1, inplace=True)
        return X

    def split_payment_behavior(self, value):
        if isinstance(value, str):
            parts = value.split('_')
            charge_part = '_'.join(parts[:2])
            payment_part = '_'.join(parts[2:])
            charge_value = self.charge_mapping.get(charge_part, np.nan)
            payment_value = self.payment_mapping.get(payment_part, np.nan)
            return pd.Series([charge_value, payment_value])
        else:
            return pd.Series([np.nan, np.nan])

# Define the LoanType transformer
class LoanType(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.ml_binarizer = MultiLabelBinarizer()

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame) and self.column in X.columns
        assert 'Customer_ID' in X.columns 
        loan_types = X[self.column].fillna('Not Specified').astype(str).str.split(', ')
        loan_types = loan_types.apply(lambda x: [item.strip() for item in x if 'and' not in item])
        self.ml_binarizer.fit(loan_types)
        return self

    def transform(self, X):
        check_is_fitted(self, 'ml_binarizer')
        if 'Customer_ID' not in X.columns:
            raise KeyError("Column 'Customer_ID' not found in DataFrame.")
        loan_types = X[self.column].fillna('Not Specified').astype(str).str.split(', ')
        loan_types = loan_types.apply(lambda x: [item.strip() for item in x if 'and' not in item])
        loan_type_dummies = self.ml_binarizer.transform(loan_types)
        loan_type_df = pd.DataFrame(loan_type_dummies, columns=self.ml_binarizer.classes_, index=X.index)
        X = pd.concat([X, loan_type_df], axis=1)
        X.drop(self.column, axis=1, inplace=True)
        return X

# Define the HiLoImputer transformer
class HiLoImputer(BaseEstimator, TransformerMixin):
   def __init__(self, lo, hi, columnName):
      self.hi = hi
      self.lo = lo
      self.columnName = columnName

      self.isCategory = False

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame) 
      assert self.columnName in X.columns
      
      # Check if the column is categorical
      self.isCategory = X[self.columnName].dtype.name == 'object' or X[self.columnName].dtype.name == 'category'
      if self.isCategory:
         #convert to category

         self.categories_ = X[self.columnName].astype('category').cat.categories

      # Store the number of features
      self._n_features = X.shape[1]
      return self

   def transform(self, X):
      check_is_fitted(self, '_n_features')
      assert isinstance(X, pd.DataFrame) and X.shape[1] == self._n_features
      print
      if self.isCategory:
        
         # Convert categories to codes for hi/lo processing
         X[self.columnName] = X[self.columnName].astype('category').cat.codes

         # Apply hi/lo bounds, setting out-of-bounds to NaN
         X[self.columnName] = np.where(
               (X[self.columnName] > self.hi) | (X[self.columnName] < self.lo), 
               np.nan, X[self.columnName]
         )
     
         
         # Replace NaN values with the mode per Customer_ID group
         X[self.columnName] = X.groupby('Customer_ID', group_keys=False)[self.columnName].apply(
               lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)
         )

         # Convert back to categories
         X[self.columnName] = pd.Categorical.from_codes(
               X[self.columnName].fillna(-1).astype(int), categories=self.categories_, ordered=True
         ).remove_unused_categories()
         X[self.columnName] = X[self.columnName].astype('category')

      else:
         # Apply hi/lo bounds for numerical data
         X[self.columnName] = np.where(
               (X[self.columnName] > self.hi) | (X[self.columnName] < self.lo), 
               np.nan, X[self.columnName]
         )

         # Replace NaN values with the mode per Customer_ID group
         X[self.columnName] = X.groupby('Customer_ID', group_keys=False)[self.columnName].apply(
               lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)
         )

      return X
class HiLoImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, hi_lo_bounds):
        self.hi_lo_bounds = hi_lo_bounds
        self.imputers = {}

    def fit(self, X, y=None):
        for column, bounds in self.hi_lo_bounds.items():
            lo = bounds.get('lo', float('-inf'))
            hi = bounds.get('hi', float('inf'))
            self.imputers[column] = HiLoImputer(lo=lo, hi=hi, columnName=column).fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        for column, imputer in self.imputers.items():
            X = imputer.transform(X)
        return X

# Define hi/lo bounds for numerical columns
hi_lo_bounds = {
    'Age': {'lo': 0, 'hi': 100},
    'Annual_Income': {'lo': 5000, 'hi': 500000},
    'Monthly_Inhand_Salary': {'lo': 500, 'hi': 30000},
    'Num_Bank_Accounts': {'lo': 0, 'hi': 20},
    'Num_Credit_Card': {'lo': 0, 'hi': 15},
    'Interest_Rate': {'lo': 0, 'hi': 50},
    'Delay_from_due_date': {'lo': 0, 'hi': 100},
    'Num_Credit_Inquiries': {'lo': 0, 'hi': 30},
    'Changed_Credit_Limit': {'lo': 0, 'hi': 30},
    'Num_of_Loan': {'lo': 0, 'hi': 10},
    'Credit_Mix': {'lo': 0, 'hi': 2},
    'Occupation': {'lo': 0, 'hi': 14}
}

# Create the HiLoImputer transformers for each column
# Define the ColumnTransformer for HiLoImputer with explicit columns
hilo_imputers = ColumnTransformer(
    transformers=[
        (f'hilo_imputer_{column}', HiLoImputer(lo=bounds['lo'], hi=bounds['hi'], columnName=column), [column])
        for column, bounds in hi_lo_bounds.items()

    ]
)
class DropNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # save feature name
        Xpd = pd.DataFrame(X)
        self.feature_names_in_ = Xpd.columns.tolist()
    
        return self

    def transform(self, X):
        # Drop rows where any of the specified columns have NaN values
        Xpd = pd.DataFrame(X)
        X_dropped = Xpd.dropna(subset=self.columns)
        return X_dropped
    def get_feature_names_out(self, input_features=None):
        # Return the feature names after transformation (all features are retained, just rows are dropped)
        # lets return self.features if input_feature is none but without self.columns
        # Return all feature names except those in self.columns
        if input_features is None:
            return [feature for feature in self.feature_names_in_ if feature not in self.columns]
        return [feature for feature in input_features if feature not in self.columns]

    
columns_with_na_to_drop = ['Monthly_Inhand_Salary', 'Num_of_Delayed_Payment', 'Amount_invested_monthly', 'Monthly_Balance', 'Occupation']


# lets create a column transformer that drops thesse columns ['Customer_ID', 'ID', 'Name', 'SSN']

# letts create a transformer that drops 


# Full Pipeline
pipeline = Pipeline([
    
    ('numeric_cleaner', NumericCleaner(columns=[
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance'
    ])),('hi_lo_imputer', HiLoImputerWrapper(hi_lo_bounds=hi_lo_bounds)),
    ('credit_history', CreditHistoryTransformer(column='Credit_History_Age')),
    ('payment_behaviour', PaymentBehaviourTransformer(column='Payment_Behaviour')),
    ('loan_type', LoanType(column='Type_of_Loan'))
    ,('data_preprocessing', DataPreprocessing()),('drop_na', DropNaTransformer(columns=columns_with_na_to_drop)) , ('column_transformer', ColumnTransformer(
        [ ('drop', 'drop', ['Customer_ID', 'ID', 'Name', 'SSN', 'Occupation'])],
        remainder='passthrough'
    ))
     
])
pipeline.fit(train)  
data = pipeline.transform(train)
print('here')

print('jhj')
type(data)
print(pipeline[-2].get_feature_names_out())
#convert to dataframe
data = pd.DataFrame(data, columns= pipeline[-2].get_feature_names_out())


X = data.drop(columns=['Credit_Score'])
y = data['Credit_Score']
print(X.describe())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('here again')

# train the model
model = RandomForestClassifier(random_state=42)
#model.fit(X_train, y_train)
import joblib

# Load the model from the file
model = joblib.load('credit_score_model.pkl')

# Now you can use the model to make predictions
# For example, if you have new data in a DataFrame called `new_data`:
# predictions = model.predict(new_data)
# Predict on the test set

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2%}')


data = pipeline.fit_transform(test)
data = pd.DataFrame(data, columns= pipeline[-2].get_feature_names_out())

X = data

print(data.info())
# Predict on the test set
y_pred = model.predict(data)
print(y_pred)







#acuracy on test data

