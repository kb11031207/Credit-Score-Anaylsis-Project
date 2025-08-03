"""
************************************************
* dataprocessing.py
* Kesiena Berezi
* Credit Score Analysis Project
* Description: Data preprocessing pipeline for credit score prediction.
*              Implements custom transformers for cleaning and transforming
*              financial data including numeric cleaning, credit history
*              transformation, payment behavior analysis, and outlier detection.
* Usage: python dataprocessing.py
************************************************
"""

import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

class NumericCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_clean =  ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                                  'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        print('Cleaning numeric columns')
        # Make a copy of the DataFrame to avoid modifying the original data
        X = X.copy()
        
        # Apply cleaning function to each specified column
        for column in self.columns_to_clean:
            X[column] = pd.to_numeric(X[column].str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
        
        return X

class CreditHistoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column = 'Credit_History_Age'

    def fit(self, X, y=None): 
        assert isinstance(X, pd.DataFrame) and self.column in X.columns
        # Set number of features
        self._n_features = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, '_n_features')
        assert isinstance(X, pd.DataFrame) and self.column in X.columns   
        print('Transforming Credit_History_Age')
        # Convert the verbal description to numerical values
        X[self.column] = X[self.column].astype(str)
        years = X[self.column].str.extract(r'(\d+)\s*Years?')[0].fillna(0).astype(int)
        months = X[self.column].str.extract(r'(\d+)\s*Months?')[0].fillna(0).astype(int)
        # Calculate total months
        X[self.column] = years * 12 + months
        return X

class PaymentBehaviourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.charge_mapping = {'Low_spent': 0, 'Medium_spent': 1, 'High_spent': 2}
        self.payment_mapping = {'Small_value_payments': 0, 'Medium_value_payments': 1, 'Large_value_payments': 2}
        self.columns = 'Payment_Behaviour'

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        'Payment_Behaviour' in X.columns
        return self

    def transform(self, X):
        charge, payment = X['Payment_Behaviour'].apply(self.split_payment_behavior)
        X['Charge'] = charge
        X['Payment'] = payment
        X.drop('Payment_Behaviour', axis=1, inplace=True)
        print('Transforming Payment_Behaviour')
        return X

    def split_payment_behavior(self, value):
        if isinstance(value, str):
            # Split the value into components based on the underscore
            parts = value.split('_')
            # The first part corresponds to charge, and the last two parts correspond to payment
            charge_part = '_'.join(parts[:2])  # Join the first two parts for charge
            payment_part = '_'.join(parts[2:])  # Join the rest for payment
            # Get the corresponding numerical values from the mappings
            charge_value = self.charge_mapping.get(charge_part, np.nan)
            payment_value = self.payment_mapping.get(payment_part, np.nan)
            return pd.Series([charge_value, payment_value])
        else:
            return pd.Series([np.nan, np.nan])  # Return NaN for non-string values

class LoanType(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ml_binarizer = MultiLabelBinarizer()

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame) and 'Type_of_Loan' in X.columns
        # Ensure the column is treated as a string and handle NaN values
        loan_types = X['Type_of_Loan'].fillna('Not Specified').astype(str).str.split(', ')
        # Clean up the loan types to remove unwanted phrases
        loan_types = loan_types.apply(lambda x: [item.strip() for item in x if 'and' not in item])
        self.ml_binarizer.fit(loan_types)      
        self._n_features = len(self.ml_binarizer.classes_)
        return self 

    def transform(self, X):
        # Check if the transformer has been fitted
        print('Transforming Loan Type')
        check_is_fitted(self, '_n_features')
        assert isinstance(X, pd.DataFrame) and 'Type_of_Loan' in X.columns
        # Ensure the column is treated as a string and handle NaN values
        loan_types = X['Type_of_Loan'].fillna('Not Specified').astype(str).str.split(', ')
        # Clean up the loan types to remove unwanted phrases
        loan_types = loan_types.apply(lambda x: [item.strip() for item in x if 'and' not in item])
        loan_type_dummies = self.ml_binarizer.transform(loan_types)
        # Create a DataFrame for the binary columns
        loan_type_df = pd.DataFrame(loan_type_dummies, columns=self.ml_binarizer.classes_, index=X.index)
        # Concatenate the new binary columns with the original DataFrame
        X = pd.concat([X, loan_type_df], axis=1)
        X.drop('Type_of_Loan', axis=1, inplace=True)
        return X

class toCategoryCodes(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings=None):
        self.ordinal_mappings = ordinal_mappings or {}

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame) and len(X.columns) > 1
        # select only credit score and credit mix as categorical column
        self

        self.columnName = X.select_dtypes(include=['object', 'category']).columns
        
        self._n_features = X.shape[1]
        return self


    # ... existing code ...

    def transform(self, X):
        check_is_fitted(self, '_n_features')
        assert isinstance(X, pd.DataFrame) and X.shape[1] == self._n_features
        print('Converting to category')
        X = X.copy()
        for column in self.columnName:
            if column in self.ordinal_mappings:
                # Print current categories for debugging
                print(f'Current categories for {column}: {X[column].unique()}')
                # Apply explicit ordinal mapping
                X[column] = X[column].astype('category')
                # Check if all categories in the mapping are present in the column
                missing_categories = set(self.ordinal_mappings[column]) - set(X[column].cat.categories)
                if missing_categories:
                    print(f'Missing categories for {column}: {missing_categories}')
                X[column] = X[column].cat.reorder_categories(
                    self.ordinal_mappings[column], 
                    ordered=True
                ).cat.codes
        
            #print the categpories 
            print(f'New categories for {column}: {X[column].unique()}')
            #print categories mapping 
           
            #print the numerical 

        return X
    

class HiLoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bounds):
        self.bounds = bounds 

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self._n_features = X.shape[1]
        self._feature_names = X.columns
        return self

    def transform(self, X):
        check_is_fitted(self, '_n_features')
        assert isinstance(X, pd.DataFrame) and X.shape[1] == self._n_features
        print('Transforming HiLo values')
        #time for each column
        start1 = time.time()
        for column, bounds in self.bounds.items():
            #time taken for each column
            start = time.time()
            isCategory = X[column].dtype.name == 'category' or X[column].dtype.name == 'object'

            if isCategory:
                X[column] = X[column].astype('category')
                categories = X[column].cat.categories
                X[column] = X[column].astype('category').cat.codes

            X[column] = np.where(
                (X[column] > bounds['hi']) | (X[column] < bounds['lo']), 
                np.nan, X[column]
            )

            # Use mode to fill NaN values
            X[column] = X.groupby('Customer_ID', group_keys=False)[column].apply(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)
            )
            # X[column].fillna(mode_value)

            # Convert back to categorical if necessary
            if isCategory:
                X[column] = pd.Categorical.from_codes(
                    X[column].fillna(-1).astype(int), categories=categories, ordered=True
                ).remove_unused_categories()
                X[column] = X[column].astype('category')
            
              #  print(X[column].value_counts())
            print(f'{column} took {time.time() - start} seconds')
        print(f'Total time taken: {time.time() - start1} seconds')

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self._feature_names
        return input_features

from sklearn.preprocessing import OneHotEncoder
class OccupationOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        print('OccupationOneHotEncoder')

    def fit(self, X, y=None):
        # Handle missing data by filling it with a placeholder
        self.onehot.fit(X[['Occupation', 'Month']])  # Fit on both columns
        self.feature_names_out = self.onehot.get_feature_names_out(['Occupation', 'Month'])
        return self

    def transform(self, X):
        print('Here')
        encoded = self.onehot.transform(X[['Occupation', 'Month']])  # Transform both columns
        print(f'Shape of encoded data: {encoded.shape}')  # Debugging line
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_out, index=X.index)
        X = X.reset_index(drop=True)
        return pd.concat([X, encoded_df], axis=1).drop(columns=['Occupation', 'Month'])  # Drop both columns

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_out
        return input_features

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_out_ = [col for col in X.columns if col not in self.columns]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            # Use default column names if X is a numpy array
            print('Transforming DropColumns')
            X = pd.DataFrame(X)
        X.drop(columns=self.columns, inplace=True, errors='ignore')
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

hi_lo_bounds = {
    # Numerical Columns
    'Age': {'lo': 0, 'hi': 100},
    'Annual_Income': {'lo': 5000, 'hi': 500000},
    'Monthly_Inhand_Salary': {'lo': 500, 'hi': 30000},
    'Num_Bank_Accounts': {'lo': 0, 'hi': 20},
    'Num_Credit_Card': {'lo': 0, 'hi': 15},
    'Interest_Rate': {'lo': 0, 'hi': 50},
    'Delay_from_due_date': {'lo': 0, 'hi': 100},
    'Num_Credit_Inquiries': {'lo': 0, 'hi': 30},
    'Changed_Credit_Limit': {'lo': 0, 'hi': 30},  # Bounds after converting to numeric
    'Num_of_Loan': {'lo': 0, 'hi': 10},  # Bounds after converting to numeric
    'Num_of_Delayed_Payment': {'lo': 0, 'hi': 100},  # Bounds after converting to numeric

    
    # Categorical Columns (ordered)
    'Credit_Mix': {'lo': 0, 'hi': 2},  # Assuming "Bad" = 0, "Standard" = 1, "Good" = 2
    # Categorical Column (nominal)
    'Occupation': {'lo': 0, 'hi': 14}  # No bounds, only impute missing values
}

ordinal_mapping = {
    'Credit_Score': ['Good', 'Standard', 'Poor'],
    
    'Credit_Mix': ['Bad', 'Standard', 'Good'],
    'Payment_of_Min_Amount' : ['No', 'NM','Yes'],
}
# Transformer to drop NaN rows
class DropNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('Dropping Nan rows')
        return X.dropna()

class MonthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder( handle_unknown='ignore')

    def fit(self, X, y=None):
        self.encoder.fit(X[['Month']])
        self.feature_names_out = self.encoder.get_feature_names_out(['Month'])
        return self

    def transform(self, X):
        one_hot = self.encoder.transform(X[['Month']])
        one_hot_df = pd.DataFrame(one_hot, columns=self.feature_names_out, index=X.index)
        X = pd.concat([X, one_hot_df], axis=1).drop(columns=['Month'])
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out
    
pipeline = Pipeline(steps= [
    ('NumericCleaner', NumericCleaner()),
    ('CreditHistoryTransformer', CreditHistoryTransformer()),
    ('PaymentBehaviourTransformer', PaymentBehaviourTransformer()),
    ('LoanType', LoanType()),
    ('HiLoTransformer', HiLoTransformer(bounds=hi_lo_bounds)), 
    ('DropColumnsTransformer', DropColumnsTransformer(columns=[ 'ID', 'Name', 'SSN'])),
    ('toCategoryCodes', toCategoryCodes(ordinal_mappings=ordinal_mapping)),
   # ('MonthTransformer', MonthTransformer()),
    
    ('OccupationOneHotEncoder', OccupationOneHotEncoder()),
    ('DropNanRows', DropNanRows())
])


# Fit the pipeline to the training data
pipeline.fit(train)
# Transform the training data
train_transformed = pipeline.transform(train)
train_transformed.info()
#save the transformed data
train_transformed.to_csv('train_transformed.csv', index=False)
#print first credit score 
print(train_transformed['Credit_Score'].head())
train_transformed.info()
