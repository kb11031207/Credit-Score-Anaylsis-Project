import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
    def __init__(self):
        pass

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame) and len(X.columns) > 1
        # Get the categorical columns
        self.columnName = X.select_dtypes(include=['object', 'category']).columns
        # Store the number of features
        self._n_features = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, '_n_features')
        assert isinstance(X, pd.DataFrame) and X.shape[1] == self._n_features
        print('Converting to category')
        for column in self.columnName:
            # Convert to category codes
            X[column] = X[column].astype('category').cat.codes
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

        for column, bounds in self.bounds.items():
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
            mode_value = X[column].mode().iloc[0] if not X[column].mode().empty else np.nan
            X[column] = X[column].fillna(mode_value)
            # X[column].fillna(mode_value)

            # Convert back to categorical if necessary
            if isCategory:
                X[column] = pd.Categorical.from_codes(
                    X[column].fillna(-1).astype(int), categories=categories, ordered=True
                ).remove_unused_categories()
                X[column] = X[column].astype('category')
              #  print(X[column].value_counts())

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
        self.onehot.fit(X[['Occupation']])
        self.feature_names_out = self.onehot.get_feature_names_out(['Occupation'])
        return self

    def transform(self, X):
        encoded = self.onehot.transform(X[['Occupation']])
        encoded_df = pd.DataFrame(encoded, columns=self.onehot.get_feature_names_out(['Occupation']))
        X = X.reset_index(drop=True)
        return pd.concat([X, encoded_df], axis=1).drop(columns=['Occupation'])

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
    
    # Categorical Columns (ordered)
    'Credit_Mix': {'lo': 0, 'hi': 2},  # Assuming "Bad" = 0, "Standard" = 1, "Good" = 2
    # Categorical Column (nominal)
    'Occupation': {'lo': 0, 'hi': 14}  # No bounds, only impute missing values
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
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('Transforming Month')
        X['Month'] = X['Month'].astype('category').cat.codes
        X['Month_sin'] = np.sin(2 * np.pi * X['Month'] + 1 / 12)
        X['Month_cos'] = np.cos(2 * np.pi * X['Month'] + 1 / 12)
        X.drop(columns=['Month'], inplace=True)
        return X    

pipeline = Pipeline(steps= [
    ('NumericCleaner', NumericCleaner()),
    ('CreditHistoryTransformer', CreditHistoryTransformer()),
    ('PaymentBehaviourTransformer', PaymentBehaviourTransformer()),
    ('LoanType', LoanType()),
    ('HiLoTransformer', HiLoTransformer(bounds=hi_lo_bounds)), 
    ('OccupationOneHotEncoder', OccupationOneHotEncoder()),
    ('DropColumnsTransformer', DropColumnsTransformer(columns=['Customer_ID', 'ID', 'Name', 'SSN'])),
    ('toCategoryCodes', toCategoryCodes()),
    ('MonthTransformer', MonthTransformer()),
    ('DropNanRows', DropNanRows())
])


import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted

# Assuming the custom transformers and pipeline are already defined as in the previous code

# Fit and transform the data using the pipeline
cleanedData = pipeline.fit_transform(train)
X = cleanedData.drop('Credit_Score', axis=1)
y = cleanedData['Credit_Score']

#lets set aside the data for neural network
#so we can scale the data


#now we can scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_nn = scaler.fit_transform(X)




# Split into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step to scale the data
    ('mlp', MLPClassifier(random_state=42))  # Step to apply the MLPClassifier
])

# Train the model
mlp_pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = mlp_pipeline.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print('Validation Accuracy:', accuracy_val)

# Make predictions on the test set
y_pred_test = mlp_pipeline.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print('Test Accuracy:', accuracy_test)
# Define the models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42),
    'RandomForest_100': RandomForestClassifier(random_state=42, n_estimators=100),
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == 'MLP':
        #model.fit(X_nn, y_train)
        print('MLP')
        continue
    else:
        model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print(f"{model_name} Validation Accuracy: {accuracy_val}")
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"{model_name} Test Accuracy: {accuracy_test}")
    #lets print the cross validation score
    

# Example of hyperparameter tuning for one of the models (e.g., GradientBoosting)
param_grid_gb = {
    'n_estimators': [200, 100, 150],
    'max_depth': [8, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

gb_clf = GradientBoostingClassifier(random_state=42)
start_time = time.time()
gb_random_search = RandomizedSearchCV(
    gb_clf, param_distributions=param_grid_gb, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
gb_random_search.fit(X_train, y_train)
print("GradientBoosting best params:", gb_random_search.best_params_)
print("best score:", gb_random_search.best_score_)
print("Training time for GradientBoosting:", time.time() - start_time)

# Evaluate the best model on the test set
y_pred_test = gb_random_search.best_estimator_.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Best GradientBoosting Test Accuracy:", accuracy_test)
y_pred_val = gb_random_search.best_estimator_.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Best GradientBoosting Validation Accuracy:", accuracy_val)




rf_clf = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=None,
    bootstrap=False,
    random_state=42  # You can set a random state for reproducibility
)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

y_pred_test = rf_clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)

param_grid_rf = {
    'n_estimators': [200, 100, 150],           # Number of trees in forest
    'max_depth': [10, 15, 20, None],          # Maximum depth of each tree
    'max_features': ['sqrt', 'log2'],         # Number of features to consider at each split
    'min_samples_split': [2, 5, 10],          # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],            # Minimum samples required at each leaf node
    'bootstrap': [True, False]                # Whether bootstrap samples are used
}

rf_clf = RandomForestClassifier(random_state=42)
start_time = time.time()
rf_random_search = RandomizedSearchCV(
    rf_clf, param_distributions=param_grid_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
rf_random_search.fit(X_train, y_train)
print("RandomForest best params:", rf_random_search.best_params_)
print("Training time for RandomForest:", time.time() - start_time)