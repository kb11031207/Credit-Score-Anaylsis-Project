# Credit Score Analysis Project

A comprehensive machine learning implementation for credit score prediction using borrower behavior data. This project analyzes 27 borrower properties to predict credit scores as Good, Standard, or Poor using multiple ML approaches including decision trees, ensemble methods, and neural networks.

## 🚀 Features

- **Advanced Data Preprocessing**: Custom transformers for cleaning financial data
- **Multiple ML Models**: Random Forest, Decision Trees, Extra Trees, Gradient Boosting, AdaBoost, and Neural Networks
- **Hyperparameter Optimization**: Automated tuning using RandomizedSearchCV
- **Outlier Detection**: Customer-specific outlier correction using mode imputation
- **Feature Engineering**: Cyclical encoding, ordinal encoding, and one-hot encoding
- **Deep Learning**: TensorFlow/Keras implementation with early stopping

## 📁 Project Structure

```
Credit-Score-Analysis-Project/
├── dataprocessing.py              # Data preprocessing pipeline with custom transformers
├── models.py                      # Multiple ML models with hyperparameter optimization
├── NN_MODEL.py                    # Deep learning model using TensorFlow/Keras
├── DecisionTrees.py               # Decision tree-based models with optimized parameters
├── Hyperparam_Results.txt         # Model performance results and optimization output
├── requirements.txt               # Python dependencies
├── train.csv                      # Training dataset (29MB)
├── test.csv                       # Test dataset (15MB)
├── train_transformed.csv          # Preprocessed training data (24MB)
├── experimental/                  # Development and experimental files
│   ├── Lastmodel.py
│   ├── somthn.py
│   ├── printcol.py
│   ├── model.ipynb
│   ├── last.ipynb
│   └── modelsTest.ipynb
└── CCDataSpec (1).html           # Project specifications
```

## 🛠️ Setup & Usage

### Prerequisites
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (for large datasets)
- **Storage**: 100MB+ for datasets and models

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Credit-Score-Analysis-Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Run data preprocessing**:
   ```bash
   python dataprocessing.py
   ```
   This creates `train_transformed.csv` with cleaned and transformed data.

2. **Train multiple ML models**:
   ```bash
   python models.py
   ```
   Performs hyperparameter optimization on all models.

3. **Train neural network**:
   ```bash
   python NN_MODEL.py
   ```
   Trains deep learning model with early stopping.

4. **Run decision tree models**:
   ```bash
   python DecisionTrees.py
   ```
   Evaluates optimized decision tree-based models.

## 📊 Implementation Details

### Data Preprocessing Pipeline (`dataprocessing.py`)

#### Required Transformers (per project specifications):

1. **Wild Outlier Correction** (`HiLoTransformer`):
   - Uses hi/lo bounds for outlier detection
   - Customer-specific mode imputation for corrections
   - Handles missing data in categorical variables

2. **Loan Type Expansion** (`LoanType`):
   - MultiLabelBinarizer for multiple loan types
   - Creates binary columns for each loan type

3. **Payment Behavior Translation** (`PaymentBehaviourTransformer`):
   - Splits payment behavior into charge and payment components
   - Converts to numerical values (0-2 scale)

4. **Credit History Age Conversion** (`CreditHistoryTransformer`):
   - Converts verbal descriptions to total months
   - Handles "X Years Y Months" format

5. **Feature Scaling** (`StandardScaler`):
   - Required for MLP models
   - Optional for decision tree algorithms

### Model Implementations

#### Decision Tree Models (`models.py`, `DecisionTrees.py`)
- **Random Forest**: Ensemble of decision trees with optimized parameters
- **Extra Trees**: Extremely randomized trees for better generalization
- **Gradient Boosting**: Sequential boosting with gradient descent
- **AdaBoost**: Adaptive boosting for improved accuracy

#### Neural Network (`NN_MODEL.py`)
- **Architecture**: Multiple dense layers with dropout
- **Optimization**: Adam optimizer with early stopping
- **Scaling**: StandardScaler for feature normalization

### Performance Targets

- **Minimum Accuracy**: 93% on test.csv (per project specifications)
- **Training Time**: Maximum 10 minutes for demo requirements
- **Validation**: Group-based splitting to prevent data leakage

## 🧪 Testing & Validation

### Data Splitting Strategy
- **GroupShuffleSplit**: Ensures customers don't appear in both train/test sets
- **Three-way split**: Train (70%), Validation (15%), Test (15%)

### Model Evaluation
- **Cross-validation**: 5-fold CV for hyperparameter tuning
- **Multiple metrics**: Accuracy, validation performance, test performance
- **Hyperparameter optimization**: RandomizedSearchCV for efficient tuning

## 📈 Performance Results

### Best Model Performances (from Hyperparam_Results.txt):
- **Gradient Boosting**: 70.92% test accuracy
- **Random Forest**: 70.48% test accuracy  
- **Extra Trees**: 69.17% test accuracy
- **MLP**: 66.64% test accuracy

### Key Achievements:
- ✅ **Data Cleaning**: Comprehensive outlier detection and correction
- ✅ **Feature Engineering**: Advanced transformations for financial data
- ✅ **Model Diversity**: Multiple algorithm approaches
- ✅ **Hyperparameter Optimization**: Automated tuning for all models
- ✅ **Deep Learning**: Neural network implementation with early stopping

## 🎯 Use Cases

- **Credit Risk Assessment**: Predict borrower creditworthiness
- **Financial Services**: Automated credit scoring systems
- **Machine Learning Education**: Comprehensive ML pipeline example
- **Data Science Projects**: Advanced preprocessing and modeling techniques

## 📝 Project Requirements Met

### Data Cleaning :
- ✅ **Wild Outlier Correction**: HiLoTransformer with customer-specific imputation
- ✅ **Loan Type Expansion**: MultiLabelBinarizer for multiple categories
- ✅ **Payment Behavior Translation**: Split into charge/payment components
- ✅ **Credit History Age**: Convert verbal descriptions to months
- ✅ **Feature Scaling**: StandardScaler for MLP compatibility
- ✅ **Pipeline Assembly**: Complete preprocessing pipeline

### Model Requirements:
- ✅ **Decision Tree Method**: Random Forest with ensemble improvements
- ✅ **Alternative Method**: MLP neural network
- ✅ **Multiple Approaches**: 6+ different algorithms tested
- ✅ **Hyperparameter Tuning**: Automated optimization for all models

## 👨‍💻 Author

**Kesiena Berezi** - Credit Score Analysis Project

## 📄 License

This project is licensed under the MIT License.

## 🔍 Notes

- **Large Datasets**: train.csv (29MB) and test.csv (15MB) are included
- **Experimental Files**: Development versions stored in `experimental/` directory
- **Performance**: Results stored in `Hyperparam_Results.txt`
- **Dependencies**: All requirements listed in `requirements.txt`
- **Specifications**: Project requirements detailed in `CCDataSpec (1).html` 