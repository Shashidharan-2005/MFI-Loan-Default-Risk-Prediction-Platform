# generate_mfi_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MFIDataGenerator:
    """Generate realistic MFI loan data for training ML models"""
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_synthetic_data(self):
        """Generate synthetic MFI loan data"""
        
        # Age distribution (18-65, normal distribution around 35)
        age = np.random.normal(35, 12, self.n_samples)
        age = np.clip(age, 18, 65).astype(int)
        
        # Gender (slightly more females in microfinance)
        gender = np.random.choice(['Male', 'Female'], self.n_samples, p=[0.4, 0.6])
        
        # Income (log-normal distribution, IDR thousands)
        income = np.random.lognormal(7.5, 0.8, self.n_samples)
        income = np.clip(income, 500, 20000).astype(int)
        
        # Education levels
        education = np.random.choice(['Primary', 'Secondary', 'Higher'], 
                                   self.n_samples, p=[0.4, 0.45, 0.15])
        
        # Employment type
        employment = np.random.choice(['Formal', 'Informal', 'Self-employed'], 
                                    self.n_samples, p=[0.3, 0.4, 0.3])
        
        # Marital status
        marital_status = np.random.choice(['Single', 'Married', 'Divorced'], 
                                        self.n_samples, p=[0.25, 0.65, 0.1])
        
        # Location
        location = np.random.choice(['Urban', 'Rural'], self.n_samples, p=[0.4, 0.6])
        
        # Mobile usage months (0-60)
        mobile_usage = np.random.exponential(15, self.n_samples)
        mobile_usage = np.clip(mobile_usage, 0, 60).astype(int)
        
        # Transaction frequency (correlated with mobile usage)
        transaction_freq = mobile_usage * np.random.uniform(0.5, 2.5, self.n_samples)
        transaction_freq = np.clip(transaction_freq, 0, 100).astype(int)
        
        # Previous loans (0-8)
        previous_loans = np.random.poisson(1.5, self.n_samples)
        previous_loans = np.clip(previous_loans, 0, 8)
        
        # Previous defaults (always <= previous loans)
        previous_defaults = []
        for loans in previous_loans:
            if loans == 0:
                defaults = 0
            else:
                # 20% chance of having defaults, increasing with more loans
                default_prob = min(0.2 + loans * 0.05, 0.6)
                if np.random.random() < default_prob:
                    defaults = np.random.randint(1, min(loans + 1, 4))
                else:
                    defaults = 0
            previous_defaults.append(defaults)
        previous_defaults = np.array(previous_defaults)
        
        # Loan amount (5-12 IDR, influenced by income)
        loan_amount_base = 5 + (income / 20000) * 7  # Scale with income
        loan_amount = loan_amount_base + np.random.normal(0, 1, self.n_samples)
        loan_amount = np.clip(loan_amount, 5, 12)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'income': income,
            'loan_amount': loan_amount,
            'education': education,
            'employment': employment,
            'marital_status': marital_status,
            'location': location,
            'mobile_usage': mobile_usage,
            'transaction_freq': transaction_freq,
            'previous_loans': previous_loans,
            'previous_defaults': previous_defaults
        })
        
        # Generate target variable (default) based on realistic factors
        data['default'] = self.generate_default_labels(data)
        
        return data
    
    def generate_default_labels(self, data):
        """Generate realistic default labels based on risk factors"""
        
        # Base default probability
        default_prob = np.full(len(data), 0.1)
        
        # Age factor (younger and older are riskier)
        age_risk = np.where(data['age'] < 25, 0.15, 
                           np.where(data['age'] > 55, 0.10, 0.0))
        default_prob += age_risk
        
        # Income factor (lower income = higher risk)
        income_risk = np.where(data['income'] < 2000, 0.20,
                              np.where(data['income'] < 3500, 0.10, 0.0))
        default_prob += income_risk
        
        # Previous defaults (strongest predictor)
        default_history_risk = data['previous_defaults'] * 0.25
        default_prob += default_history_risk
        
        # Employment risk
        employment_risk = np.where(data['employment'] == 'Informal', 0.15,
                                  np.where(data['employment'] == 'Self-employed', 0.08, 0.0))
        default_prob += employment_risk
        
        # Education protective factor
        education_protection = np.where(data['education'] == 'Higher', -0.10,
                                       np.where(data['education'] == 'Secondary', -0.05, 0.0))
        default_prob += education_protection
        
        # Mobile experience protective factor
        mobile_protection = np.where((data['mobile_usage'] > 12) & (data['transaction_freq'] > 20), 
                                    -0.12, 0.0)
        default_prob += mobile_protection
        
        # Debt-to-income ratio risk
        debt_ratio = data['loan_amount'] / (data['income'] / 1000)
        debt_risk = np.where(debt_ratio > 0.4, 0.18,
                            np.where(debt_ratio > 0.3, 0.10, 0.0))
        default_prob += debt_risk
        
        # Location factor
        location_risk = np.where(data['location'] == 'Rural', 0.05, 0.0)
        default_prob += location_risk
        
        # Ensure probabilities are in valid range
        default_prob = np.clip(default_prob, 0.01, 0.85)
        
        # Generate binary labels based on probabilities
        defaults = np.random.binomial(1, default_prob, len(data))
        
        return defaults

class MFIDataPreprocessor:
    """Preprocess MFI data for machine learning"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def preprocess_data(self, data):
        """Complete preprocessing pipeline"""
        
        # Create a copy
        processed_data = data.copy()
        
        # Feature engineering
        processed_data = self.engineer_features(processed_data)
        
        # Encode categorical variables
        processed_data = self.encode_categorical(processed_data)
        
        # Separate features and target
        X = processed_data.drop('default', axis=1)
        y = processed_data['default']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y
    
    def engineer_features(self, data):
        """Create new features from existing ones"""
        
        # Debt-to-income ratio
        data['debt_to_income_ratio'] = data['loan_amount'] / (data['income'] / 1000)
        
        # Default rate (previous defaults / previous loans)
        data['default_rate'] = np.where(data['previous_loans'] > 0, 
                                       data['previous_defaults'] / data['previous_loans'], 0)
        
        # Mobile experience score
        data['mobile_experience'] = (data['mobile_usage'] * 0.6 + 
                                   data['transaction_freq'] * 0.4)
        
        # Age groups
        data['age_group'] = pd.cut(data['age'], 
                                 bins=[0, 25, 35, 50, 100], 
                                 labels=['Young', 'Adult', 'Middle', 'Senior'])
        
        # Income categories
        data['income_category'] = pd.cut(data['income'], 
                                       bins=[0, 2000, 5000, 10000, float('inf')], 
                                       labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # Risk score based on multiple factors
        risk_components = []
        
        # Age risk
        age_risk = np.where(data['age'] < 25, 2, 
                           np.where(data['age'] > 55, 1, 0))
        risk_components.append(age_risk)
        
        # Income risk
        income_risk = np.where(data['income'] < 2500, 3, 
                             np.where(data['income'] < 5000, 1, 0))
        risk_components.append(income_risk)
        
        # Employment risk
        employment_risk = np.where(data['employment'] == 'Informal', 2,
                                 np.where(data['employment'] == 'Self-employed', 1, 0))
        risk_components.append(employment_risk)
        
        # Default history risk
        default_history_risk = data['previous_defaults'] * 2
        risk_components.append(default_history_risk)
        
        # Combine risk components
        data['composite_risk_score'] = sum(risk_components)
        
        return data
    
    def encode_categorical(self, data):
        """Encode categorical variables"""
        
        categorical_columns = ['gender', 'education', 'employment', 'marital_status', 
                             'location', 'age_group', 'income_category']
        
        for column in categorical_columns:
            if column in data.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                data[column] = self.label_encoders[column].fit_transform(data[column])
        
        return data

def perform_eda(data):
    """Perform Exploratory Data Analysis"""
    
    print("=== MFI Loan Data - Exploratory Data Analysis ===\n")
    
    # Basic info
    print("Dataset Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)
    
    print("\nBasic Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nDefault Rate:")
    default_rate = data['default'].mean()
    print(f"Overall default rate: {default_rate:.2%}")
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. Default distribution
    plt.subplot(3, 4, 1)
    data['default'].value_counts().plot(kind='bar')
    plt.title('Default Distribution')
    plt.xticks([0, 1], ['No Default', 'Default'], rotation=0)
    
    # 2. Age distribution
    plt.subplot(3, 4, 2)
    plt.hist(data['age'], bins=20, alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    
    # 3. Income distribution
    plt.subplot(3, 4, 3)
    plt.hist(data['income'], bins=30, alpha=0.7)
    plt.title('Income Distribution (IDR thousands)')
    plt.xlabel('Income')
    
    # 4. Loan amount distribution
    plt.subplot(3, 4, 4)
    plt.hist(data['loan_amount'], bins=20, alpha=0.7)
    plt.title('Loan Amount Distribution')
    plt.xlabel('Loan Amount (IDR)')
    
    # 5. Default rate by gender
    plt.subplot(3, 4, 5)
    default_by_gender = data.groupby('gender')['default'].mean()
    default_by_gender.plot(kind='bar')
    plt.title('Default Rate by Gender')
    plt.xticks(rotation=45)
    
    # 6. Default rate by education
    plt.subplot(3, 4, 6)
    default_by_education = data.groupby('education')['default'].mean()
    default_by_education.plot(kind='bar')
    plt.title('Default Rate by Education')
    plt.xticks(rotation=45)
    
    # 7. Default rate by employment
    plt.subplot(3, 4, 7)
    default_by_employment = data.groupby('employment')['default'].mean()
    default_by_employment.plot(kind='bar')
    plt.title('Default Rate by Employment')
    plt.xticks(rotation=45)
    
    # 8. Income vs Loan Amount
    plt.subplot(3, 4, 8)
    colors = ['blue' if x == 0 else 'red' for x in data['default']]
    plt.scatter(data['income'], data['loan_amount'], c=colors, alpha=0.5)
    plt.title('Income vs Loan Amount')
    plt.xlabel('Income (IDR thousands)')
    plt.ylabel('Loan Amount (IDR)')
    
    # 9. Previous defaults impact
    plt.subplot(3, 4, 9)
    default_by_prev_defaults = data.groupby('previous_defaults')['default'].mean()
    default_by_prev_defaults.plot(kind='bar')
    plt.title('Default Rate by Previous Defaults')
    plt.xlabel('Previous Defaults')
    
    # 10. Mobile usage impact
    plt.subplot(3, 4, 10)
    data['mobile_usage_bins'] = pd.cut(data['mobile_usage'], bins=5)
    default_by_mobile = data.groupby('mobile_usage_bins')['default'].mean()
    default_by_mobile.plot(kind='bar')
    plt.title('Default Rate by Mobile Usage')
    plt.xticks(rotation=45)
    
    # 11. Correlation heatmap
    plt.subplot(3, 4, 11)
    numerical_cols = ['age', 'income', 'loan_amount', 'mobile_usage', 
                     'transaction_freq', 'previous_loans', 'previous_defaults', 'default']
    correlation_matrix = data[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # 12. Default rate by location
    plt.subplot(3, 4, 12)
    default_by_location = data.groupby('location')['default'].mean()
    default_by_location.plot(kind='bar')
    plt.title('Default Rate by Location')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('mfi_eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print insights
    print("\n=== Key Insights ===")
    print(f"1. Overall default rate: {default_rate:.2%}")
    print(f"2. Gender impact: Female default rate: {data[data['gender']=='Female']['default'].mean():.2%}, Male: {data[data['gender']=='Male']['default'].mean():.2%}")
    print(f"3. Education impact: Primary: {data[data['education']=='Primary']['default'].mean():.2%}, Secondary: {data[data['education']=='Secondary']['default'].mean():.2%}, Higher: {data[data['education']=='Higher']['default'].mean():.2%}")
    print(f"4. Employment impact: Formal: {data[data['employment']=='Formal']['default'].mean():.2%}, Informal: {data[data['employment']=='Informal']['default'].mean():.2%}, Self-employed: {data[data['employment']=='Self-employed']['default'].mean():.2%}")
    print(f"5. Location impact: Urban: {data[data['location']=='Urban']['default'].mean():.2%}, Rural: {data[data['location']=='Rural']['default'].mean():.2%}")
    
    # High risk indicators
    high_risk = data[data['previous_defaults'] > 0]['default'].mean()
    low_income_risk = data[data['income'] < 2000]['default'].mean()
    
    print(f"6. High risk indicators:")
    print(f"   - Customers with previous defaults: {high_risk:.2%} default rate")
    print(f"   - Low income customers (<2000): {low_income_risk:.2%} default rate")

if __name__ == "__main__":
    # Generate data
    print("Generating MFI loan data...")
    generator = MFIDataGenerator(n_samples=10000)
    data = generator.generate_synthetic_data()
    
    # Save raw data
    data.to_csv('mfi_loan_data.csv', index=False)
    print("Raw data saved as 'mfi_loan_data.csv'")
    
    # Perform EDA
    perform_eda(data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = MFIDataPreprocessor()
    X, y = preprocessor.preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("Processed data saved:")
    print(f"- Training set: {X_train.shape}")
    print(f"- Test set: {X_test.shape}")
    print(f"- Training default rate: {y_train.mean():.2%}")
    print(f"- Test default rate: {y_test.mean():.2%}")
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, 'mfi_preprocessor.pkl')
    print("Preprocessor saved as 'mfi_preprocessor.pkl'")
