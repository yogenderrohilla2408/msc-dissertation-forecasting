#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import Libraries and Load Dataset

# In[1]:


# Load Library and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

tsla_data=pd.read_csv('tsla_2014_2023.csv')


# # Step 2: Initial Data Exploration

# In[2]:


tsla_data.head()


# In[3]:


tsla_data.describe()


# In[4]:


tsla_data.info()


# # Step 3: Data Preprocessing, (a) Handling Missing Values

# In[5]:


# Check for missing values in the dataset
missing_values = tsla_data.isnull().sum()
print(missing_values[missing_values > 0])  # This will print columns with missing values and their count


# In[6]:


# Get a statistical summary of the numerical features
statistical_summary = tsla_data.describe()
print(statistical_summary)


# In[7]:


# Using median imputation for 'low'
tsla_data['low'].fillna(tsla_data['low'].median(), inplace=True)


# In[8]:


# Assuming the dates are sorted and there is a regular interval (e.g., daily)
# You can fill the missing date with the next day after the last valid date before the missing value
tsla_data['date'] = pd.to_datetime(tsla_data['date'])  # Convert to datetime if it's not already
tsla_data = tsla_data.sort_values('date')  # Ensure the data is sorted by date
tsla_data['date'].fillna(method='ffill', inplace=True)  # Forward-fill the missing date
tsla_data['date'] = tsla_data['date'] + pd.Timedelta(days=1)  # Add one day to the last valid date


# In[9]:


# Verify there are no missing values left
missing_values = tsla_data.isnull().sum()
print(missing_values)


# # (b)Data Cleaning

# In[10]:


# Check data types
print(tsla_data.dtypes)
tsla_data['date'] = pd.to_datetime(tsla_data['date'], errors='coerce')  # Coerce any problematic data into NaT


# In[11]:


import pandas as pd

# tsla_data is DataFrame

# Find duplicate rows
duplicates = tsla_data.duplicated()

# Count the number of duplicates
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Optional: Display the duplicate rows
if num_duplicates > 0:
    print(tsla_data[duplicates])
    
    
# Remove duplicate rows
# tsla_data = tsla_data.drop_duplicates()

# Verify the duplicates are removed
# print(f"Number of duplicate rows after removal: {tsla_data.duplicated().sum()}")



# In[12]:


# Specify columns to handle outliers
price_columns = ['open', 'high', 'low', 'close', 'next_day_close']
indicator_columns = ['rsi_7', 'rsi_14', 'cci_7', 'cci_14', 'sma_50', 'ema_50', 'sma_100', 'ema_100', 'macd', 'bollinger', 'TrueRange', 'atr_7', 'atr_14']

# Function to handle outliers
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping the data
    data[column] = np.where(data[column] > upper_bound, upper_bound,
                            np.where(data[column] < lower_bound, lower_bound, data[column]))

# Apply the function to each column
for col in price_columns + indicator_columns:
    handle_outliers(tsla_data, col)

# Log transformation for 'volume' because of skewness
tsla_data['volume_log'] = np.log(tsla_data['volume'] + 1)  # Adding 1 to avoid log(0)

# Save the cleaned data
tsla_data.to_csv('tsla_cleaned.csv', index=False)


# In[13]:


# Separate features and target
X = tsla_data.drop('next_day_close', axis=1)
y = tsla_data['next_day_close']

# Reset index to align X and y
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)


# # (c) Feature Engineering

# In[14]:


# Select only the numerical columns for descriptive statistics
numerical_data = tsla_data.select_dtypes(include=['float64', 'int64'])

# Now you can safely compute descriptive statistics
print(numerical_data.describe())

# Skewness and Kurtosis
print(numerical_data.skew())
print(numerical_data.kurtosis())


# # (d) Further Feature Engineering & Visual Data Analysis

# In[15]:


import matplotlib.pyplot as plt

# Now you can use plt to create plots
# Define the number of rows and columns you need
nrows = (len(tsla_data.columns) + 3) // 4  # Adjust based on number of columns

# Adjust figsize if needed
figsize = (20, nrows * 4)  # You can adjust the height based on the number of rows

# Create box plots with the correct layout
tsla_data.plot(kind='box', subplots=True, layout=(nrows, 4), sharex=False, sharey=False, figsize=figsize)
plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.show()


# In[16]:


import seaborn as sns  # Make sure to add this import statement at the beginning of code

# Pearson Correlation
correlation_matrix = tsla_data.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Spearman Correlation
spearman_matrix = tsla_data.corr(method='spearman')
sns.heatmap(spearman_matrix, annot=True)
plt.show()


# In[17]:


# Pair plots for a subset of features
sns.pairplot(tsla_data[['close', 'volume', 'sma_50', 'ema_50']])
plt.show()


# In[18]:


# Time series plot for closing prices
tsla_data.plot(x='date', y='close', figsize=(14,7))
plt.show()


# In[19]:


# Preparing data for phase space plot
tsla_data['lagged_close'] = tsla_data['close'].shift(1)

plt.figure(figsize=(10, 8))
plt.scatter(tsla_data['lagged_close'], tsla_data['close'], alpha=0.5)
plt.title('Phase Space Plot of Tesla Stock Closing Prices')
plt.xlabel('Lagged Closing Price')
plt.ylabel('Closing Price')
plt.grid(True)
plt.show()


# In[20]:


# Rolling mean and standard deviation
tsla_data['rolling_mean'] = tsla_data['close'].rolling(window=20).mean()
tsla_data['rolling_std'] = tsla_data['close'].rolling(window=20).std()

# Plot rolling statistics
plt.plot(tsla_data['date'], tsla_data['close'], label='Original')
plt.plot(tsla_data['date'], tsla_data['rolling_mean'], label='Rolling Mean')
plt.plot(tsla_data['date'], tsla_data['rolling_std'], label='Rolling Std')
plt.legend()
plt.show()


# In[21]:


# Log transformation
tsla_data['log_close'] = np.log(tsla_data['close'])

# Polynomial features (example: squared term of 'volume')
tsla_data['volume_squared'] = tsla_data['volume'] ** 2


# In[22]:


from scipy import stats

# Z-score for outlier detection
tsla_data['z_score_close'] = stats.zscore(tsla_data['close'])

# Consider data points where z-score > 3 or < -3 as outliers
outliers = tsla_data[(tsla_data['z_score_close'] > 3) | (tsla_data['z_score_close'] < -3)]


# In[23]:


# Scatter plot for outlier detection in 'close' prices
plt.scatter(tsla_data['date'], tsla_data['close'])
plt.show()


# In[24]:


# Define 'specific_date' with the actual date string you want to compare against
specific_date = pd.to_datetime('2021-01-29')  # Replace with actual date

# Now, perform the comparison and the t-test as before
pre_launch = tsla_data[tsla_data['date'] < specific_date]['close']
post_launch = tsla_data[tsla_data['date'] >= specific_date]['close']

# Perform the t-test
t_stat, p_val = stats.ttest_ind(pre_launch.dropna(), post_launch.dropna(), equal_var=False)  # Drop NaNs and assuming unequal variance
print(f"T-statistic: {t_stat}, P-value: {p_val}")


# In[25]:


# Check normality of distributions using the Shapiro-Wilk test
print(stats.shapiro(pre_launch))
print(stats.shapiro(post_launch))

# Visualize the distributions using histograms
sns.histplot(pre_launch, color='blue', kde=True, label='Pre-Launch')
sns.histplot(post_launch, color='red', kde=True, label='Post-Launch')
plt.legend()
plt.show()

# Visualize the distributions using boxplots
sns.boxplot(data=[pre_launch, post_launch])
plt.xticks([0, 1], ['Pre-Launch', 'Post-Launch'])
plt.show()

# Check for homogeneity of variances using Levene's test
print(stats.levene(pre_launch, post_launch))

# Calculate and print the effect size (Cohen's d)
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / ( ((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof ) ** 0.5

d = cohens_d(pre_launch, post_launch)
print(f"Cohen's d: {d}")


# In[26]:


from scipy.stats import mannwhitneyu

# Perform the Mann-Whitney U Test
u_stat, p_value = mannwhitneyu(pre_launch, post_launch, alternative='two-sided')

print(f"Mann-Whitney U statistic: {u_stat}, P-value: {p_value}")


# In[27]:


# calculate the Lyapunov exponent from a time series

def estimate_lyapunov(data, min_divergence=1e-10):
    N = len(data)
    divergence_sum = 0
    divergent_points = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            # Calculate the absolute difference between two points
            delta = np.abs(data[j] - data[i])

            # Avoid log of zero or negative by ensuring a minimum divergence
            if delta > min_divergence:
                divergence_sum += np.log(delta)
                divergent_points += 1

    # Take the average divergence; here we use the max to avoid division by zero
    avg_divergence = divergence_sum / max(divergent_points, 1)
    
    # Estimate the Lyapunov exponent
    lyapunov_exp = avg_divergence / (N - 1)
    return lyapunov_exp

# Generate a sample data set - replace this with your actual data
data = np.random.random(100)

# Estimate the Lyapunov exponent
lyap_exp = estimate_lyapunov(data)
print(f"Estimated Lyapunov Exponent: {lyap_exp}")


# In[28]:


# Rolling mean and standard deviation of the closing prices
tsla_data['rolling_mean'] = tsla_data['close'].rolling(window=20).mean()
tsla_data['rolling_std'] = tsla_data['close'].rolling(window=20).std()

# Logarithmic return of the closing price
tsla_data['log_return'] = np.log(tsla_data['close'] / tsla_data['close'].shift(1))

# Volatility measurement (standard deviation of logarithmic return)
tsla_data['volatility'] = tsla_data['log_return'].rolling(window=10).std() * np.sqrt(10)

# Plotting feature engineering results for visualization
fig, axs = plt.subplots(nrows=3, figsize=(10, 15))
tsla_data[['close', 'rolling_mean']].plot(ax=axs[0])
tsla_data[['close', 'rolling_std']].plot(ax=axs[1])
tsla_data[['close', 'volatility']].plot(ax=axs[2])
plt.tight_layout()
plt.show()

# Save the plot as a figure in your dissertation document where appropriate
# plt.savefig('path/to/save/feature_engineering_plots.png')

# Preparing data for machine learning
selected_features = [
    'open', 'high', 'low', 'close', 'volume',
    'rolling_mean', 'rolling_std', 'volatility'
]

# Scale features
scaler = StandardScaler()
X = tsla_data[selected_features]
y = tsla_data['close']  # Target variable, could be changed based on specific needs
X_scaled = scaler.fit_transform(X)



# Summary statistics of the dataset post-feature engineering
print(tsla_data.describe())


# In[29]:


# Generate normal distributed data for demonstration
data = np.random.normal(0, 1, 1000)

# Q-Q plot
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()


# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

# Data Preprocessing


# Normally, we would determine which features to keep based on domain knowledge and EDA
selected_features = [
    'open', 'high', 'low', 'close', 'volume', 
    'rsi_7', 'rsi_14', 'cci_7', 'cci_14', 
    'sma_50', 'ema_50', 'sma_100', 'ema_100', 
    'macd', 'bollinger', 'TrueRange', 'atr_7', 'atr_14'
]

# Feature engineering have already been done as part of EDA



# # Step 7: Data Preparation for Modeling

# In[31]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Separate features from target
X = tsla_data[selected_features]
y = tsla_data['next_day_close']  # Assuming 'next_day_close' is the target variable

# Scale the features
X_scaled = scaler.fit_transform(X)


# In[32]:


from sklearn.model_selection import train_test_split

# Assuming X_scaled and y have been defined as shown in previous code:
# X_scaled = scaler.fit_transform(X)
# y = tsla_data['next_day_close']

# Now you can split scaled features and target variable into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# In[33]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create a pipeline for Random Forest Regressor
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Fit the pipeline on the training data
rf_pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
# You can use rf_pipeline.predict(X_test) to make predictions and evaluate the model


# # Step 8: Model Training and Evaluation

# In[34]:


# Training Phase of RF
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming X_scaled is the entire scaled feature set and y is the target variable
# Initialize the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict on the training set to get training MSE and R²
train_predictions = rf_regressor.predict(X_train)

# Calculate MSE and R² for the training set
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

# Output the MSE and R² results for the training set
print(f'Random Forest Regressor Train MSE: {train_mse:.4f}')
print(f'Random Forest Regressor Train R²: {train_r2:.4f}')

# Perform cross-validation using the whole dataset
rf_cv_scores = cross_val_score(rf_regressor, X_scaled, y, cv=5)

# Perform cross-validation to get negative MSE scores
rf_neg_mse_scores = cross_val_score(rf_regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

# Convert to positive MSE scores
rf_mse_scores = -rf_neg_mse_scores

# Output the cross-validation results
print("Random Forest Regressor Cross-Validation Scores:", rf_cv_scores)

# Calculate the mean and standard deviation of the cross-validation scores
rf_cv_mean = rf_cv_scores.mean()
rf_cv_std = rf_cv_scores.std()

# Calculate the mean and standard deviation of the MSE scores
rf_mse_mean = rf_mse_scores.mean()
rf_mse_std = rf_mse_scores.std()

# Print the mean and standard deviation
print(f"Mean cross-validation score for RF: {rf_cv_mean:.4f}")
print(f"Standard deviation of cross-validation scores for RF: {rf_cv_std:.4f}")
print(f"Mean MSE score for Random Forest: {rf_mse_mean:.4f}")
print(f"Standard deviation of MSE scores for Random Forest: {rf_mse_std:.4f}")


# In[35]:


# Gradient Boosting Regressor model
from sklearn.ensemble import GradientBoostingRegressor

# Create a pipeline for Gradient Boosting Regressor
gbr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Fit the pipeline on the training data
gbr_pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
# You can use gbr_pipeline.predict(X_test) to make predictions and evaluate the model


# In[36]:


# Training Phase of GBR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# Assuming X_scaled is the entire scaled feature set and y is the target variable

# Initialize the Gradient Boosting Regressor model
gbr_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model on the training data
gbr_regressor.fit(X_train, y_train)

# Predict on the training set to get training MSE and R²
train_predictions = gbr_regressor.predict(X_train)

# Calculate MSE and R² for the training set
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

# Output the MSE and R² results for the training set
print(f'Gradient Boosting Regressor Train MSE: {train_mse:.4f}')
print(f'Gradient Boosting Regressor Train R²: {train_r2:.4f}')

# Perform cross-validation using the whole dataset
gbr_cv_scores = cross_val_score(gbr_regressor, X_scaled, y, cv=5)

# Perform cross-validation to get negative MSE scores
gbr_neg_mse_scores = cross_val_score(gbr_regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

# Convert to positive MSE scores
gbr_mse_scores = -gbr_neg_mse_scores

# Output the cross-validation results
print("Gradient Boosting Regressor Cross-Validation Scores:", gbr_cv_scores)

# Calculate the mean and standard deviation of the cross-validation scores
gbr_cv_mean = gbr_cv_scores.mean()
gbr_cv_std = gbr_cv_scores.std()

# Calculate the mean and standard deviation of the MSE scores
gbr_mse_mean = gbr_mse_scores.mean()
gbr_mse_std = gbr_mse_scores.std()

# Print the mean and standard deviation
print(f"Mean cross-validation score for GBR: {gbr_cv_mean:.4f}")
print(f"Standard deviation of cross-validation scores for GBR: {gbr_cv_std:.4f}")
print(f"Mean MSE score for Gradient Boosting: {gbr_mse_mean:.4f}")
print(f"Standard deviation of MSE scores for Gradient Boosting: {gbr_mse_std:.4f}")


# In[37]:


import matplotlib.pyplot as plt
from IPython.display import clear_output

# This function is called every epoch
def plot_interactive_loss_curve(epoch, train_loss, val_loss):
    # Clear the previous plot
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    
    # Plot training loss
    plt.plot(range(1, epoch+1), train_loss, label='Training Loss')
    
    # Plot validation loss if available
    if val_loss is not None:
        plt.plot(range(1, epoch+1), val_loss, label='Validation Loss')
    
    # Label the plot
    plt.title('Loss Curve over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example usage: Simulate training process
train_losses = []
val_losses = []
for epoch in range(1, 101):  # Let's say we train for 100 epochs
    # Generate some fake loss data
    train_loss = 0.01 * (100 - epoch) + (np.random.rand() - 0.5)
    val_loss = 0.01 * (100 - 0.9 * epoch) + (np.random.rand() - 0.5)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Plot the loss curve
    plot_interactive_loss_curve(epoch, train_losses, val_losses)


# # Step 9: Model Tuning

# In[38]:


# Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Example for Random Forest
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters for RF:", grid_search.best_params_)

# Example for Gradient Boosting Regressor
gbr_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 4, 5],
}

gbr_grid_search = GridSearchCV(gbr_pipeline, gbr_param_grid, cv=5, scoring='neg_mean_squared_error')
gbr_grid_search.fit(X_train, y_train)

# Best parameters for GBR
print("Best parameters for GBR:", gbr_grid_search.best_params_)


# In[39]:


#Feature Importance Analysis

import matplotlib.pyplot as plt

# Feature importance for Random Forest
importances = rf_pipeline.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting feature importance
plt.figure()
plt.title("Feature importances for Random Forest")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), indices)
plt.show()

# Feature importance for Gradient Boosting Regressor
gbr_importances = gbr_pipeline.named_steps['regressor'].feature_importances_
gbr_indices = np.argsort(gbr_importances)[::-1]

# Plotting feature importance for GBR
plt.figure()
plt.title("Feature importances for GBR")
plt.bar(range(X_train.shape[1]), gbr_importances[gbr_indices])
plt.xticks(range(X_train.shape[1]), gbr_indices)
plt.show()


# In[40]:


# For Random Forest Regressor
rf_predictions = rf_pipeline.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Random Forest')
plt.show()

# Calculating residuals for Random Forest
rf_residuals = y_test - rf_predictions
plt.figure(figsize=(10, 6))
plt.scatter(rf_predictions, rf_residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot for Random Forest Regressor')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[41]:


# For Gradient Boosting Regressor
gbr_predictions = gbr_pipeline.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, gbr_predictions, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Gradient Boosting')
plt.show()

# Calculating residuals for Gradient Boosting
gbr_residuals = y_test - gbr_predictions
plt.figure(figsize=(10, 6))
plt.scatter(gbr_predictions, gbr_residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot for Gradient Boosting Regressor')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[42]:


#Model Ensemble - Averaging Predictions

from sklearn.metrics import mean_squared_error

# Averaging predictions from RF and GBR
ensemble_predictions = (rf_predictions + gbr_predictions) / 2

# Evaluate ensemble
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
print(f"Ensemble MSE: {ensemble_mse}")


# In[43]:


# Test Phase
# Model Evaluation on Test Set

from sklearn.metrics import mean_squared_error, r2_score

# Evaluation for Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest Test MSE: {rf_mse}, R2: {rf_r2}")

# Evaluation for Gradient Boosting Regressor
gbr_mse = mean_squared_error(y_test, gbr_predictions)
gbr_r2 = r2_score(y_test, gbr_predictions)
print(f"GBR Test MSE: {gbr_mse}, R2: {gbr_r2}")


# In[44]:


# Visualizations for Feature Importance
import matplotlib.pyplot as plt

# Assuming 'importances' is a list/array of feature importances from a fitted model
# Ensure 'feature_names' is defined; for example:
feature_names = X.columns  # X should be DataFrame of features

plt.figure(figsize=(10, 6))
# Sort feature importances in descending order
feature_importance_sorted = sorted(zip(importances, feature_names), reverse=True)
# Unzip into two lists
importances_sorted, feature_names_sorted = zip(*feature_importance_sorted)
# Plot the feature importances
plt.barh(range(len(feature_names_sorted)), importances_sorted, align='center')
plt.yticks(range(len(feature_names_sorted)), feature_names_sorted)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()


# In[45]:


# Correlation Heatmap

import seaborn as sns

# Assuming 'X' is features DataFrame
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()


# In[46]:


# Model Validation

from sklearn.model_selection import learning_curve


# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Assuming feature scaling is desired
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Now the pipeline is defined and can be used for predictions
predictions = pipeline.predict(X_test)

# Assuming that you have less than 2000 samples, adjust the train_sizes accordingly
# If you have 1609 samples, all values in train_sizes should be less than or equal to 1609
train_sizes = [50, 100, 250, 500, 1000, 1600]  # Adjust the last value to be less than or equal to number of training samples

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = pipeline,  # Replace 'pipeline' with actual trained model
    X = X_train,
    y = y_train,
    train_sizes = train_sizes,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0)  # Adjust the ylim to better fit error range if necessary
plt.show()



# In[47]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error

# Initial Performance Metrics
# Compute metrics for the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
evs = explained_variance_score(y_test, predictions)
median_ae = median_absolute_error(y_test, predictions)

# Print the metrics
print("Initial Performance Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Median Absolute Error: {median_ae}\n")


# Ensure predictions are computed for both models
rf_predictions = rf_pipeline.predict(X_test)
gbr_predictions = gbr_pipeline.predict(X_test)

# Compute metrics for Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mape = np.mean(np.abs((y_test - rf_predictions) / y_test)) * 100
rf_explained_variance = explained_variance_score(y_test, rf_predictions)
rf_median_ae = median_absolute_error(y_test, rf_predictions)

# Compute metrics for Gradient Boosting Regressor
gbr_mse = mean_squared_error(y_test, gbr_predictions)
gbr_rmse = np.sqrt(gbr_mse)
gbr_mape = np.mean(np.abs((y_test - gbr_predictions) / y_test)) * 100
gbr_explained_variance = explained_variance_score(y_test, gbr_predictions)
gbr_median_ae = median_absolute_error(y_test, gbr_predictions)

# Print metrics for Random Forest
print("Final Random Forest Model Assessment")
print(f"Mean Squared Error: {rf_mse}")
print(f"Root Mean Squared Error: {rf_rmse}")
print(f"Mean Absolute Percentage Error: {rf_mape}%")
print(f"Explained Variance Score: {rf_explained_variance}")
print(f"Median Absolute Error: {rf_median_ae}\n")

# Print metrics for Gradient Boosting Regressor
print("Final Gradient Boosting Regressor Model Assessment")
print(f"Mean Squared Error: {gbr_mse}")
print(f"Root Mean Squared Error: {gbr_rmse}")
print(f"Mean Absolute Percentage Error: {gbr_mape}%")
print(f"Explained Variance Score: {gbr_explained_variance}")
print(f"Median Absolute Error: {gbr_median_ae}")


# In[48]:


# 1. Sensitivity Analysis: This could involve changing hyperparameters or assumptions and seeing how they affect the model's performance

from sklearn.model_selection import GridSearchCV

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    # Add other parameters you want to tune
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='neg_mean_squared_error', 
                           verbose=1)

# Perform the grid search on the scaled features and target
grid_search.fit(X_scaled, y)

# Output the best parameters
print("Best parameters found: ", grid_search.best_params_)


# In[49]:


2# Advanced Visualizations:
# Create more sophisticated visualizations, such as a 3D plot showing relationships between three variables or an interactive chart.

from mpl_toolkits.mplot3d import Axes3D

# Example of a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Assume 'feature1', 'feature2' are features, and 'target' is what you're predicting
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], y)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.show()


# In[50]:


#Robustness Check via Time-based Cross-validation

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from math import sqrt

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit model (example with Random Forest)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    # Predict and calculate RMSE
    predictions = rf_regressor.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    rmse_scores.append(rmse)

# Average RMSE over the splits
print(f"Average RMSE: {sum(rmse_scores) / len(rmse_scores)}")


# # Step 10: Final Model Validation and Reporting

# In[51]:


# Final Model Predictions
# Assuming 'gbr_pipeline' or 'rf_pipeline' is chosen final model
# Replace 'final_model' with 'gbr_pipeline' or 'rf_pipeline' according to chosen final model
final_predictions = gbr_pipeline.predict(X_test)  # Use rf_pipeline.predict(X_test) if using Random Forest

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(final_predictions, columns=['Predictions'])

# Save the predictions to a CSV file
predictions_df.to_csv('final_model_predictions.csv', index=False)


# In[52]:


# Code to Automate Model Evaluation of gbr_pipeline and rf_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f'{model_name} Model Evaluation')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}\n')
    
    return mse, mae, r2

# Evaluate or Compare the models and print out the results with model names
rf_mse, rf_mae, rf_r2 = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
gbr_mse, gbr_mae, gbr_r2 = evaluate_model(gbr_pipeline, X_test, y_test, "Gradient Boosting")

# Write the evaluation results to a file
with open('model_evaluation.txt', 'w') as f:
    f.write('Random Forest Model Evaluation\n')
    f.write(f'Mean Squared Error: {rf_mse:.4f}\n')
    f.write(f'Mean Absolute Error: {rf_mae:.4f}\n')
    f.write(f'R^2 Score: {rf_r2:.4f}\n\n')
    
    f.write('Gradient Boosting Model Evaluation\n')
    f.write(f'Mean Squared Error: {gbr_mse:.4f}\n')
    f.write(f'Mean Absolute Error: {gbr_mae:.4f}\n')
    f.write(f'R^2 Score: {gbr_r2:.4f}\n')

    


# In[53]:


# Potential non-linear feature - Historical Volatility
tsla_data['historical_volatility'] = tsla_data['close'].rolling(window=10).std() / tsla_data['close'].rolling(window=10).mean()


# In[54]:


# Example: Sensitivity to 'n_estimators' in RandomForestRegressor
for n_estimators in [50, 100, 150, 200]:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'n_estimators: {n_estimators}, MSE: {mse:.4f}')


# In[55]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert X_train to a numpy array if it's a DataFrame
X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

# Data should be reshaped if you're using LSTM (samples, timesteps, features)
X_train_reshaped = X_train_array.reshape((X_train_array.shape[0], 1, X_train_array.shape[1]))

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(1, X_train_array.shape[1])))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Fit LSTM model
lstm_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32)


# In[56]:


# Compile results into a DataFrame
results_summary = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Ensemble'],
    'MSE': [rf_mse, gbr_mse, ensemble_mse],
    'MAE': [rf_mae, gbr_mae, None],  # Assuming you have these values calculated
    'R2': [rf_r2, gbr_r2, None]      # Assuming you have these values calculated
})

# Export summary to CSV
results_summary.to_csv('model_results_summary.csv', index=False)


# In[57]:


import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore', category=UserWarning)

# Assuming 'tsla_data' is your DataFrame which includes the 'date' column and is already loaded
dates = tsla_data['date']  # Extract dates from the tsla_data DataFrame

# Assuming 'X_scaled' and 'y' are already defined as scaled features and target arrays
column_names = ['open', 'high', 'low', 'close', 'volume', 'rsi_7', 'rsi_14',
                'cci_7', 'cci_14', 'sma_50', 'ema_50', 'sma_100', 'ema_100', 
                'macd', 'bollinger', 'TrueRange', 'atr_7', 'atr_14']

X_scaled_df = pd.DataFrame(X_scaled, columns=column_names)
y_series = pd.Series(y)

# Add date information back to your DataFrame
X_scaled_df['date'] = pd.to_datetime(dates)  # Convert dates to datetime if not already

# Define the evaluation function
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, mae, r2

# Define time periods for evaluation
time_periods = [
    ('2019-01-01', '2020-01-01'),
    ('2020-01-01', '2021-01-01')
]

models = {
    'Random Forest': rf_pipeline,
    'Gradient Boosting': gbr_pipeline
}

# Evaluate models over each period
for model_name, model_pipeline in models.items():
    print(f"Evaluating {model_name}")
    for start_date, end_date in time_periods:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (X_scaled_df['date'] >= start_date) & (X_scaled_df['date'] < end_date)
        X_period = X_scaled_df.loc[mask].drop('date', axis=1)  # Drop date before passing to model
        y_period = y_series[X_scaled_df.index[mask]]  # Use same mask index to ensure alignment

        mse, mae, r2 = evaluate_model(model_pipeline, X_period, y_period)
        print(f"Performance from {start_date.date()} to {end_date.date()}: MSE={mse}, MAE={mae}, R2={r2}")


# In[58]:


# Predict on test data
rf_predictions = rf_regressor.predict(X_test)
gbr_predictions = gbr_regressor.predict(X_test)

# Prepare a DataFrame for the comparison
results_df = pd.DataFrame({
    'Date': tsla_data.loc[y_test.index, 'date'],
    'Actual Values': y_test,
    'RF Predictions': rf_predictions,
    'GBR Predictions': gbr_predictions
})

# Reset index for clarity
results_df.reset_index(drop=True, inplace=True)

# Display or save the results
print(results_df.head())
results_df.to_csv('model_comparisons.csv', index=False)


# In[59]:


# Data
data = {
    'Date': ['2022-05-03', '2022-05-04', '2022-05-05', '2022-05-06', '2022-05-07'],
    'Actual Values': [303.083344, 317.540009, 291.093323, 288.549988, 262.369995],
    'RF Predictions': [302.111128, 315.563427, 290.973731, 287.646227, 262.476428],
    'GBR Predictions': [310.894533, 308.035023, 289.434568, 286.359899, 262.979408]
}

# Create a DataFrame
results_df = pd.DataFrame(data)

# Calculate Error columns
results_df['Error (RF)'] = abs(results_df['Actual Values'] - results_df['RF Predictions'])
results_df['Error (GBR)'] = abs(results_df['Actual Values'] - results_df['GBR Predictions'])

# Display the DataFrame
print(results_df)

# If you want to export this table to a CSV file
results_df.to_csv('model_predictions_comparison.csv', index=False)


# In[60]:


from IPython.display import Image
Image(filename='DFD_Diagram.png')


# In[ ]:




