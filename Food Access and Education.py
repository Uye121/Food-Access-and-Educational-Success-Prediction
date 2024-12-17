# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Healthy Food Access and Educational Success Prediction

# %% [markdown]
# The datasets used in this project comes from the Stanford Education Data Archive (SEDA) and USDA. 

# %% [markdown]
# # Fetching and Loading Data

# %%
import os
import numpy as np

DATA_PATH = os.path.join("datasets", "education")
SEDA_URL = DATA_PATH + "/seda_county_long_cs_5.0.csv"
USDA_FOOD_ENV_URL = os.path.join(DATA_PATH, "FoodEnvironmentAtlas", "StateAndCountyData.csv")
USDA_SNAP_ENV_URL = DATA_PATH + "/FoodEnvironmentAtlas.xls"

# %%
import pandas as pd

def load_data(file_path, sheet_name=None):
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xls':
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        print("The file is neither an Excel file nor a CSV file.")

def merge_education_and_food_data(seda_df, usda_df):
    return pd.merge(seda_df, usda_df, left_on='sedacounty', right_on='FIPS', how='inner')


# %%
seda_df = load_data(SEDA_URL)
usda_food_df = load_data(USDA_FOOD_ENV_URL)

seda_df.head(), usda_food_df.head()

# %%
usda_snap_df = load_data(USDA_SNAP_ENV_URL, 'ASSISTANCE')

usda_snap_df.info()

# %% [markdown]
# # Data Exploration

# %% [markdown]
# SEDA dataset note:
# - rla: Reading Language Arts.
# - cs_mn_all: average composite score across grade.
# - cs_mn_se_all: standard error of the composite score.
#
# USDA dataset note:
# - PCT_LACCESS_POP15: Percentage of the population with low access to healthy food in 2015.
# - PCT_LACCESS_LOWI15: Percentage of low-income individuals with low access to healthy food in 2015.
# - PCT_LACCESS_SNAP15: Percentage of SNAP recipients with low access to healthy food in 2015.
# - PCT_LACCESS_CHILD15: Percentage of children with low access to healthy food in 2015.
# - PCT_FREE_LUNCH15: Percentage of children with free lunch in 2015.
# - PCT_REDUCED_LUNCH10: Percentage of children with reduced lunch in 2015.
# - PCT_LACCESS_POP10: Percentage of the population with low access to healthy food in 2010.
# - PCT_LACCESS_LOWI10: Percentage of low-income individuals with low access to healthy food in 2010.
# - PCT_LACCESS_SNAP10: Percentage of SNAP recipients with low access to healthy food in 2010.
# - PCT_LACCESS_CHILD10: Percentage of children with low access to healthy food in 2010.
# - PCT_FREE_LUNCH10: Percentage of children with free lunch in 2010.
# - PCT_REDUCED_LUNCH10: Percentage of children with reduced lunch in 2010.

# %%
print(f'Before filter:\nSEDA data size - {seda_df.size}\nUSDA data size - {usda_food_df.size}')
print('-' * 100)

# Create separate dataframes for data from 2010 and from 2015.
seda15_df = seda_df[(seda_df['year'] == 2015) & (seda_df['subject'] == 'rla')]
seda10_df = seda_df[(seda_df['year'] == 2010) & (seda_df['subject'] == 'rla')]

seda15_df = seda15_df[["sedacounty", "grade", "year", "cs_mn_all", "cs_mn_se_all"]]
seda10_df = seda10_df[["sedacounty", "grade", "year", "cs_mn_all", "cs_mn_se_all"]]

usda_food15_df = usda_food_df[usda_food_df['Variable_Code'].str.endswith('15', na=False)]
usda_food10_df = usda_food_df[usda_food_df['Variable_Code'].str.endswith('10', na=False)]

usda_snap10_df = usda_snap_df[['FIPS', 'PCT_FREE_LUNCH10', 'PCT_REDUCED_LUNCH10']].copy()
usda_snap15_df = usda_snap_df[['FIPS', 'PCT_FREE_LUNCH15', 'PCT_REDUCED_LUNCH15']].copy()

# Combine percent free and percent reduced lunch data
usda_snap10_df['PCT_FREE_REDUCED_LUNCH10'] = usda_snap10_df['PCT_FREE_LUNCH10'] + usda_snap10_df['PCT_REDUCED_LUNCH10']
usda_snap10_df.drop(columns=['PCT_FREE_LUNCH10', 'PCT_REDUCED_LUNCH10'], inplace=True)

usda_snap15_df['PCT_FREE_REDUCED_LUNCH15'] = usda_snap15_df['PCT_FREE_LUNCH15'] + usda_snap15_df['PCT_REDUCED_LUNCH15']
usda_snap15_df.drop(columns=['PCT_FREE_LUNCH15', 'PCT_REDUCED_LUNCH15'], inplace=True)

usda_food10_df = pd.merge(usda_food10_df, usda_snap10_df, on='FIPS', how='inner')
usda_food15_df = pd.merge(usda_food15_df, usda_snap15_df, on='FIPS', how='inner')

usda_food15_df = usda_food15_df[usda_food15_df['Variable_Code'].isin([
    'PCT_LACCESS_POP15',
    'PCT_LACCESS_LOWI15',
    'PCT_LACCESS_SNAP15',
    'PCT_LACCESS_CHILD15'
])]
usda_food10_df = usda_food10_df[usda_food10_df['Variable_Code'].isin([
    'PCT_LACCESS_POP10',
    'PCT_LACCESS_LOWI10',
    'PCT_LACCESS_SNAP10',
    'PCT_LACCESS_CHILD10'
])]

print(f'After filter:\nSEDA data size - {seda10_df.size}\nUSDA data size - {usda_food10_df.size}')
usda_food10_df.head()

# %%
merged10_df = merge_education_and_food_data(seda10_df, usda_food10_df)
merged15_df = merge_education_and_food_data(seda15_df, usda_food15_df)

merged10_df.info()

# %%
merged10_df.info()

# %%
merged10_df = merged10_df.drop(columns=['year', 'sedacounty', 'FIPS', 'cs_mn_se_all'], axis=1, errors='ignore')
merged15_df = merged15_df.drop(columns=['year', 'sedacounty', 'FIPS', 'cs_mn_se_all'], axis=1, errors='ignore')

# %%
reshaped10_df = merged10_df.copy()
reshaped15_df = merged15_df.copy()

# Reshape table to have the population data label as its own column with their
# respective values under them.
reshaped10_df = reshaped10_df.pivot(index=[
    'cs_mn_all',
    'grade',
    'State',
    'County',
    'PCT_FREE_REDUCED_LUNCH10'
], columns='Variable_Code', values='Value').reset_index()
reshaped10_df.sort_values(by=['State', 'County', 'grade'])
reshaped10_df = reshaped10_df.drop(columns=['State', 'County'], axis=1, errors='ignore')
reshaped10_df.dropna(subset=['PCT_FREE_REDUCED_LUNCH10'], inplace=True)

reshaped15_df = reshaped15_df.pivot(index=[
    'cs_mn_all',
    'grade',
    'State',
    'County',
    'PCT_FREE_REDUCED_LUNCH15'
], columns='Variable_Code', values='Value').reset_index()
reshaped15_df.sort_values(by=['State', 'County', 'grade'])
reshaped15_df = reshaped15_df.drop(columns=['State', 'County'], axis=1, errors='ignore')
reshaped15_df.dropna(subset=['PCT_FREE_REDUCED_LUNCH15'], inplace=True)


reshaped10_df.head(10)

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
reshaped10_df.hist(bins=40, figsize=(20,15))
plt.show()

# %%
import seaborn as sns

corr = reshaped10_df.corr(numeric_only=True)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

# %% [markdown]
# ## Data Observation

# %% [markdown]
# Based on the histogram, we can see that the standardized test scores are normally distributed as expected due to the way the data was normalized by SEDA beforehand. The test scores came from a similar number of students across the grade level (3rd - 8th). The shape of the graph for child with low access to healthy food (PCT_LACCESS_CHILD10) and population with low access to healthy food (PCT_LACCESS_POP10) is roughly similar, except the percentage for the child is lower than that of the population. This makes sense since children with low access to food are probably also included in the population with low access to food. What is surprising is that there are around 250 counties with close to 100% of low access to healthy food, yet there isn’t any county with close to 100% low-income individuals with low access to healthy food. As for the correlation matrix, there appears to be low inverse correlation between the test scores and low access to healthy food.

# %% [markdown]
# ## Check Feature Variance Via PCA

# %%
from sklearn.preprocessing import StandardScaler

X = reshaped10_df[['PCT_FREE_REDUCED_LUNCH10', 'PCT_LACCESS_CHILD10', 'PCT_LACCESS_POP10', 'PCT_LACCESS_LOWI10']]
y = reshaped10_df['cs_mn_all']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

cumulative_variance = pca.explained_variance_ratio_.cumsum()
print("Cumulative Explained Variance:", cumulative_variance)

# %% [markdown]
# PCA indicated that the first principal component (PC) contributed 70.8% of the variance and the second PC contributed 26.1% of the variance, which sums up to roughly 97% of the overall variance. This warrant for using PCA to reduce the data dimensionality to 2. Although the third PC contributed little to the overall variance, it is included to get a better sense of the data.

# %% [markdown]
# # Testing Different Models

# %% [markdown]
# ## Predicting Using Linear Regression

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=reshaped10_df['grade'], random_state=42
)

# %%
# Verify split
train_dist = reshaped10_df.loc[X_train.index, 'grade'].value_counts(normalize=True)
test_dist = reshaped10_df.loc[X_test.index, 'grade'].value_counts(normalize=True)
train_dist, test_dist

# %%
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([
    ('st_scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('linear_regression', LinearRegression())
])

pipe_lr.fit(X_train, y_train)
y_train_pred = pipe_lr.predict(X_train)
y_test_pred = pipe_lr.predict(X_test)


# %%
def plot_predicted_vs_actual(y_test, y_pred, title):
    x_range = [min(y_test), max(y_test)]
    y_range = [min(y_test), max(y_test)]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot(x_range, y_range, color='red', linestyle='--', label='Perfect Fit Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
plot_predicted_vs_actual(
    y_test, y_test_pred, 'Predicted vs. Actual Values (2010) - Linear Regression'
)

# %%
from sklearn.metrics import mean_squared_error, r2_score

def display_scores(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(rmse)
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')


# %%
print("Training scores:")
display_scores(y_train, y_train_pred)

# %%
print("Testing scores:")
display_scores(y_test, y_test_pred)

# %% [markdown]
# The default hyperparameters were used when training the model using Linear Regression. The dataset is not manipulated in any ways, other than removing NaN rows and standardizing using standard scaler, we can leave the fit_intercept as making no assumption on the data. Computation time isn’t that important for the size of the dataset and number of features, so n_jobs can remain default. The hyperparameters are as follows:
# - fit_intercept: True, default value
# - n_jobs: None, default value
#
# Linear Regression is first used to test whether the dataset have linear relationships. While the root mean squared error (RMSE) and r-squared scores suggest the model is neither overfitting nor underfitting due to their closeness, it is still a poor predictive model due to the low r-squared score of less than 0.5. When graphing the predicted vs actual values, the points are scattered around the perfect fit line with a couple of outliers. If the model is good, the points should be closely around the perfect fit line.
#

# %% [markdown]
# ### Test Model Against 2015 Data

# %%
# Rename column to see how well model generalizes on data from 2015
reshaped15_df.rename(columns={
    'PCT_FREE_REDUCED_LUNCH15': 'PCT_FREE_REDUCED_LUNCH10',
    'PCT_LACCESS_CHILD15': 'PCT_LACCESS_CHILD10',
    'PCT_LACCESS_LOWI15': 'PCT_LACCESS_LOWI10',
    'PCT_LACCESS_POP15': 'PCT_LACCESS_POP10',
    'PCT_LACCESS_SNAP15': 'PCT_LACCESS_SNAP10'
}, inplace=True)
X15 = reshaped15_df[['PCT_FREE_REDUCED_LUNCH10', 'PCT_LACCESS_CHILD10', 'PCT_LACCESS_POP10', 'PCT_LACCESS_LOWI10']].copy()			

y15 = reshaped15_df['cs_mn_all']
_, X15_test, _, y15_test = train_test_split(
    X15, y15, test_size=0.2, stratify=reshaped15_df['grade'], random_state=42
)

y15_pred = pipe_lr.predict(X15_test)

# %%
plot_predicted_vs_actual(
    y15_test, y15_pred, 'Predicted vs. Actual Values (2015) - Linear Regression'
)

# %%
display_scores(y15_test, y15_pred)

# %% [markdown]
# To further examine how well the model generalizes against a completely different dataset, the model is tested against the 2015 dataset. Unexpectedly, it performed much worse with points scattering all over the place. The RMSE and r-squared scores are 0.23 and 0.272 respectively. This suggest that the model is making better predictions using the 2015 dataset but less effective in explaining the variance in the dataset. It is interesting that the model is a better fit for the 2015 data since the economic factors between the two years differ, so the relationships between the features might change. Overall, based on the result, it seems that the dataset has a non-linear relationship, and Linear Regression would not be enough to create a good enough model for it.

# %% [markdown]
# ## Predicting Using Random Forest

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
pipe_rf = Pipeline([
    ('pca', PCA(n_components=3)),
    ('rf_regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipe_rf.fit(X_train, y_train)
y_train_pred = pipe_rf.predict(X_train)
y_test_pred = pipe_rf.predict(X_test)

# %%
plot_predicted_vs_actual(
    y_test, y_test_pred, 'Predicted vs. Actual Values (2010) - Random Forest'
)

# %%
print("Training scores:")
display_scores(y_train, y_train_pred)

# %%
print("Testing scores:")
display_scores(y_test, y_test_pred)

# %% [markdown]
# Most of the hyperparameter values used are the default value to get an initial feel of how the model performs before diving deeper and performing parameter tuning. The hyperparameters used for the Random Forest model are as follows:
# - n_estimators = 100, default value
# - max_depth = None, default value
# - min_samples_split = 2, default value
# - min_samples_leaf = 1, default value
# - min_weight_fraction_leaffloat = 0.0, default value
# - max_features = sqrt, default value
# - max_leaf_nodes = None, default value
# - random_state = 42
# - class_weight = None, default value
#
# Surprisingly, the model created using Random Forest performed much better than the model created by Linear Regression, though it is still lacking. When the predicted values and actual values are plotted against each other, they are more closely aligned to the perfect fit line, meaning it is better at predicting the result than using Linear Regression. A 0.683 r-squared score suggest that the model captured a significant portion of the data’s variability despite using default hyperparameters. The lower r-squared score and higher RMSE score (training vs testing) suggest that the model is overfitting. The model should improve after tuning the hyperparameters.

# %% [markdown]
# ### Test Model Against 2015 Data

# %%
y15_pred = pipe_rf.predict(X15_test)

# %%
plot_predicted_vs_actual(
    y15_test, y15_pred, 'Predicted vs. Actual Values (2015) - Random Forest'
)

# %%
display_scores(y15_test, y15_pred)

# %% [markdown]
# When tested against the 2015 dataset, it performed with a similar RMSE of 0.48 and a significantly worse r-squared score of 0.272. The r-squared score is about as bad as that of the linear regression model on the 2015 dataset. As explained previously, the much worse scores might be due to economic factors of the different fiscal years affecting the features differently. Hypertuning the hyperparameters might result in a better predictive model for the 2010 dataset, but not much for the 2015 dataset.
#

# %% [markdown]
# ## Detecting Abnormal Districts Using One-Class SVM

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
from sklearn.svm import OneClassSVM

gamma = nu = 0.1
oc_svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
oc_svm.fit(X_train_scaled)

y_pred = oc_svm.predict(X_test_scaled)

# %%
scores = oc_svm.decision_function(X_test_scaled)

# %%
plt.hist(scores, bins=50)
plt.title('One Class SVM Decision Scores Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# %%
outlier_percent = (y_pred == -1).mean() * 100
print(f'Percentage outliers: {outlier_percent:.2f}%')

# %% [markdown]
# Since the One-Class SVM does not help with the business objective of this project, default values or very small numbers are used for the hyperparameters. The hyperparameters used are as follows:
# - kernel = rbf, default value
# - degree = 3, default value
# - gamma = 0.1
# - coef0 = 0.0, default value
# - nu = 0.1
# - max_iter = -1, default value
#
# The last model used is the One-Class SVM, which is an unsupervised learning model. The use case for this model will differ from the main use case. The One-Class SVM model is used to detect abnormalities in the districts. Interestingly enough, the percentage outliers predicted by the model is 9.85%. The feature histograms do show some level of outliers, but the level predicted by the model indicated that there might be far more outliers. There might be issue with the data collection process or there are indeed a number of counties that are living in what USDA considered low access to healthy food. The validity of the model cannot be determined due to not being a subject matter expert. There is also no information to validate the result of the SVM.

# %% [markdown]
# n_estimators = 100, default value\
# max_depth = None, default value\
# min_samples_splitint = 2, default value\
# min_samples_leaf = 1, default value\
# min_weight_fraction_leaffloat = 0.0, default value\
# max_features = sqrt, default value\
# max_leaf_nodes = None, default value\
# random_state = 42\
# class_weight = None, default value

# %% [markdown]
# ## Hyperparameter Tuning

# %%
from sklearn.model_selection import GridSearchCV

n_estimators = [10, 25, 50, 100, 150]
max_depth = [None, 3, 5, 10, 20]
min_sample_split = [2, 5, 10]
min_samples_leaf = [0.5, 1, 5]
max_features = [1.0, 'sqrt', 'log2']

param_grid = [
    {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_split': min_sample_split,
        'min_samples_leaf': min_samples_leaf
    },
    {
        'bootstrap': [False],
        'n_estimators': n_estimators[:3],
        'max_depth': max_depth[:3],
        'max_features': max_features,
        'min_samples_split': min_sample_split[:2],
        'min_samples_leaf': min_samples_leaf[:2]
    },
]

# %%
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

# %%
grid_search.best_params_

# %%
rf_best_model = grid_search.best_estimator_
rf_best_model

# %%
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# %%
y_train_pred = rf_best_model.predict(X_train)
y_test_pred = rf_best_model.predict(X_test)

print("Training scores:")
display_scores(y_train, y_train_pred)

# %%
print("Testing scores:")
display_scores(y_test, y_test_pred)

# %%
rf_best_model.score(X_test, y_test)

# %% [markdown]
# The GridSearchCV results indicated that the hyperparameters min_samples_split and n_estimators should be adjusted from 2 to 5 and from 100 to 150 respectively. All other hyperparameters remain unchanged. Upon inspecting the testing RMSE and r-squared scores, the best model identified by GridSearchCV scored 0.1439 and 0.6845 respectively. These results are slightly worse than those of the initial model of mostly default hyperparameters, which scored 0.1443 and 0.683 respectively. This is unusual as the hyperparameters of the initial model were tested in GridSearchCV. Compared to the training RMSE score of 0.1124 and r-squared score of 0.8198, the lower r-squared score and higher RMSE score suggest that the model is overfitting.

# %% [markdown]
# ## Modeling with Stacking

# %%
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR

# %%
estimators = [
    ('rf', rf_best_model),
    ('lr', LinearRegression()),
    ('svr', SVR(kernel='rbf'))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

stacking.fit(X_train, y_train)
y_train_pred = stacking.predict(X_train)
y_test_pred = stacking.predict(X_test)

# %%
plot_predicted_vs_actual(
    y_test, y_test_pred, 'Predicted vs. Actual Values (2010) - Stacking'
)

# %%
print("Training scores:")
display_scores(y_train, y_train_pred)

# %%
print("Testing scores:")
display_scores(y_test, y_test_pred)

# %% [markdown]
# Stacking improved the testing RMSE and r-squared scores slightly with the values of 0.1428 and 0.6895. The training RMSE and r-squared scores are 0.1124 and 0.8198 respectively, which indicate the model is still overfitting. In the future, I will try using a simpler model that is something in between Linear Regression and RandomForest.

# %% [markdown]
# ## Testing with Other Models

# %%
from sklearn.tree import DecisionTreeRegressor

# %%
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# %%
display_scores(y_train, y_train_pred)

# %%
display_scores(y_test, y_test_pred)

# %%
from sklearn.preprocessing import PolynomialFeatures

# %%
pipe_poly = Pipeline([
    ('poly', PolynomialFeatures()),
    ('st_scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('linear_regression', LinearRegression())
])

pipe_poly.fit(X_train, y_train)
y_train_pred = pipe_poly.predict(X_train)
y_test_pred = pipe_poly.predict(X_test)

# %%
display_scores(y_train, y_train_pred)

# %%
display_scores(y_test, y_test_pred)

# %%
pipe_svr = Pipeline([
    ('st_scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('svr', SVR(kernel='rbf'))
])

pipe_svr.fit(X_train, y_train)
y_train_pred = pipe_svr.predict(X_train)
y_test_pred = pipe_svr.predict(X_test)

# %%
display_scores(y_train, y_train_pred)

# %%
display_scores(y_test, y_test_pred)

# %%
pipe_svr.score(X_test, y_test)

# %%
