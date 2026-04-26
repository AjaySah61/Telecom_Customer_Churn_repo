from xml.etree.ElementInclude import include

import pandas as pd, numpy as np
# Load Dataset
df = pd.read_csv(r"C:\Users\RAMAN KUMAR\OneDrive\Desktop\Ajay\AI_COURSE\SESSION 9 (Model Tuning and Optimization)\Day 7 (Optimization Project - Building and Tuning a Final Model)\Telco-Customer-Churn.csv")
print(df.head())

# Dataset information
print(f"\n\n{df.info()}")

# Convert dtype of column TotalCharges into numerical
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"\n\n{df.info()}")

# Check duplicates
print(f"\n\n{df.duplicated().sum()}")

# Handle missing values of column TotalCharges
print("\n\n")
print(df[pd.isnull(df['TotalCharges'])==True])
df.fillna({"TotalCharges": df['TotalCharges'].median()}, inplace=True)
print(f"\n\n{df.info()}")

# Outliers detection
for col in df.select_dtypes('number').columns:
    if col != 'SeniorCitizen':
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lowerFence = q1 - 1.5*iqr
        upperFence = q3 + 1.5*iqr
        outliers = df.loc[(df[col]<lowerFence) | (df[col]>upperFence)]
        outlier_percentage = round((len(outliers) / len(df[col]) * 100), 4)
        if outlier_percentage == 0.0:
            print(f"{col} has zero outlier!\n")
        elif outlier_percentage < 0.1:
            print(f"{col} has unwanted outliers. Can drop it!\n")
        elif outlier_percentage < 5.0:
            print(f"{col} has outliers but It may be useful for data or may be rare events!\n")
        else:
            print(f"{col} has very large outliers!\n")

# Mapping SeniorCitizen Column
print("\n\n")
print(f"{df['customerID'].value_counts()}")
print(f"\n\n{df['SeniorCitizen'].value_counts()}")

seniorCitizenMapping = {
    1: 'Yes',
    0: 'No'
}
df['SeniorCitizen'] = df['SeniorCitizen'].map(seniorCitizenMapping).astype('object')
print(f"\n\n{df['SeniorCitizen'].value_counts()}")

# Feature Importance
from scipy.stats import chi2_contingency, mannwhitneyu
features = []
print(f"\n\nCategorical vs target(Categorical) relation....")
for col in df.select_dtypes('object').columns:
    if col != 'Churn':
        table = pd.crosstab(df[col], df['Churn'])
        chi2, p_value, dof, expected = chi2_contingency(table)
        if p_value < 0.05:
            print(f"\n{col}: Relation Exist!")
            features.append(col)
        else:
            print(f"\n{col}: Weak / no strong relation!")

print(f"\n\nNumrical vs target(Categorical) relation....")
for col in df.select_dtypes('number').columns:
    yes = df[df['Churn']=='Yes'][col]
    no = df[df['Churn']=='No'][col]
    f_stat, p_value = mannwhitneyu(yes, no)
    if p_value < 0.05:
        print(f"\n{col}: Relation Exist!")
        features.append(col)
    else:
        print(f"\n{col}: Weak / no strong relation!")

print(f"\n\nFinal Features: {features}")

# Define features and target
X, y = df[features], df['Churn']
print(f"\n\nFeatures: {X.columns}")
print(f"\nTarget: \n{y.value_counts()}")

# Split features and target into train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature and target transformation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
preprossed = ColumnTransformer(
    transformers= [
        ('num', StandardScaler(), X.select_dtypes('number').columns.tolist()),
        ('obj', OneHotEncoder(drop='first'), X.select_dtypes('object').columns.tolist()),
    ],
    remainder='passthrough'
)
X_train = preprossed.fit_transform(X_train)
X_test = preprossed.transform(X_test)

y_train, y_test = LabelEncoder().fit_transform(y_train), LabelEncoder().fit_transform(y_test)

print(f"\n\nFeatures and Target are successfully transformed!")
print(f"Training Shape (Features): {X_train.shape} | Testing Shape (Features): {X_test.shape}")

# Training simple Logistic Regression model 
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
log_y_pred = log_model.predict(X_test)
print(f"\n\nTrain Score (Logistic): {log_model.score(X_train, y_train)}")
print(f"Test Score (Logistic): {log_model.score(X_test, y_test)}")

# Hyperparameter Tuning 
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# param_grid = {
#     "C": [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#     "penalty": ["l1", "l2"],
#     "solver": ["liblinear", "saga"]
# }
# grid_search = GridSearchCV(
#     estimator=LogisticRegression(random_state=42),
#     param_grid=param_grid,
#     scoring='accuracy',
#     n_jobs=-1,
#     cv=5
# )
# grid_search.fit(X_train, y_train)
# best_model_log = grid_search.best_estimator_
# grid_log_y_pred = best_model_log.predict(X_test)
# print(f"\n\nTrain Score (Logistic - GridSearchCV): {best_model_log.score(X_train, y_train)}")
# print(f"Test Score (Logistic - GridSearchCV): {best_model_log.score(X_test, y_test)}")


# # Training Random Forest Classifier with Hyperparameter Tuning
# from sklearn.ensemble import RandomForestClassifier
# param_dist = {
#     "n_estimators": np.arange(50, 500, 30, dtype=int),
#     "max_depth": [None] + list(np.arange(2, 25, 2, dtype=int)),
#     "min_samples_split": np.arange(2, 15, 1, dtype=int),
#     "min_samples_leaf": np.arange(1, 8, 1, dtype=int),
#     "max_features": ["sqrt", "log2", None],
#     "class_weight": [None, "balanced"]
# }
# random_search = RandomizedSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_distributions=param_dist,
#     scoring='accuracy',
#     n_jobs=-1,
#     cv=5,
#     random_state=42
# )
# random_search.fit(X_train, y_train)
# best_model_rf = random_search.best_estimator_
# random_rf_y_pred = best_model_rf.predict(X_test)
# print(f"\n\nTrain Score (Random Forest - RandomizedSearchCV): {best_model_rf.score(X_train, y_train)}")
# print(f"Test Score (Random Forest - RandomizedSearchCV): {best_model_rf.score(X_test, y_test)}")

# # Training Gradient Boosting model with Hyperparameter Tuning
# from sklearn.ensemble import GradientBoostingClassifier
# param_dist= {
#     'n_estimators': np.arange(50, 400, 30, dtype=int),
#     'learning_rate': np.arange(0.001, 0.3, 0.03, dtype=float),
#     'max_depth': np.arange(2, 8, 1, dtype=int),
#     'subsample': np.arange(0.5, 0.8, 0.03, dtype=float),
#     'min_samples_leaf': np.arange(1, 10, 1, dtype=int),
#     'max_features': ['sqrt', 'log2', None]
# }
# random_search = RandomizedSearchCV(
#     estimator=GradientBoostingClassifier(random_state=42),
#     param_distributions=param_dist,
#     scoring='accuracy',
#     n_jobs=-1,
#     cv=5,
#     random_state=42
# )
# random_search.fit(X_train, y_train)
# best_model_gb = random_search.best_estimator_
# random_gb_y_pred = best_model_gb.predict(X_test)
# print(f"\n\nTrain Score (Gradient Boosting - RandomizedSearchCV): {best_model_gb.score(X_train, y_train)}")
# print(f"Test Score (Gradient Boosting - RandomizedSearchCV): {best_model_gb.score(X_test, y_test)}")


# # Neural Network with TensorFlow
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam

# model = Sequential([
#     Dense(28, activation='relu', input_shape=(X_train.shape[1],)), # input layer
#     Dropout(0.2), # To prevent overfitting
#     Dense(16, activation='relu'), # Hidden layer 1
#     Dense(8, activation='relu'), # Hidden layer 2
#     Dense(1, activation='sigmoid') # Output layer
# ])

# optimizer = Adam(learning_rate=0.0001)

# model.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1
# )
# model.summary()
# keras_y_pred = (model.predict(X_test) > 0.5).astype("int32")
# print(keras_y_pred)

# train_loss, train_accuracy = model.evaluate(X_train, y_train)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"\n\nTrain Accuracy (Keras Neural Network): {train_accuracy:.4f}")
# print(f"Test Accuracy (Keras Neural Network): {test_accuracy:.4f}")


# # Choosing best model
# from sklearn.metrics import classification_report, confusion_matrix
# print(f"\n\nClassifiction Report (Logistic Regression):\n{classification_report(y_test, log_y_pred)}")
# print(f"\nClassifiction Report (Logistic Regression with GridSearchCV):\n{classification_report(y_test, grid_log_y_pred)}")
# print(f"\nClassifiction Report (Random Forest Classifier with GridSearchCV):\n{classification_report(y_test, random_rf_y_pred)}")
# print(f"\nClassifiction Report (Gradient Boosting Classifier with GridSearchCV):\n{classification_report(y_test, random_gb_y_pred)}")
# print(f"\nClassifiction Report (Keras Neural Network):\n{classification_report(y_test, keras_y_pred)}")


# print(f"\n\n\nConfusion Matrix (Logistic Regression): \n{confusion_matrix(y_test, log_y_pred)}")
# print(f"\nConfusion Matrix (Logistic Regression with GridSearchCV): \n{confusion_matrix(y_test, grid_log_y_pred)}")
# print(f"\nConfusion Matrix (Random Forest Classifier with RandomizedSearchCV): \n{confusion_matrix(y_test, random_rf_y_pred)}")
# print(f"\nConfusion Matrix (Gradient Boosting Classifier with RandomizedSearchCV): \n{confusion_matrix(y_test, random_gb_y_pred)}")
# print(f"\nConfusion Matrix (Keras Neural Network): \n{confusion_matrix(y_test, keras_y_pred)}")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(steps=[
    ('preprocessed', preprossed),
    ('logistic', LogisticRegression(random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train, y_test = LabelEncoder().fit_transform(y_train), LabelEncoder().fit_transform(y_test)
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
print(f"\n\nTrain Score (Pipeline: {pipeline.score(X_train, y_train)}")
print(f"Test Score (Pipeline): {pipeline.score(X_test, y_test)}")

import pickle
pickle.dump(pipeline, open("telecom_customer_churn_logistic_model.pkl", "wb"))