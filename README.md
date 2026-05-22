# 🚀 Live Demo: https://huggingface.co/spaces/ajaysah-ai/telecom-churn-prediction

# 1. Project Title & Description - Telecom Customer Churn Prediction
    ~ The main objective of this project is to find out which customers are likely to leave the telecom service (Churn), so that the company can offers to stop them.

# 2. Workflow
    ~ Handling Missing Values:
        Column TotalCharges was object dtype and I converted it into number with `errors='coerce'`.
        After conversion, TotalChages had 11 NaN values. So I filled it with `df['TotalCharges'].median()`.

    ~ Detecting Outliers:
        I detected outliers on numerical columns using `IQR` technique.
        In dataset, was zeror outlier.

    ~ Mapping SeniorCitizen Column:
        Because SeniorCitizen column was already encoded through `OrdinalEncoder` or `LabelEncoder`.
        I mapped it as `{1: "Yes", 0:"No"}`.

    ~ Feature Importance:
        I checked relation between features(categorical) vs target(categorical) using `chi2_contingency` and relation between features(numerical) vs target(categorical) using `mannwhitneyu`. Also I was appending related columns using `features.append(col)`.

    ~ Feature Selection:
        I selected features using `df[features]` and target using `df['Churn]`.

    ~ Spliting Features & Target into Train-and-Test:
        I Splited using `train_test_split(X, y, test_size=0.2, random_state=42)`. So I got my training and testing datasets.

    
    ~ Feature and Target Transformation:
        I applied StandardScaler on numerical feature, OneHotEncoder on categorical features and LabelEncoder on target.
    
    ~ Logistic Regression:
        First I trained LogisticRegression model without hyperparameter tuning and I found Train Accuracy = 0.8035 and Test Accuracy = 0.8211.
    
        Secondly I trained LogisticRegression model with hyperparameter tuning using GridSearchCV and I found low accuracy from the previous one (Train Accuracy = 0.8024 and Test Accuracy = 0.8204).

    ~ Random Forest Classifier:
        I trained RandomForestClassifer model with hyperparameter tuning using RandomizedSearchCV and I got high traing accuracy and low test accuracy from the previous (Train Accuracy = 0.8571 and Test Accuracy = 0.8112).

    ~ Gradient Boosting Classifier:
        I trained GradientBoostingClassifier with hyperparameter tuning using RandomizedSearchCV and I got Low training accuracy and high test accuracy than RandomForestClassifier (train Accuracy = 0.8223 and Test Accuracy = 0.8154).

    ~ Neural Network using TensorFlow.Keras:
        I build NeuralNetwork using `tensorflow.keras` and give layers as:
            Input layer - Dense(28, activation='relu', input_shape=(X_train.shape[1],)).
            To avoid overfitting I used `Dropout(0.2)`.
            Hidden layer1 - Dense(16, activation='relu')
            Hidden layer2 - Dense(8, activation='relu')
            Output layer - Dense(1, activation='sigmoid)

            This is the summary of my NeuralNetwork model:
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
            ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
            │ dense (Dense)                        │ (None, 28)                  │             812 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ dropout (Dropout)                    │ (None, 28)                  │               0 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ dense_1 (Dense)                      │ (None, 16)                  │             464 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ dense_2 (Dense)                      │ (None, 8)                   │             136 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ dense_3 (Dense)                      │ (None, 1)                   │               9 │
            └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

        I used Adam optimizer with learning rate 0.0001 using `Adam(learning_rate=0.0001)` and also I compile the model using `loss='binary_crossentropy'` and `metrics=['accuracy']`.


        After some steps... I trained the model and I got low train and test accuracy (Train Accuracy = 0.8039 and Test Accuracy = 0.8141).
    
# 3. Results and comparison
    Seeing this We can't conclude that which model is best, We need to check classification_report and confusion_matrix.

    **MODEL**                                           **Train Accuracy**     **Test Accuracy**     **True Positive**     **False Negative**
    LogisticRegression                                        0.8035                 0.8211                 222                   151
    LogisticRegression (GridSearchCV)                         0.8024                 0.8204                 221                   152
    RandomForestClassifier (RandomizedSearchCV)               0.8571                 0.8112                 189                   184
    GradientBoostingClassifier (RandomizedSearchCV)           0.8223                 0.8154                 217                   156
    NeuralNetwork (tensorflow.keras)                          0.8039                 0.8141                 219                   154

    Seeing this, I am choosing my best model is `LogisticRegression` for "Telecom Customer Churm Predict" dataset.
    Do you want to suggest me something?

# 4. Saving the Logitic Regression model
    ~ After Results and comparison, I decided to select Logistic Regression as the best model for `telecom customer churn prediction`.
    ~ Using `sklearn.pipeline`, I combined One-Hot Encoding, Standard Scaler and the Logistic Regression in one pipeline.
    ~ After I saved the mode as `telecom_customer_churn_logistic_model.pkl` using library pickle.

# 5. Wraping with FastAPI
    ~ I created a new file as `main.py` for reload my saved model and wraping it with FastAPI.
    ~ In this, I used POST method to give results and this main.py give results as form of JSON.
    ~ Gives Multiple results in the form of ```[list]```.

# 6. requirements.txt
    ~ In `requirements.txt`, I written librarie's name (e.g., fastapi, uvicorn, pandas, numpy, scikit-learn) for dependencies.

# 7. Dockerize
    ~ I used `Docker` to contain all my files and model.
    ~ Also I used for deployment.

# 8. Summary
    ~ This project contains `end-to-end` AI/ML pipeline like real industry work flow.
    ~ Defining problem, Data Preprocessing, Baseline Model, Hyperparameter Tuning, Model Evaluation, Model Packaging, and Deployment.

# Information about me
    ~ Full Name: Ajay Sah
    ~ Domain: AI/ML Engineering
    ~ Gmail: ajaysah.jobs.20@gmail.com
    ~ Address: Jamshedpur, Jharkhand, India
