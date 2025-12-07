# Heart Disease Prediction using Logistic Regression

## Project Overview
This project implements a machine learning model to predict the presence of heart disease in patients based on various medical attributes. The model uses Logistic Regression, a binary classification algorithm, to determine whether a patient has heart disease (target = 1) or not (target = 0).

## Dataset Description
The dataset (`heart_disease_data.csv`) contains 303 patient records with 14 attributes:

### Features:
1. **age**: Age of the patient (years)
2. **sex**: Gender (1 = male, 0 = female)
3. **cp**: Chest pain type (0-3)
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment (0-2)
12. **ca**: Number of major vessels colored by fluoroscopy (0-4)
13. **thal**: Thalassemia (0-3)

### Target Variable:
- **target**: Presence of heart disease (1 = disease, 0 = no disease)

## Dataset Statistics
- **Total Records**: 303 patients
- **Total Features**: 13 (excluding target)
- **Missing Values**: None (all 303 entries are complete)
- **Target Distribution**: Approximately 54.5% patients have heart disease

### Key Statistics:
- Average age: 54.4 years (range: 29-77 years)
- Gender distribution: 68.3% male
- Average resting blood pressure: 131.6 mm Hg
- Average cholesterol: 246.3 mg/dl
- Average maximum heart rate: 149.6 bpm

## Project Workflow

### 1. Data Collection and Processing
```python
heart_data = pd.read_csv("heart_disease_data.csv")
```
- Loaded the dataset using pandas
- Explored data structure with `.head()`, `.info()`, and `.describe()`
- Verified data completeness (no missing values)

### 2. Data Splitting
```python
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]
```
- **Features (X)**: All 13 medical attributes
- **Target (Y)**: Heart disease presence indicator

### 3. Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```
- **Training Set**: 242 samples (80%)
- **Testing Set**: 61 samples (20%)
- **Stratification**: Ensures proportional class distribution in both sets
- **Random State**: Set to 2 for reproducibility

### 4. Model Training
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```
- Algorithm: Logistic Regression (binary classifier)
- Trained on 242 patient records
- **Note**: Convergence warning appeared due to small dataset size (not a critical issue)

### 5. Model Evaluation
```python
train_accuracy = accuracy_score(train_data_prediction, Y_train)
test_accuracy = accuracy_score(test_data_prediction, Y_test)
```

## Results

### Model Performance:
- **Training Accuracy**: 85.12%
- **Testing Accuracy**: 81.97%

### Interpretation:
- The model correctly predicts heart disease presence in approximately 82% of unseen cases
- Small gap between training and testing accuracy (3.15%) indicates good generalization
- No significant overfitting observed
- Performance is reasonable given the small dataset size (303 samples)

## Dependencies
```python
numpy
pandas
scikit-learn
```

### Installation:
```bash
pip install numpy pandas scikit-learn
```

## Usage

### Running the Model:
1. Ensure `heart_disease_data.csv` is in the same directory
2. Run the Jupyter notebook cells sequentially
3. The model will train and display accuracy metrics

### Making Predictions:
```python
# Example: Predict for a new patient
input_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
prediction = model.predict(input_data)
# Output: 0 (no disease) or 1 (disease present)
```

## Model Limitations

1. **Small Dataset**: Only 303 samples may limit model robustness
2. **Convergence Warning**: Indicates the dataset size is minimal for optimal convergence
3. **Feature Scaling**: Not applied (may improve performance)
4. **Model Selection**: Only Logistic Regression tested (other algorithms might perform better)
5. **Feature Engineering**: No advanced feature creation or selection performed

## Potential Improvements

1. **Data Preprocessing**:
   - Apply feature scaling (StandardScaler or MinMaxScaler)
   - Handle potential outliers in cholesterol and blood pressure

2. **Model Enhancements**:
   - Increase `max_iter` parameter to resolve convergence warning
   - Try other algorithms (Random Forest, SVM, Neural Networks)
   - Implement cross-validation for more robust evaluation

3. **Feature Engineering**:
   - Create interaction features
   - Apply feature selection techniques
   - Analyze feature importance

4. **Evaluation Metrics**:
   - Add confusion matrix
   - Calculate precision, recall, and F1-score
   - Generate ROC curve and AUC score

## Clinical Significance
This model can serve as a preliminary screening tool to identify patients at risk of heart disease, enabling early intervention and potentially saving lives. However, it should **not replace professional medical diagnosis** and should be used only as a supportive decision-making tool.

## License
This project is for educational purposes.

## Author
Created as a machine learning practice project for heart disease prediction.

---

**Disclaimer**: This model is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and clinical oversight.