# Housing Price Prediction Using Multiple Linear Regression

## Backward Elimination + Forward Selection and RFE + Backward Elimination

## 1. Project Overview

Housing price prediction is a supervised machine learning regression problem in which the objective is to estimate the selling price of a residential property using its structural and categorical characteristics.

This project uses the `Housing.csv` dataset to develop and evaluate Multiple Linear Regression models using two different feature-selection approaches:

**Approach 1:** Multiple Linear Regression with Backward Elimination and Forward Selection.

**Approach 2:** Multiple Linear Regression with Recursive Feature Elimination (RFE) and Backward Elimination.

Both approaches aim to identify relevant predictors, reduce unnecessary features, and build a regression model capable of predicting house prices.

The project also includes data preprocessing, exploratory data analysis, model evaluation, residual diagnostics, coefficient interpretation, and prediction for new properties.

---

## 2. Problem Statement

To develop a Multiple Linear Regression model that predicts the selling price of a residential property based on its available features.

The project compares two feature-selection strategies to investigate how different predictor-selection methods affect model complexity, interpretability, and prediction performance.

The target variable is `price`, while the remaining variables are considered potential predictors.

---

## 3. Project Objectives

The main objectives are:

- Load and analyse the housing dataset.
- Perform exploratory data analysis and visualisation.
- Identify missing values, duplicate records, and data types.
- Separate the target variable from the input features.
- Encode categorical variables into numerical representations.
- Scale numerical predictors where required.
- Build Multiple Linear Regression models.
- Apply Backward Elimination using statistical p-values.
- Apply Forward Selection using Adjusted R².
- Apply Recursive Feature Elimination (RFE).
- Combine RFE with Backward Elimination.
- Compare the two feature-selection approaches.
- Evaluate model performance using regression metrics.
- Analyse residuals and multicollinearity.
- Predict the selling price of a new house.

---

## 4. Dataset Description

The project uses the `Housing.csv` dataset.

The supplied dataset contains 545 records and 13 columns.

### Dataset attributes

| Feature | Description | Data Type |
|---|---|---|
| `price` | Selling price of the house | Numerical |
| `area` | Area of the house | Numerical |
| `bedrooms` | Number of bedrooms | Numerical |
| `bathrooms` | Number of bathrooms | Numerical |
| `stories` | Number of stories | Numerical |
| `mainroad` | Whether the property has main-road access | Categorical |
| `guestroom` | Whether the property has a guest room | Categorical |
| `basement` | Whether the property has a basement | Categorical |
| `hotwaterheating` | Whether hot-water heating is available | Categorical |
| `airconditioning` | Whether air conditioning is available | Categorical |
| `parking` | Number of parking spaces | Numerical |
| `prefarea` | Whether the property is in a preferred area | Categorical |
| `furnishingstatus` | Furnishing status of the property | Categorical |

**Target variable:** `price`

**Input features:** The remaining 12 columns.

The dataset contains both numerical and categorical predictors. These variables are processed before being used in the regression models.

---

## 5. Technologies Used

The project uses Python and is designed to run in Google Colab.

| Technology or Library | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development and execution environment |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Graph plotting |
| Seaborn | Statistical data visualisation |
| Scikit-learn | Data splitting, scaling, RFE, and evaluation metrics |
| Statsmodels | Ordinary Least Squares regression, p-values, and statistical summaries |
| SciPy | Statistical analysis and diagnostic plots, where required |

### Installation

Google Colab generally provides these libraries by default. For a local Python environment, install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

---

## 6. Methodology

### 6.1 Multiple Linear Regression

Multiple Linear Regression is a supervised learning algorithm used to predict a continuous numerical target using multiple independent variables.

The general regression equation is:

\[
\hat{y}=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
\]

Where:

- \(\hat{y}\) is the predicted house price.
- \(\beta_0\) is the intercept.
- \(x_1,x_2,\ldots,x_p\) are the selected input features.
- \(\beta_1,\beta_2,\ldots,\beta_p\) are the regression coefficients.
- \(p\) is the number of selected predictors.

The regression coefficients estimate the relationship between each predictor and the target, holding the other included predictors constant.

### 6.2 Data Preprocessing

Before training the regression models, the dataset undergoes preprocessing.

The main steps are:

1. Load the dataset from `Housing.csv`.
2. Inspect the dataset structure and descriptive statistics.
3. Check for missing values and duplicate records.
4. Separate `price` from the independent variables.
5. Split the dataset into training and testing sets.
6. Identify numerical and categorical features.
7. Apply one-hot encoding to categorical features.
8. Scale numerical predictors where required.
9. Ensure the training and testing datasets contain matching feature columns.

The scaler and other preprocessing transformations should be fitted on the training data only. The same fitted transformations are then applied to the test data to prevent data leakage.

### 6.3 Categorical Encoding

Regression models require numerical input.

Categorical variables such as `mainroad`, `guestroom`, `airconditioning`, and `furnishingstatus` are converted into numerical indicator columns using one-hot encoding.

Using `drop_first=True` removes one reference category from each encoded categorical group, helping avoid perfect multicollinearity among the full set of indicator columns.

### 6.4 Approach 1: Backward Elimination and Forward Selection

This approach combines two feature-selection techniques.

#### A. Backward Elimination

Backward Elimination begins with all available candidate predictors.

The procedure is:

1. Fit a Multiple Linear Regression model.
2. Obtain the p-value for each predictor from the fitted OLS model.
3. Identify the predictor with the highest p-value.
4. If its p-value exceeds the selected significance threshold, remove that predictor.
5. Refit the regression model.
6. Repeat until the stopping criterion is met.

The project uses a significance level of:

\[
\alpha=0.05
\]

A predictor with a p-value greater than 0.05 is considered for removal under this procedure.

**Advantages:**

- Uses statistical significance to assess individual regression coefficients.
- Starts with the complete candidate feature set.
- Can produce a simpler and more interpretable regression model.

**Limitations:**

- Results may depend on the sample and the relationships between predictors.
- P-values can be affected by multicollinearity and model assumptions.
- Statistical significance does not necessarily imply practical importance.

#### B. Forward Selection

Forward Selection starts with an empty set of predictors.

The procedure is:

1. Begin with no predictors in the model.
2. Evaluate the candidate predictors individually or in combination with those already selected.
3. Add the predictor that gives the best improvement according to the selected criterion.
4. Recalculate the criterion after adding the predictor.
5. Continue until no remaining predictor meets the stopping rule.

This project uses Adjusted R² as the selection criterion.

Adjusted R² measures model fit while accounting for the number of predictors. Unlike ordinary R², it can decrease when a newly added predictor does not improve the model sufficiently.

**Advantages:**

- Builds the model gradually.
- Can reduce the number of unnecessary predictors.
- Uses Adjusted R² to account for model complexity.

**Limitations:**

- Results can depend on the order in which predictors are considered.
- Greedy selection does not guarantee the globally best feature subset.
- A selected predictor may become less useful after other predictors are added.

#### C. Model selection for Approach 1

The feature subsets produced by Backward Elimination and Forward Selection are compared according to the rule implemented in the Python code.

The selected candidate is subsequently evaluated on the held-out test set.

For rigorous evaluation, feature selection and model-choice decisions should be made using training data or cross-validation, not by repeatedly tuning against the final test set.

### 6.5 Approach 2: Recursive Feature Elimination (RFE) and Backward Elimination

This approach combines Recursive Feature Elimination with statistical Backward Elimination.

#### A. Recursive Feature Elimination (RFE)

RFE is a wrapper-based feature-selection technique. It repeatedly trains an estimator and removes the least important predictor according to the estimator's coefficients or feature-importance scores.

For this project, RFE can use a linear regression estimator.

The procedure is:

1. Start with the candidate predictor set.
2. Fit the selected regression estimator.
3. Rank the predictors using the estimator's coefficient magnitudes or feature-importance values.
4. Remove the least important predictor or predictors.
5. Refit the estimator using the remaining predictors.
6. Repeat until the specified number of predictors remains.

The number of features retained is controlled by the RFE configuration.

**Advantages:**

- Can reduce a large predictor set systematically.
- Uses the fitted estimator to rank feature usefulness.
- Can be integrated into a machine-learning pipeline.

**Limitations:**

- The result depends on the chosen estimator and number of retained features.
- Coefficient-based rankings can be affected by feature scaling.
- RFE does not directly use statistical p-values to rank predictors.

#### B. Backward Elimination after RFE

After RFE selects a smaller subset of predictors, Backward Elimination is applied to the reduced feature set.

The procedure is:

1. Perform RFE using the training predictors and target.
2. Retain the predictors selected by RFE.
3. Fit an OLS regression model using those predictors.
4. Examine the predictors' p-values.
5. Remove the predictor with the highest p-value if it exceeds the significance threshold.
6. Refit the model and repeat.
7. Evaluate the final model using the held-out test set.

This combines two different selection criteria:

- **RFE:** ranks predictors using the fitted estimator's coefficients or feature-importance values.
- **Backward Elimination:** removes predictors using OLS p-values.

The final predictor set may therefore differ from the subset obtained using Approach 1.

### 6.6 Comparison of the Two Approaches

| Aspect | Approach 1 | Approach 2 |
|---|---|---|
| Project method | Backward Elimination + Forward Selection | RFE + Backward Elimination |
| Initial selection strategy | Forward Selection starts empty; Backward Elimination starts with candidate predictors | RFE reduces the feature set first |
| Primary selection criteria | p-values and Adjusted R² | Estimator-based feature ranking and p-values |
| Statistical p-values | Used in Backward Elimination | Used in the final Backward Elimination stage |
| Feature ranking by estimator coefficients | Not required for Forward Selection | Used by coefficient-based RFE |
| Main purpose | Compare stepwise statistical selection procedures | Combine estimator-based ranking with statistical elimination |

Neither approach is guaranteed to perform better in every dataset. Their performance depends on the data, preprocessing, feature-selection settings, and validation procedure.

---

## 7. Project Workflow

### Workflow A: Backward Elimination + Forward Selection

```text
Start
  |
  v
Load Housing.csv
  |
  v
Inspect and explore the dataset
  |
  v
Separate predictors and target
  |
  v
Split into training and test sets
  |
  v
Encode categorical variables
  |
  v
Scale numerical predictors
  |
  v
Fit Multiple Linear Regression
  |
  +-----------------------------+
  |                             |
  v                             v
Backward Elimination       Forward Selection
  |                             |
  +--------------+--------------+
                 |
                 v
       Compare selected models
                 |
                 v
       Evaluate on test data
                 |
                 v
       Analyse residuals
                 |
                 v
       Predict house prices
```

### Workflow B: RFE + Backward Elimination

```text
Start
  |
  v
Load Housing.csv
  |
  v
Inspect and explore the dataset
  |
  v
Separate predictors and target
  |
  v
Split into training and test sets
  |
  v
Encode categorical variables
  |
  v
Scale numerical predictors
  |
  v
Apply RFE to training data
  |
  v
Retain RFE-selected predictors
  |
  v
Fit OLS regression model
  |
  v
Apply Backward Elimination
  |
  v
Evaluate final model on test data
  |
  v
Analyse residuals
  |
  v
Predict house prices
```

Both workflows should use the same train/test split and compatible preprocessing when comparing their results.

---

## 8. How to Run the Project in Google Colab

### Step 1: Open Google Colab

Visit [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Upload the dataset

Upload `Housing.csv` to your Colab session.

If the Python code includes an upload prompt, select the dataset when requested.

### Step 3: Copy the Python code

Paste the complete Python implementation into the notebook.

For the first approach, run the code containing Backward Elimination and Forward Selection.

For the second approach, include the RFE stage followed by Backward Elimination and the separate evaluation section.

### Step 4: Execute the notebook

Run the cells in order, or select **Runtime → Run all**.

### Step 5: Review the output

Review the dataset summary, selected features, OLS summaries, evaluation metrics, diagnostic graphs, and predicted prices.

### Step 6: Save the results

Download any required CSV outputs or graphs from the Colab file panel.

---

## 9. Exploratory Data Analysis

Exploratory Data Analysis (EDA) helps understand the structure and distribution of the dataset before modelling.

The project may include:

- Dataset preview.
- Dataset shape and data types.
- Descriptive statistics.
- Missing-value analysis.
- Duplicate-record analysis.
- Distribution of house prices.
- Distribution of numerical predictors.
- Correlation analysis.
- Scatter plots showing relationships between predictors and price.
- Box plots to inspect possible outliers.

EDA helps identify data-quality issues and understand the relationships that may influence the regression model.

---

## 10. Model Evaluation

Both approaches should be evaluated on the same held-out test set.

### A. R² Score

R² measures how much of the variation in the target is explained by the model on the evaluated dataset.

Higher test R² generally indicates a better fit, but it does not prove that the model is appropriate or that its predictions are unbiased.

### B. Adjusted R²

Adjusted R² accounts for the number of predictors in the model. It is useful when comparing models with different numbers of predictors fitted to the same observations.

### C. Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE measures the average absolute difference between actual and predicted house prices.

A lower MAE indicates smaller average absolute prediction errors.

### D. Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE gives greater weight to large prediction errors because the errors are squared before averaging.

A lower RMSE generally indicates better prediction accuracy on the evaluated dataset.

### E. Model Comparison Table

Fill this table with the actual values printed by your Python notebook.

| Evaluation Measure | Approach 1: Backward Elimination + Forward Selection | Approach 2: RFE + Backward Elimination |
|---|---|---|
| Number of selected predictors | Actual output | Actual output |
| Test R² | Actual output | Actual output |
| Test Adjusted R² | Actual output, if calculated | Actual output, if calculated |
| Test MAE | Actual output | Actual output |
| Test RMSE | Actual output | Actual output |

Do not enter assumed or invented results. The actual metrics depend on the data, train/test split, and model settings.

---

## 11. Residual Analysis and Diagnostics

A residual is the difference between an observed price and the model's predicted price.

\[
e_i=y_i-\hat{y}_i
\]

Residual analysis helps determine whether the model's errors show systematic patterns.

The project may use the following diagnostics:

- **Residuals versus fitted values:** checks for visible patterns and changes in residual spread.
- **Residual histogram:** examines the shape of the residual distribution.
- **Normal Q–Q plot:** compares residual quantiles with normal-distribution quantiles.
- **Residuals versus predictors:** helps identify possible non-linear patterns.
- **Durbin–Watson statistic:** assesses residual autocorrelation in an appropriate observation order.
- **Variance Inflation Factor (VIF):** investigates multicollinearity among predictors.

These diagnostics help assess the suitability of the regression model. They do not guarantee that every modelling assumption is satisfied.

---

## 12. Predicting the Price of a New House

After selecting and fitting a final model, the project can predict the selling price of a new property.

The prediction function should accept the required property attributes:

- Area
- Number of bedrooms
- Number of bathrooms
- Number of stories
- Main-road access
- Guest-room availability
- Basement availability
- Hot-water heating
- Air-conditioning availability
- Number of parking spaces
- Preferred-area status
- Furnishing status

The function must apply the same categorical encoding, scaling, and selected feature columns used during training.

The resulting value is an estimated selling price based on the relationships learned from the dataset. It is not a guaranteed valuation or a live real-estate market quote.

---

## 13. Expected Project Outputs

Depending on the Python implementation, the notebook may produce:

1. Dataset preview and descriptive statistics.
2. Exploratory visualisations.
3. Initial Multiple Linear Regression summary.
4. Features retained by Backward Elimination.
5. Features retained by Forward Selection.
6. Features selected by RFE.
7. Final features retained by RFE followed by Backward Elimination.
8. Regression coefficients and p-values.
9. VIF table.
10. Test-set R², Adjusted R², MAE, and RMSE.
11. Actual-versus-predicted plots.
12. Residual diagnostic plots.
13. Example house-price predictions.
14. Exported CSV files, if enabled in the code.

The exact feature lists and numerical results must be taken from the actual notebook execution.

---

## 14. Important Implementation Considerations

- **Both approaches require separate implementations.** A README that documents RFE does not automatically add RFE to the Python notebook.
- **Use a consistent train/test split.** This makes comparison between the two approaches more meaningful.
- **Avoid data leakage.** Fit preprocessing tools and feature selectors using training data only.
- **Keep categorical encoding consistent.** New-house inputs must use the same encoding rules as the training dataset.
- **Scale predictors appropriately.** Coefficient-based RFE can be sensitive to differences in feature scales.
- **Understand the selection criteria.** Backward Elimination uses p-values, Forward Selection uses the chosen improvement criterion, and RFE uses estimator-based ranking.
- **Avoid tuning against the final test set.** Use cross-validation or training-only validation for model-selection decisions.
- **Interpret p-values cautiously.** They depend on statistical assumptions and should not be treated as direct measures of practical importance.
- **Check multicollinearity.** Highly correlated predictors may make coefficient estimates unstable.
- **Do not assume causation.** Regression coefficients describe estimated relationships in the model, not necessarily causal effects.
- **Report actual metrics.** Use the outputs generated by your own run rather than expected or invented scores.

---

## 15. Limitations

The project has several limitations:

- The model assumes that the relationship between the predictors and house price is adequately represented by a linear form.
- The dataset may not include all factors affecting property prices, such as exact location, building age, local market demand, or changes over time.
- Outliers and data-quality problems can influence the fitted regression model.
- Multicollinearity can affect the stability and interpretation of coefficients.
- Stepwise selection methods can be sensitive to the sample and the candidate predictors.
- RFE results depend on the estimator, feature scaling, and selected feature count.
- A model trained on one housing market may not generalise well to another market.
- Strong performance on one test split does not guarantee performance on future data.

---

## 16. Future Improvements

Possible extensions include:

- Use cross-validation to compare feature-selection methods more robustly.
- Tune the number of RFE-selected features using training-only validation.
- Build preprocessing and modelling pipelines.
- Compare Multiple Linear Regression with Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, and Gradient Boosting models.
- Investigate non-linear transformations where residual diagnostics suggest they are needed.
- Analyse influential observations and potential outliers.
- Add reliable property and market information if available.
- Save the fitted preprocessing transformations and final model for repeatable predictions.
- Evaluate the model on a separate dataset from the intended deployment market.

---

## 17. Conclusion

This project demonstrates how Multiple Linear Regression can be used to estimate residential property prices from structural and categorical characteristics.

It documents and compares two feature-selection approaches:

1. **Backward Elimination + Forward Selection**, which combines p-value-based elimination with incremental feature selection using Adjusted R².
2. **Recursive Feature Elimination (RFE) + Backward Elimination**, which first ranks and removes predictors using an estimator and then applies statistical p-value-based elimination.

Comparing the approaches helps illustrate how different feature-selection criteria produce different predictor sets and potentially different prediction performance.

The final model should be chosen using appropriate validation, held-out test metrics, residual diagnostics, and an understanding of the dataset's limitations.

---

## 18. Project Summary

| Item | Description |
|---|---|
| Project title | Housing Price Prediction Using Multiple Linear Regression |
| Problem type | Supervised regression |
| Dataset | `Housing.csv` |
| Target variable | `price` |
| Regression model | Multiple Linear Regression / Ordinary Least Squares |
| Approach 1 | Backward Elimination + Forward Selection |
| Approach 2 | Recursive Feature Elimination (RFE) + Backward Elimination |
| Feature-selection criteria | p-values, Adjusted R², and estimator-based feature ranking |
| Evaluation metrics | R², Adjusted R², MAE, RMSE |
| Programming language | Python |
| Development environment | Google Colab |
