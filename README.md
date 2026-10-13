# House Price Prediction Using Multiple Linear Regression

## 1. Project Overview

The **House Price Prediction** project uses machine learning and statistical techniques to predict the selling price of residential properties based on their characteristics, such as area, number of bedrooms, bathrooms, stories, parking spaces, and other facilities.

The project uses **Multiple Linear Regression** as the prediction model and applies two feature-selection techniques:

- **Recursive Feature Elimination (RFE)**
- **Backward Elimination using Ordinary Least Squares (OLS)**

It also performs multicollinearity analysis using the **Variance Inflation Factor (VIF)**, residual analysis, and model evaluation using standard regression metrics.

The complete implementation is designed to run in **Google Colab**, making it easy to execute without installing Python or configuring a local development environment.

## 2. Objectives

The main objectives of this project are:

1. To analyze a housing dataset and understand the relationships between housing features and selling prices.
2. To perform exploratory data analysis (EDA) using statistical summaries and visualizations.
3. To preprocess numerical and categorical variables for regression analysis.
4. To build a Multiple Linear Regression model for house price prediction.
5. To identify important features using Recursive Feature Elimination.
6. To perform Backward Elimination using statistical significance and p-values.
7. To analyze multicollinearity using the Variance Inflation Factor.
8. To evaluate the model using appropriate regression metrics.
9. To visualize actual versus predicted prices and examine regression residuals.
10. To create a prediction function that estimates the price of a house from user-provided characteristics.

## 3. Dataset Description

The project uses the `Housing.csv` dataset, which contains information about residential properties and their selling prices.

**Dataset dimensions:** 545 rows and 13 columns.

The dataset contains one target variable and twelve predictor variables.

### Dataset attributes

| Attribute | Description | Type |
|---|---|---|
| `price` | Selling price of the house | Numerical target |
| `area` | Area of the house | Numerical |
| `bedrooms` | Number of bedrooms | Numerical |
| `bathrooms` | Number of bathrooms | Numerical |
| `stories` | Number of stories | Numerical |
| `mainroad` | Whether the house is connected to the main road | Categorical |
| `guestroom` | Whether the house has a guestroom | Categorical |
| `basement` | Whether the house has a basement | Categorical |
| `hotwaterheating` | Whether hot-water heating is available | Categorical |
| `airconditioning` | Whether air conditioning is available | Categorical |
| `parking` | Number of parking spaces | Numerical |
| `prefarea` | Whether the house is in a preferred area | Categorical |
| `furnishingstatus` | Furnishing condition of the house | Categorical |

### Target variable

The target variable is `price`.

The model learns the relationship between the housing characteristics and the selling price to estimate prices for previously unseen houses.

## 4. Technologies and Libraries Used

The project uses Python and the following libraries:

| Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading, manipulation, and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing, train-test split, RFE, and evaluation metrics |
| Statsmodels | OLS regression, p-values, and statistical summaries |

Google Colab provides a convenient environment for running the project.

## 5. Project Workflow

The project follows the workflow below:

1. Import the required libraries.
2. Load the housing dataset.
3. Inspect the dataset and check its structure.
4. Perform exploratory data analysis.
5. Visualize the distributions and relationships among variables.
6. Encode categorical variables into numerical representations.
7. Divide the data into training and testing sets.
8. Scale selected numerical predictors.
9. Apply Recursive Feature Elimination.
10. Perform Backward Elimination using OLS p-values.
11. Check multicollinearity using VIF.
12. Fit the final regression model.
13. Analyze residuals and regression assumptions.
14. Evaluate the model using test-set predictions.
15. Display the regression equation and feature coefficients.
16. Predict the price of a house using new input values.

## 6. Methodology

### 6.1 Exploratory Data Analysis

Exploratory Data Analysis helps understand the dataset before model development.

The project uses:

- Dataset shape and data types.
- Statistical summaries.
- Histograms to examine numerical distributions.
- Pair plots to explore relationships between numerical features.
- Correlation heatmaps to inspect linear relationships.
- Box plots to identify possible outliers and unusual distributions.

EDA helps identify patterns and understand how housing characteristics may influence prices.

### 6.2 Data Preprocessing

Data preprocessing converts the original dataset into a format suitable for regression analysis.

**Categorical encoding**

The binary categorical variables are encoded numerically. For example:

- `yes` → 1
- `no` → 0

The `furnishingstatus` variable is converted into dummy variables using one-hot encoding with a reference category omitted. This avoids creating redundant dummy variables for all categories.

**Training and testing split**

The dataset is divided into:

- Training set: 70%
- Testing set: 30%

The split uses `random_state=100` for reproducibility.

The training data is used to fit the model, while the testing data is used to evaluate its performance on unseen observations.

**Feature scaling**

Min-Max scaling is applied to selected numerical predictors:

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking

The scaler is fitted using the training data and then applied to the testing data. This prevents information from the test set from influencing the scaling parameters.

The target variable, `price`, remains unscaled.

### 6.3 Multiple Linear Regression

Multiple Linear Regression estimates a continuous target variable using multiple independent variables.

The general equation is:

\[
Y=\beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_nX_n+\epsilon
\]

Where:

- \(Y\) = predicted house price
- \(\beta_0\) = intercept
- \(\beta_i\) = regression coefficient for predictor \(X_i\)
- \(X_i\) = housing feature
- \(n\) = number of predictors
- \(\epsilon\) = error term

The model estimates the relationship between the selected housing features and the selling price.

### 6.4 Recursive Feature Elimination (RFE)

Recursive Feature Elimination is a feature-selection technique available in Scikit-learn.

It repeatedly trains an estimator, ranks the features according to their importance, and removes less important features until the desired number remains.

In this project, RFE is used to select a subset of potentially useful predictors before statistical feature selection.

**Purpose:**

- Reduce the number of predictors.
- Identify potentially important variables.
- Simplify the regression model.
- Reduce the influence of less useful features.

RFE is a feature-selection method; it does not independently guarantee that every selected feature is statistically significant.

### 6.5 Backward Elimination Using OLS

Backward Elimination is a statistical feature-selection technique that starts with a set of candidate predictors and removes variables that do not satisfy a chosen significance criterion.

The project uses Ordinary Least Squares (OLS) regression from Statsmodels.

The procedure is:

1. Fit an OLS regression model using the current predictors.
2. Examine the p-value of each predictor.
3. Identify the predictor with the highest p-value.
4. Remove that predictor if its p-value exceeds 0.05.
5. Refit the model using the remaining predictors.
6. Repeat until the remaining predictors satisfy the selected p-value criterion or the procedure reaches its stopping condition.

The significance level is:

\[
\alpha=0.05
\]

A p-value below 0.05 is treated as evidence of statistical significance under the model's assumptions.

**Important:** Statistical significance does not necessarily mean that a feature has a large practical effect on house prices. Likewise, removing a feature based on its p-value can affect the interpretation and predictive performance of the model.

### 6.6 Multicollinearity Analysis Using VIF

Multicollinearity occurs when two or more independent variables are strongly related to one another.

It can make regression coefficients unstable and complicate the interpretation of individual feature effects.

The project uses the Variance Inflation Factor (VIF) to identify possible multicollinearity.

The formula is:

\[
VIF_i=\frac{1}{1-R_i^2}
\]

Where \(R_i^2\) is obtained by regressing predictor \(i\) against the other predictors.

A VIF value near 1 indicates little linear dependence on the other predictors. Larger values indicate increasing multicollinearity.

The implementation uses a threshold of 5 as a screening rule. Predictors exceeding this threshold may be removed iteratively, followed by refitting the regression model.

VIF thresholds are guidelines rather than universal rules. Removing variables solely to reduce VIF should be considered alongside statistical reasoning and predictive performance.

### 6.7 Final Regression Model

After feature selection and multicollinearity checks, the remaining predictors are used to fit the final OLS regression model.

The final model provides:

- Estimated regression coefficients.
- An intercept.
- P-values and other statistical information.
- Predicted house prices.

The exact predictors and coefficients depend on the dataset and the feature-selection results produced during execution.

## 7. Model Evaluation

The model is evaluated on the test dataset using the following regression metrics.

### 7.1 R-squared (R²)

R-squared measures the proportion of variation in the target variable explained by the model relative to a baseline that predicts the mean target value.

\[
R^2=1-\frac{\sum(y_i-\hat y_i)^2}
{\sum(y_i-\bar y)^2}
\]

A higher test-set R² generally indicates better explanatory performance relative to this baseline. However, it does not guarantee that the model will perform equally well on other datasets.

### 7.2 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted prices.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

A lower MAE indicates smaller average absolute prediction errors.

### 7.3 Mean Squared Error (MSE)

MSE measures the average squared prediction error.

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Because errors are squared, larger errors receive a greater penalty.

### 7.4 Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

\[
RMSE=\sqrt{MSE}
\]

RMSE is expressed in the same units as the target variable and is more sensitive to large prediction errors than MAE.

**Interpretation:** Lower MAE, MSE, and RMSE generally indicate more accurate predictions on the evaluated dataset. R² should be considered alongside these error metrics rather than in isolation.

## 8. Residual Diagnostics

Residuals are the differences between actual and predicted values.

\[
e_i=y_i-\hat y_i
\]

The project generates visualizations to inspect model errors.

### Residual histogram

Shows the distribution of residuals and helps assess whether the errors are approximately symmetric and centered near zero.

### Residuals versus fitted values

Helps identify systematic patterns, changing error variance, and possible model misspecification.

### Normal Q-Q plot

Compares residual quantiles with theoretical normal quantiles to examine whether the residual distribution is approximately normal.

These plots support diagnostic assessment but do not, by themselves, prove that all regression assumptions are satisfied.

## 9. Feature Coefficients and Explainability

The final model displays the estimated coefficients of the selected predictors.

A coefficient represents the estimated change in predicted price associated with a one-unit increase in that predictor, holding the other predictors constant.

Because some numerical predictors are scaled, their coefficients correspond to changes in the scaled values rather than directly to one original unit of area, bedrooms, bathrooms, stories, or parking.

For categorical dummy variables, a coefficient represents the estimated difference relative to the omitted reference category, holding the other predictors constant.

The project also visualizes the coefficients to make the model easier to interpret.

**Note:** Regression coefficients indicate associations learned from the dataset. They should not automatically be interpreted as causal effects.

## 10. House Price Prediction Calculator

The project includes a prediction function named `predict_house_price()`.

The function accepts housing characteristics, such as:

- Area
- Number of bedrooms
- Number of bathrooms
- Number of stories
- Main-road access
- Guestroom availability
- Basement availability
- Hot-water heating availability
- Air-conditioning availability
- Number of parking spaces
- Preferred-area status
- Furnishing status

The function applies the required encoding and scaling transformations before passing the input to the trained regression model.

It then returns the estimated house price.

The function uses the preprocessing objects and selected features produced during model training, so the prediction input must be processed consistently with the final model.

The predicted value is an estimate, not a guaranteed market selling price.

## 11. How to Run the Project in Google Colab

### Step 1: Open Google Colab

Visit:

https://colab.research.google.com/

Create a new notebook.

### Step 2: Upload the dataset

Make sure the dataset is named:

`Housing.csv`

If the code includes an automatic upload prompt, run the cell and upload the CSV file when requested.

Otherwise, upload the file through Colab's Files panel or modify the file path in the code.

### Step 3: Copy the Python code

Copy the complete project code into a single Colab code cell, as provided in the project implementation.

### Step 4: Execute the cell

Run the cell and allow the program to complete each stage.

The notebook should generate the data visualizations, feature-selection results, regression diagnostics, evaluation metrics, coefficient information, and example price prediction.

### Step 5: Review the results

Examine:

- The selected features.
- The final OLS regression summary.
- VIF values.
- Test-set R², MAE, MSE, and RMSE.
- Actual versus predicted prices.
- Residual diagnostic plots.
- The final regression equation.
- The example house price prediction.

The numerical results are generated at runtime and are not fixed in this README.

## 12. Running the Project Locally

To run the project on a local computer, install Python and the required libraries.

Install the dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

Save the Python implementation and `Housing.csv` in an accessible location.

Update the dataset path in the code if necessary, then run the script or execute the notebook in Jupyter.

Google Colab is the recommended option for users who do not want to configure a local Python environment.

## 13. Project Structure

A suggested repository structure is:

```text
house-price-prediction/
│
├── Housing.csv
├── README.md
├── house_price_prediction.ipynb
└── requirements.txt
```

- `Housing.csv` — housing dataset used for training and evaluation.
- `README.md` — project documentation.
- `house_price_prediction.ipynb` — notebook containing the implementation, visualizations, model training, evaluation, and prediction function.
- `requirements.txt` — list of Python library dependencies.

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

The filenames are suggested organization; use the actual filenames in your repository.

## 14. Advantages

- Uses multiple housing characteristics for price prediction.
- Includes exploratory data analysis and visualization.
- Applies feature selection to simplify the model.
- Uses statistical significance testing through OLS.
- Examines multicollinearity using VIF.
- Evaluates predictions using multiple regression metrics.
- Includes residual diagnostics.
- Provides an input-based house price prediction function.
- Can be executed in Google Colab with minimal setup.

## 15. Limitations

- The model depends on the quality and representativeness of the available dataset.
- Linear Regression may not capture complex nonlinear relationships between housing characteristics and prices.
- Feature selection can change with different train-test splits and datasets.
- P-values and VIF values depend on the predictors included in the model and the assumptions of the statistical analysis.
- The dataset may not represent current real-estate market conditions.
- The model does not automatically account for all potentially important factors, such as exact location, nearby amenities, market demand, interest rates, or changes in economic conditions.
- Performance on the test split does not guarantee performance on future properties.

## 16. Future Improvements

Possible extensions include:

1. Comparing Multiple Linear Regression with Ridge and Lasso Regression.
2. Comparing the model with Random Forest Regression and Gradient Boosting.
3. Using cross-validation for more reliable model evaluation.
4. Performing hyperparameter tuning for alternative models.
5. Adding more location-specific and market-related features.
6. Creating an interactive prediction interface using Streamlit.
7. Evaluating prediction errors across different price ranges.
8. Saving the trained model and preprocessing objects for later use.
9. Comparing RFE-only, Backward Elimination-only, and combined feature-selection approaches.

## 17. Conclusion

This project demonstrates a complete regression workflow for housing price prediction, from dataset exploration and preprocessing to feature selection, model fitting, statistical diagnostics, and performance evaluation.

It combines **Multiple Linear Regression, Recursive Feature Elimination, Backward Elimination, and VIF analysis** to investigate which housing characteristics are useful for predicting selling prices.

The final prediction function demonstrates how a trained regression model can be used to estimate the price of a house from its characteristics.

The project is intended for educational purposes and demonstrates the practical application of machine learning and statistical regression techniques. Its predictions should not be treated as professional property valuations.

## 18. Author

**Project:** House Price Prediction Using Multiple Linear Regression  
**Domain:** Machine Learning / Data Science  
**Language:** Python  
**Environment:** Google Colab / Jupyter Notebook  
**Dataset:** Housing.csv

---

*Disclaimer: This is an educational machine-learning project. Predictions are estimates based on the dataset and model assumptions, not guaranteed real-estate valuations.*

