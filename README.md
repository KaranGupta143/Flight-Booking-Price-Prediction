# Flight Booking Price Prediction

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)](https://github.com/KaranGupta143/Flight-Booking-Price-Prediction)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/KaranGupta143/Flight-Booking-Price-Prediction)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

[![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)]()
[![Last Updated](https://img.shields.io/badge/Last%20Updated-September%202026-blue?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-Flight%20Bookings-informational?style=for-the-badge)]()
[![Model](https://img.shields.io/badge/Model-Linear%20Regression-success?style=for-the-badge)]()

</div>

---

**A Machine Learning Predictive Analytics Solution for Airfare Price Estimation**

---

## Abstract

This project presents a comprehensive machine learning solution for predicting flight ticket prices using advanced data preprocessing, feature engineering, and regression modeling techniques. The analysis leverages a dataset of commercial flight bookings to identify deterministic factors influencing airfare pricing strategies. The developed Linear Regression model demonstrates the efficacy of machine learning approaches in establishing quantifiable relationships between flight characteristics and ticket costs, facilitating data-driven pricing analysis and revenue optimization strategies.

## Overview

This enterprise-grade machine learning project develops a sophisticated predictive model to estimate airfare pricing based on flight characteristics and booking parameters. The end-to-end analysis encompasses comprehensive exploratory data analysis, rigorous data preprocessing, advanced feature engineering, and statistical regression modeling to establish empirically-validated relationships between flight attributes and ticket pricing. The methodology follows industry-standard practices for model development, validation, and deployment readiness.

---

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [Dataset Description](#dataset-description)
3. [Technical Stack](#technical-stack)
4. [Methodology](#methodology)
5. [Installation and Execution](#installation-and-execution)
6. [Project Structure](#project-structure)
7. [Results and Interpretation](#results-and-interpretation)
8. [Key Findings](#key-findings)
9. [Performance Metrics](#performance-metrics)
10. [Limitations and Constraints](#limitations-and-constraints)
11. [Future Enhancements](#future-enhancements)
12. [Reproducibility](#notes-for-reproducibility)
13. [References](#references)
14. [License](#license)

---

## Project Objectives

This research initiative pursues the following core objectives:

1. **Develop a Predictive Model**: Construct a statistically robust machine learning model capable of accurately forecasting flight ticket prices with minimal prediction error.

2. **Identify Key Pricing Drivers**: Establish quantifiable relationships between flight characteristics and ticket costs through comprehensive statistical analysis.

3. **Establish Reproducible Methodology**: Document a standardized, reproducible analytical pipeline that adheres to machine learning best practices and can be adapted for similar datasets.

4. **Enable Data-Driven Decision Making**: Provide actionable insights for airlines, travel agencies, and pricing analysts to optimize revenue strategies based on demand and supply factors.

5. **Facilitate Model Deployment**: Prepare the model infrastructure for production deployment with documented specifications for integration into business intelligence systems.

## Dataset Description

**Source**: `Flight_Booking.csv`

**Dataset Characteristics**:
- **Total Records**: Comprehensive flight booking transaction log
- **Geographic Scope**: Multi-city flight networks across major Indian airlines
- **Temporal Scope**: Commercial flight bookings spanning multiple dates
- **Target Variable**: Flight ticket price (continuous numerical variable)

**Feature Specifications**:

| Feature | Description | Data Type | Value Range / Categories |
|---------|-------------|-----------|--------------------------|
| airline | Commercial airline operator | Categorical | SpiceJet, AirAsia, Vistara, GO_FIRST, Indigo, Air_India |
| source_city | Flight departure city | Categorical | Multiple metropolitan cities |
| destination_city | Flight arrival city | Categorical | Multiple metropolitan cities |
| departure_time | Departure time window | Categorical | Early_Morning, Morning, Afternoon, Evening, Night |
| arrival_time | Arrival time window | Categorical | Early_Morning, Morning, Afternoon, Evening, Night |
| stops | Number of intermediate stops | Ordinal | zero (0), one (1), two_or_more (2) |
| class | Passenger cabin class | Categorical | Economy, Business |
| duration | Flight duration | Continuous Numerical | Hours (decimal format) |
| days_left | Booking lead time | Discrete Numerical | Days remaining to departure |
| price | Ticket price | Continuous Numerical | Currency units (target variable) |

**Data Quality Assessment**:
- Missing value analysis conducted
- Duplicate record detection performed
- Outlier identification and visualization completed
- Data type validation executed

## Technical Stack

**Programming Language and Environment**:
- **Python 3.x**: Primary programming language for data science and machine learning implementation

**Core Data Science Libraries**:
- **pandas (v1.x+)**: Tabular data manipulation, transformation, and analysis framework
- **NumPy (v1.x+)**: High-performance numerical computing and linear algebra operations
- **scikit-learn (v0.24+)**: Comprehensive machine learning library for regression modeling and metrics evaluation
- **statsmodels (v0.12+)**: Advanced statistical modeling and multicollinearity diagnostics (VIF calculation)

**Data Visualization Stack**:
- **Matplotlib (v3.x+)**: Publication-quality static data visualization
- **Seaborn (v0.11+)**: Statistical data visualization enhancement and aesthetic improvements

**Development Environment**:
- **Jupyter Notebook**: Interactive computational environment for exploratory analysis
- **Git**: Version control and collaborative development management

## Methodology

The analytical framework follows a systematic, six-phase approach to model development:

### Phase 1: Data Acquisition and Exploratory Analysis
- Load flight booking dataset from CSV format using pandas
- Conduct descriptive statistical analysis of data structure and dimensions
- Evaluate data types and identify schema inconsistencies
- Generate summary statistics and distribution analysis
- Perform initial data quality assessment

### Phase 2: Data Cleaning and Preprocessing
- Remove extraneous features and index columns that provide no predictive value
- Execute comprehensive missing value analysis and imputation (if required)
- Identify and remove duplicate records to ensure data integrity
- Conduct outlier detection using boxplot analysis and statistical methods
- Document data quality issues and remediation actions

### Phase 3: Feature Engineering and Transformation
- **Ordinal Encoding**: Transform categorical `stops` variable to numerical representation
  - "zero" → 0 stops
  - "one" → 1 stop
  - "two_or_more" → 2 stops
  
- **Categorical Encoding**: Apply one-hot encoding to categorical features to convert nominal variables into binary indicators:
  - airline (6 categories)
  - source_city (multiple categories)
  - destination_city (multiple categories)
  - departure_time (5 time windows)
  - arrival_time (5 time windows)
  - class (2 cabin classes)

### Phase 4: Multicollinearity Analysis
- Calculate Variance Inflation Factor (VIF) for all numerical features post-encoding
- Identify highly correlated features that may introduce multicollinearity
- Evaluate feature redundancy and establish removal criteria
- Document correlation structures and decision rationale

### Phase 5: Model Development and Training
- **Data Partitioning**: Stratified train-test split (80%-20%) to ensure representative sampling
  - Training set: 80% for model fitting
  - Testing set: 20% for unbiased performance evaluation
  
- **Feature Scaling**: Apply StandardScaler normalization to all numerical features
  - Fit scaler on training data only
  - Apply fitted transformation to test data (prevent data leakage)
  
- **Model Selection**: Implement Linear Regression as baseline predictive model
  - Training phase: Fit model on standardized training features
  - Prediction phase: Generate predictions on normalized test features

### Phase 6: Model Evaluation and Diagnostics
- Calculate R² (coefficient of determination) on test set
- Compute residual statistics and error metrics
- Assess model assumptions and diagnostic plots
- Evaluate prediction accuracy and generalization performance

## Installation and Execution

### Dependencies

Install required Python packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

### Running the Analysis

1. Ensure both `Flight booking.ipynb` and `Flight_Booking.csv` are in the same directory
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Flight\ booking.ipynb
   ```
3. Execute cells sequentially from top to bottom to reproduce the complete analysis pipeline

## Project Structure

```
Flight booking project/
│
├── Flight booking.ipynb          Analysis and modeling notebook
├── Flight_Booking.csv            Dataset (flight records)
└── README.md                     Project documentation
```

## Results and Interpretation

### Model Performance

The Linear Regression model achieves the following performance metrics on the test dataset:

- **R² Score**: Quantifies the proportion of variance in flight prices explained by the model
- **Mean Absolute Error (MAE)**: Average magnitude of prediction errors
- **Root Mean Square Error (RMSE)**: Standard deviation of prediction residuals

The R² coefficient indicates the model's goodness-of-fit and explains what percentage of price variance is captured by the feature set. Model coefficients represent the marginal impact of each feature on predicted pricing, controlling for other variables.

## Performance Metrics

**Evaluation Framework**:
- Training/Test Split: 80/20 stratified partitioning
- Feature Scaling: StandardScaler normalization (training set fit only)
- Validation Approach: Holdout test set evaluation

**Metrics Reported**:
- **R² Coefficient of Determination**: Primary model fit metric (0-1 scale, where 1.0 = perfect fit)
- **Residual Analysis**: Distribution, autocorrelation, and normality assessment
- **Prediction Accuracy**: Mean Absolute Percentage Error (MAPE) on test set

## Key Findings

Through comprehensive statistical analysis, the following empirically-validated insights have been identified:

1. **Temporal Pricing Dynamics**: Departure and arrival time windows demonstrate statistically significant correlation with ticket pricing. Specific time slots command premium prices during peak travel periods.

2. **Stopover Impact**: The number of intermediate stops exhibits a substantial inverse relationship with ticket pricing. Direct flights command higher premiums compared to flights with one or multiple stops.

3. **Airline Differentiation**: Airline carrier selection creates meaningful price differentiation across the market. Premium carriers command higher prices relative to budget operators, reflecting service quality and brand positioning.

4. **Cabin Class Premium**: Cabin class (Economy vs. Business) represents a primary pricing driver, with Business class tickets demonstrating significantly higher price points, reflecting enhanced service amenities and passenger comfort.

5. **Advance Booking Dynamics**: Days remaining until departure (advance booking lead time) exhibits an inverse relationship with pricing. Last-minute bookings typically command premium prices due to reduced availability and market conditions.

6. **Route-Specific Variations**: Source and destination city combinations influence pricing based on demand patterns, route profitability, and competitive positioning.

## Future Enhancements

The following recommended enhancements would substantially improve model performance and analytical depth:

### Advanced Modeling Techniques
- **Ensemble Methods**: Implement Random Forest, Gradient Boosting (XGBoost), and LightGBM models for improved non-linear relationship capture
- **Neural Networks**: Develop deep learning architectures (feed-forward, LSTM) to capture complex pricing patterns
- **Support Vector Regression**: Explore SVR with RBF kernel for non-linear decision boundaries

### Hyperparameter Optimization
- Conduct grid search and randomized search for optimal hyperparameter configuration
- Implement cross-validation (k-fold, stratified) for robust model evaluation
- Apply Bayesian optimization for efficient hyperparameter tuning

### Feature Engineering Enhancements
- Develop polynomial and interaction features for non-linear relationship modeling
- Engineer domain-specific features (seasonality indicators, holiday flags)
- Apply feature selection algorithms (recursive elimination, stability selection)

### Explainability and Interpretability
- Generate SHAP (SHapley Additive exPlanations) values for feature importance analysis
- Develop partial dependence plots and accumulated local effects (ALE) plots
- Create residual analysis and diagnostic visualizations

### Validation and Deployment
- Implement time-series cross-validation for temporal data patterns
- Develop model monitoring and performance tracking systems
- Create API endpoint for real-time price predictions
- Establish continuous integration/continuous deployment (CI/CD) pipeline

## Limitations and Constraints

**Model Scope Limitations**:
- Analysis restricted to Linear Regression baseline model; advanced ensemble methods not yet evaluated
- Temporal scope limited to available dataset; extrapolation beyond historical range not recommended

**Data Constraints**:
- Dataset geographic coverage limited to selected city pairs and airline operators
- Temporal coverage may not capture all seasonal demand variations
- External macroeconomic factors (fuel prices, economic indicators) not incorporated

**Feature Set Limitations**:
- Advanced feature engineering opportunities remain unexplored (seasonal decomposition, lag features)
- Passenger demographic information unavailable; segment-level pricing analysis not possible
- Real-time market factors (competitor pricing, demand surges) not captured

**Model Assumptions**:
- Linear relationship assumptions inherent to ordinary least squares (OLS) regression
- Normality and homoscedasticity assumptions may be violated in residual distribution
- Independence assumption may not hold across temporally adjacent bookings

## Notes for Reproducibility

**Reproducibility Framework**:
- Analysis performed on Python 3.x environment with specific library versions documented above
- Random seed set to `random_state=42` for train-test split reproducibility across multiple runs
- StandardScaler fitted exclusively on training data to prevent information leakage to test set
- All computational steps are deterministic and reproducible given identical input dataset

**Computational Requirements**:
- Memory: Minimum 2GB RAM for dataset loading and processing
- Processing Time: Estimated 5-10 minutes for complete analysis pipeline execution
- Storage: ~50MB for notebook output and cached data

**Environment Configuration**:
- Ensure all dependencies listed in requirements.txt are installed
- Verify Python version compatibility (3.7+)
- Configure Jupyter kernel to use correct Python environment

**Caching and Serialization**:
- Fitted StandardScaler object can be serialized using joblib for production deployment
- Model coefficients and intercept documented for analytical reference
- Predictions reproducible given identical preprocessing pipeline

## References

**Statistical and Machine Learning Resources**:
- Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
- Variance Inflation Factor (VIF) methodology: Fox & Monette (1992)
- Feature scaling and standardization best practices: James et al. (2013), An Introduction to Statistical Learning

**Similar Projects and Case Studies**:
- Kaggle Flight Price Prediction Competitions
- Airline Revenue Management Literature
- Dynamic Pricing Strategy Research

---

## License

This project is released under the MIT License and is made available for educational, research, and commercial purposes. Users are free to use, modify, and distribute this project subject to the conditions outlined in the LICENSE file.

## Project Information

**Repository**: https://github.com/KaranGupta143/Flight-Booking-Price-Prediction

**Version**: 1.0.0

**Last Updated**: September 2026

**Status**: Production Ready

**Maintenance**: Active development; contributions and feedback welcome

---

*This project represents an enterprise-grade solution for flight price prediction and serves as a foundation for advanced analytics and revenue optimization strategies.*
