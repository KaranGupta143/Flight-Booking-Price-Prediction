# Flight Booking Price Prediction

## Overview

This project develops a machine learning-based predictive model to estimate airfare pricing based on flight characteristics and booking parameters. The analysis encompasses exploratory data analysis, data preprocessing, feature engineering, and regression modeling to establish relationships between flight attributes and ticket pricing.

## Project Objectives

The primary objective is to develop a predictive model capable of accurately forecasting flight ticket prices based on the following factors:

- Airline carrier
- Source and destination cities
- Departure and arrival times
- Number of stops
- Flight duration
- Cabin class
- Days until departure

## Dataset Description

**Source File**: `Flight_Booking.csv`

The dataset contains flight booking records with the following features:

| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| airline | Airline operator | Categorical | SpiceJet, AirAsia, Vistara, GO_FIRST, Indigo, Air_India |
| source_city | Departure city | Categorical | Multiple cities |
| destination_city | Arrival city | Categorical | Multiple cities |
| departure_time | Departure time slot | Categorical | Early_Morning, Morning, Afternoon, Evening, Night |
| arrival_time | Arrival time slot | Categorical | Early_Morning, Morning, Afternoon, Evening, Night |
| stops | Number of stops | Ordinal | zero, one, two_or_more |
| class | Cabin class | Categorical | Economy, Business |
| duration | Flight duration | Numerical | Hours (continuous) |
| days_left | Days until departure | Numerical | Integer |
| price | Ticket price | Numerical | Target variable |

## Technical Stack

- **Python 3.x**: Core programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and linear algebra
- **scikit-learn**: Machine learning framework and metrics
- **Matplotlib & Seaborn**: Data visualization
- **statsmodels**: Statistical modeling and multicollinearity analysis

## Methodology

### 1. Data Acquisition and Exploration
- Load flight booking dataset from CSV format
- Examine data structure, dimensions, and statistical properties
- Assess data types and identify potential issues

### 2. Data Preprocessing
- Remove extraneous columns and identifiers
- Conduct missing value analysis
- Identify and handle duplicate records
- Detect and visualize outliers through boxplot analysis

### 3. Feature Engineering
- Convert ordinal `stops` variable to numerical representation (0, 1, 2)
- Apply one-hot encoding to categorical features:
  - airline
  - source_city
  - departure_time
  - arrival_time
  - destination_city
  - class

### 4. Multicollinearity Assessment
- Calculate Variance Inflation Factor (VIF) for numerical features
- Evaluate correlation structures to identify redundant predictors

### 5. Model Development
- Partition dataset into training (80%) and testing (20%) subsets
- Apply StandardScaler for feature normalization
- Train Linear Regression model on standardized training data
- Generate predictions on holdout test set

### 6. Model Evaluation
- Calculate R² coefficient of determination
- Assess prediction accuracy and model fit quality

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

The Linear Regression model generates R² scores on the test set, quantifying the proportion of variance in flight prices explained by the model. Model coefficients indicate the marginal impact of each feature on predicted pricing.

## Key Findings

- Temporal factors (departure and arrival times) demonstrate significant correlation with pricing
- Stopover frequency substantially influences ticket cost
- Airline selection creates meaningful price differentiation
- Cabin class represents a primary pricing driver
- Advance booking timing (days_left) affects pricing strategy and market rates

## Future Enhancements

To improve predictive performance and analytical depth, the following enhancements are recommended:

- Implement ensemble methods (Random Forest, Gradient Boosting, XGBoost)
- Conduct hyperparameter optimization using cross-validation
- Explore non-linear relationships through polynomial feature engineering
- Develop residual analysis and diagnostic plots
- Perform feature importance analysis
- Implement validation strategies for temporal data patterns

## Limitations

- Analysis restricted to Linear Regression baseline model
- Dataset temporal scope and geographic coverage constraints
- Feature engineering opportunities remain unexplored
- External factors (fuel prices, demand fluctuations) not considered

## Notes for Reproducibility

- Analysis performed on Python 3.x environment
- Random seed set to 42 for train-test split reproducibility
- Standardization applied only to training set; test set scaled using training statistics

## License

This project is made available for educational and research purposes.

---

**Last Updated**: September 2026

**Status**: Production Ready
