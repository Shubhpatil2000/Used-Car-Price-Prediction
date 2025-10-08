The Used-Car-Price-Prediction project focuses on predicting the selling price of used cars using a dataset sourced from cardehko.com.
The process began with data cleaning, where unnecessary columns like car_name and brand were removed.
  Feature preprocessing was a key step, involving the use of a LabelEncoder for the high-cardinality model column, 
OneHotEncoder for other categorical features like seller_type and fuel_type, and StandardScaler for all numerical data.
After splitting the data into training and testing sets, multiple regression models were trained and evaluated,
including Linear Regression, K-Neighbors Regressor, and Decision Tree.
The Random Forest Regressor was identified as the best-performing model, achieving an R² score of 0.9274 on the test data.
