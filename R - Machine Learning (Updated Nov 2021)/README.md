HDB Resale Flat Valuation (Singapore)


**Program and Goal**
This program aimed to predict the potential selling price of HDB resale flats based on previous transaction data. 


**The Data**
The data used in this program is a subset of "Resale Flat Prices" by Housing Development Board of Singapore obtained from Data.gov.sg: https://data.gov.sg/dataset/resale-flat-prices.


**Data Overview**
The dataset from HDB comes as a CSV file with the following information columns:
	"month" 	        - The transaction year and month.
	"town" 		        - The town the unit is located in.
	"flat_type" 	    - The type of unit. e.g. 3 ROOM / EXECUTIVE
	"block"		        - The block number part of the unit's address.
	"street_name"   	- The other parts of the unit's address.
	"storey_range"	    - The level of the unit, indicated as a range.
	"floor_area_sqm"    - The floor area indicated in square metres.
	"flat_model"	    - The model of the particular flat_type. e.g. Type A / Mansionette.
	"lease_commence_date- The date which the lease commenced.
	"remaining_lease"	- The lease duration left based on a 99-year lease, indicated in years and months.
	"resale_price"	    - The price of the unit transacted.


**Structure & Approach**
Part 1 - Manual Data Pre-processing
In this part, duplicates and missing data was handled. Exploratory data analysis was carried out to understand the data through distribution and features analysis. Features were then engineered to address the findings. A processed set of data was then passed on to Part 2 for model fitting.

Part 2 - Model Fitting
The data was further processed into training and test sets and scaled accordingly. Automatic feature selection were used to check the importance of the remaining features passed on by Part 1. Multiple models were used to fit the data to obtain the best algorithm for the regression task. Hyperparameters of the best algorithm were tuned using grid search. Best model with the optimum parameters were selected. Final model was fitted with the entire dataset.
 
 
**Result and Conclusion**
All features are deemed important and retained after feature selection using rfe(). Random forest algorithm performed better than knn after 10-fold cross validation. Grid search found minimum node size of 5, with 500 trees and minimum split at 6 optimum for the model prediction. Model predicted within 7.5% of mean resale price with RMSE of 33933.4. Final model to be fitted with full data and used for prediction and valuation.



