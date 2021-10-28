**Work-in-Progress >> Adding Data Analysis and modelling with R (Work started)**

HDB Resale Flat Valuation (Singapore)

**Program and Goal**
This program aimed to predict the potential selling price of a HDB resale flat based on previous transaction data. Data were first preprocessed and features were engineered to improve the performance of the model. The prediction on a new "unseen" data on PropertyGuru.com produced a prediction within 14.5% of the listing price.


**The Data**
The data used in this program is a subset of "Resale Flat Prices" by Housing Development Board of Singapore obtained from Data.gov.sg: https://data.gov.sg/dataset/resale-flat-prices.


**Data Overview**
The dataset from HDB comes as a CSV file with the following information columns:
	"month" 	 	 	- The transaction year and month.
	"town" 		 	 	- The town the unit is located in.
	"flat_type" 	 	- The type of unit. e.g. 3 ROOM / EXECUTIVE
	"block" 	 	 	- The block number part of the unit's address.
	"street_name" 	 	- The other parts of the unit's address.
	"storey_range" 	 	- The level of the unit, indicated as a range.
	"floor_area_sqm" 	- The floor area indicated in square metres.
	"flat_model"	 	- The model of the particular flat_type. e.g. Type A / Mansionette.
	"lease_commence_date- The date which the lease commenced.
	"remaining_lease"	- The lease duration left based on a 99-year lease, indicated in years and months.
	"resale_price" 		- The price of the unit transacted.


**Feature Engineering**
We first checked if there were any null values that needed to be addressed. We then seeked to convert all qualitative values to numeric either by one-hot or categorical encoding. Dates were converted to date-types and details (months and years) were extracted as numbers. Collinearity of variables were checked and dropped and variables with excessive unique values were also dropped.   


**How to Run**
Download the "data" folder.
Download the following 3 Python files:
1. HDB_Valuation_Model.py
2. HDB_Valuation.py
3. Utils.py
 
To train HDB_Valuation model and see the results, follow the steps below.
1. In the terminal, cd into the directory where the downloaded files are. Make sure files are organised as shown.
2. Enter command "python HDB_Valuation.py [-h] [--response_column_name RESPONSE_COLUMN_NAME]
                        [--test_data_proportion TEST_DATA_PROPORTION] [--node_multiples NODE_MULTIPLES]
                        [--patience PATIENCE] [--training_epochs TRAINING_EPOCHS]" 
3. Example: python HDB_Valuation.py --response_column_name resale_price --test_data_proportion 0.3 --node_multiples 1 --patience 10 --training_epochs 100
 

**Result**
Model produced a RMSE of around $60000, within 13.04% of the mean resale price. With a F-statistics of almost 10000, it showed that the variables were useful in predicting the resale price and we could reject the null hypothesis. With an R^2 value of 0.87, we could say that much of the resale price was explained by the variables of used. We could also say that a good proportion of the variability was removed by the regression.

The trained model was used to predict a newly listed unit on PropertyGuru.com. This dataset was totally new and have not been seen by the model. It valued the unit at around SGD$586000 while it is listed at SGD$685000. The percentage error was 14.5%.  
