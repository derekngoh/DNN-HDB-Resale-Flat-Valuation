HDB Resale Flat Valuation (Singapore)


**Program and Goal**
This program aimed to predict the potential selling price of a HDB resale flat based on previous transaction data. 


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

<SECTION_UPDATED> See updated data processing and feature engineering in the lastest R modelling on the same project. <SECTION_UPDATED>  

Section 1 : Explore Data 
Section 2 : Feature Engineering
Section 3 : Review Data
Section 4 : Model Training and Fitting  
Section 5 : Model Evaluation
Section 6 : Results and Analysis
 

**Result Analysis**
Model produced a RMSE of around $60000, within 13.04% of the mean resale price. With a F-statistics of almost 10000, it showed that the variables were useful in predicting the resale price and we could reject the null hypothesis. With an R^2 value of 0.87, we could say that much of the resale price was explained by the variables of used. We could also say that a good proportion of the variability was removed by the regression.


**Prediction**
The trained model was used to predict a newly listed unit on PropertyGuru.com. This dataset was totally new and have not been seen by the model. It valued the unit at around SGD$586000 while it is listed at SGD$685000. The percentage error was 14.5%.  

