import os,argparse,random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

from HDB_Valuation_Model import HDB_Model

cmdparser = argparse.ArgumentParser(description='hdb resale flat model')
cmdparser.add_argument('--response_column_name', help='the response column name', default='resale_price')
cmdparser.add_argument('--test_data_proportion', help='set the proportion of test data. e.g. 0.3 for 30%', default='0.3')
cmdparser.add_argument('--node_multiples', help='set the number of nodes. this number will be multiplied by the column count to give the final node count', default='1')
cmdparser.add_argument('--patience', help='set number of patience for earlystop', default='30')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='1')

args = cmdparser.parse_args()

response_column_name = args.response_column_name
test_data_proportion = float(args.test_data_proportion)
n_multiples = int(args.node_multiples)
patience = int(args.patience)
training_epochs = int(args.training_epochs)

filename = 'resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv'
current_file_loc = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_file_loc, "data", filename)
df=pd.read_csv(file_path)


# PREPROCESS data and engineer features. Process optimised prior using Google Colab.

#Column 'month'. Convert details to numeric.
df['month'] = pd.to_datetime(df['month'])
df['transaction_yr'] = df['month'].apply(lambda yr: yr.year)
df['transaction_mth'] = df['month'].apply(lambda mth: mth.month)
df = df.drop('month',axis=1)

#Column 'remaining_lease'. Convert details to numeric.
df = df.drop('lease_commence_date', axis=1) #Can be derived from remaining_lease
df['remaining_yr'] = df['remaining_lease'].apply(lambda yr: int(yr[:2]))
df['remaining_mth'] = df['remaining_lease'].apply(lambda mth: int(mth[-9:-6]))
df = df.drop('remaining_lease', axis=1)

#Column 'street_name'. #Too many unique values to be useful
df['street_name'].nunique() 
df = df.drop(['street_name', 'block'],axis=1)

#Column 'storey_range'. Convert details to numeric.
df['story_start'] = df['storey_range'].apply(lambda st: int(st[:2])) 
df['story_end'] = df['storey_range'].apply(lambda st: int(st[-2:])) 
df = df.drop('storey_range',axis=1)

#Column 'flat_type'. Convert details to numeric.
flat_type_map = {'1 ROOM':1, '2 ROOM':2, '3 ROOM':3, '4 ROOM':4, '5 ROOM':5, 'EXECUTIVE':6, 'MULTI-GENERATION':7}
df['flat_type'] = df['flat_type'].map(flat_type_map) 

#Column 'town'. One-hot encode.
df['town'].nunique() 
town_onehot = pd.get_dummies(df['town'],drop_first=True)
df = pd.concat([df,town_onehot],axis=1)
df = df.drop('town', axis=1)

#Column 'flat_model'. One-hot encode.
df['flat_model'].nunique()
flat_model_onehot = pd.get_dummies(df['flat_model'],drop_first=True)
df = pd.concat([df,flat_model_onehot],axis=1)
df = df.drop('flat_model',axis=1)

#Check collinearity. Collinearity affects standard error of mean and p-values which may affect our decision to reject the null hypothesis.
plt.figure(figsize=(25,16))
sns.heatmap(df.corr())
#plt.show()

#Drop flat_type (closely related to floor_area_sqm) and storey_end (closely related to storey start)
df = df.drop(['flat_type','story_end'],axis=1)

random_num = random.randint(1,999)

#TRAIN and fit model
hdb_model = HDB_Model()
node_multiples = n_multiples
y_column_name = response_column_name
epochs = training_epochs
early_stop_patience = patience
test_proportion = test_data_proportion

hdb_model.set_data(df,y_column_name)
hdb_model.set_xy_train_test(test_proportion)
hdb_model.scale_x_data()
hdb_model.create_model(nodes=node_multiples * hdb_model.xtrain.shape[1])
hdb_model.fit_model(early_stop_patience,epochs)
save_name = "hdb_price_predictor{a}.h5".format(a=random_num)
hdb_model.main_model.save(hdb_model.set_current_path(save_name))
hdb_model.save_loss_plot("hdb_loss_plot{a}.jpg".format(a=random_num))

#VIEW predictions on test data. SAVE results.
predictions = hdb_model.main_model.predict(hdb_model.scaled_xtest)
result_text1 = "\nRoot Mean Squared Error: {a:.2f}".format(a=np.sqrt(mean_squared_error(hdb_model.ytest,predictions)))
result_text2 = "\nMean Absolute Error: {a:.2f}".format(a=mean_absolute_error(hdb_model.ytest,predictions))
result_text3 = "\nExplained Variance Score: {a:.2f}".format(a=explained_variance_score(hdb_model.ytest, predictions))

#Percentage error to mean resale price.
percent_error_to_mean = 100*np.sqrt(mean_squared_error(hdb_model.ytest,predictions))/hdb_model.original_y.mean()
result_text4 = "\nPercent Error against mean: {a:.2f}%".format(a=percent_error_to_mean)

#See how well predictions matches true test values.
plt.scatter(hdb_model.ytest, predictions)
plt.savefig(hdb_model.set_current_path("hdb_scatter_plot{a}.jpg".format(a=random_num)))
plt.show()

#PREDICT with new data extracted from PropertyGuru https://www.propertyguru.com.sg/listing/hdb-for-sale-38b-bendemeer-road-23502555
listing_price = 685000
new_test_data = pd.DataFrame({'floor_area_sqm':[89], 'transaction_yr':[2021], 'transaction_mth':[9], 'remaining_yr':[90], 'remaining_mth':[0], 'story_start':[1], 'BEDOK':[0], 'BISHAN':[0], 'BUKIT BATOK':[0], 'BUKIT MERAH':[0], 'BUKIT PANJANG':[0], 'BUKIT TIMAH':[0], 'CENTRAL AREA':[0], 'CHOA CHU KANG':[0], 'CLEMENTI':[0], 'GEYLANG':[0], 'HOUGANG':[0], 'JURONG EAST':[0], 'JURONG WEST':[0], 'KALLANG/WHAMPOA':[1], 'MARINE PARADE':[0], 'PASIR RIS':[0], 'PUNGGOL':[0], 'QUEENSTOWN':[0], 'SEMBAWANG':[0], 'SENGKANG':[0], 'SERANGOON':[0], 'TAMPINES':[0], 'TOA PAYOH':[0], 'WOODLANDS':[0], 'YISHUN':[0], 'Adjoined flat':[0], 'Apartment':[0], 'DBSS':[0], 'Improved':[0], 'Improved-Maisonette':[0], 'Maisonette':[0], 'Model A':[1], 'Model A-Maisonette':[0], 'Model A2':[0], 'Multi Generation':[0], 'New Generation':[0], 'Premium Apartment':[0], 'Premium Apartment Loft':[0], 'Premium Maisonette':[0], 'Simplified':[0], 'Standard':[0], 'Terrace':[0], 'Type S1':[0], 'Type S2':[0],})
new_test_data = new_test_data.values
new_test_data = hdb_model.scaler.transform(new_test_data)
predicted_flat_price = hdb_model.main_model.predict(new_test_data)[0][0]
percent_error = 100*((listing_price-predicted_flat_price)/(listing_price))

result_text5 = "\nThe listed price on PropertyGuru is S${a} while the price predicted by the model is S${b:.2f}. The percentage error is {c:.2f}%\n".format(a=listing_price, b=predicted_flat_price, c=percent_error)
results = result_text1+result_text2+result_text3+result_text4+result_text5
hdb_model.save_text(results, "hdb_results{a}.txt".format(a=random_num))


