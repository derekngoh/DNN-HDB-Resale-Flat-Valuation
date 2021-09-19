import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import Utils

class HDB_Model():
    def __init__(self) -> None:
        self.main_model = None
        self.original_data = None
        self.original_x = None
        self.original_y = None
        self.scaler = None
        self.xtrain = None
        self.xtest = None
        self.scaled_xtrain = None
        self.scaled_xtest = None
        self.ytrain = None
        self.ytest = None

    def set_data(self,data, y_column_name):
        self.original_data = data
        self.original_x = data.drop(y_column_name, axis=1).values
        self.original_y = data[y_column_name].values

    def set_xy_train_test(self,test_proportion,seed=11):
        xtrain, xtest, ytrain, ytest = train_test_split(self.original_x, self.original_y, test_size=test_proportion, random_state=seed)
        self.xtrain = np.array(xtrain)
        self.xtest = np.array(xtest)
        self.ytrain = np.array(ytrain)
        self.ytest = np.array(ytest)

    def scale_x_data(self):
        scaler = MinMaxScaler()
        scaler.fit(self.xtrain)
        self.scaled_xtrain = scaler.transform(self.xtrain)
        self.scaled_xtest = scaler.transform(self.xtest)
        self.scaler = scaler

    def create_model(self, nodes, dropout=0.5):
        model = Sequential()

        model.add(Dense(nodes, activation='relu')) 
        model.add(Dropout(dropout))

        model.add(Dense(nodes, activation='relu')) 
        model.add(Dropout(dropout))

        model.add(Dense(nodes, activation='relu')) 
        model.add(Dropout(dropout))

        model.add(Dense(nodes, activation='relu')) 
        model.add(Dropout(dropout))

        model.add(Dense(nodes, activation='relu')) 
        model.add(Dropout(dropout))

        model.add(Dense(1,))

        model.compile(optimizer='adam', loss='mse')
        self.main_model = model

    def fit_model(self,patience,epochs,batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        self.main_model.fit(self.scaled_xtrain, 
                            self.ytrain,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(self.scaled_xtest,self.ytest),
                            callbacks=[early_stop]
                            )

    def save_loss_plot(self,filename,show=False,figsize=(15,6)):
        Utils.save_show_loss_plot(self.main_model,filename,show=show,figsize=figsize)

    def set_current_path(self,filename=None):
        return Utils.get_set_current_path(filename)

    def save_text(self,content, filename):
        Utils.save_as_text_file(content,filename)