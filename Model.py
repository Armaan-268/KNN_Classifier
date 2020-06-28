# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Importing the dataset
    x_train = pd.read_csv('Diabetes_XTrain.csv')
    y_train = pd.read_csv('Diabetes_YTrain.csv')
    x_train = x_train.iloc[1:,[0,1,2,3,4,5,6,7]].values
    y_train = y_train.iloc[1:,0].values
    y_train = y_train.reshape(len(y_train),1)
    X = pd.read_csv('Diabetes_Xtest.csv')
    X = X.iloc[1:,[0,1,2,3,4,5,6,7]].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.25, random_state = 0)
    
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    y_train = sc_y.fit_transform(y_train)
    x_test = sc_x.transform(x_test)
    
    # Training the K-NN model on the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(x_train.astype('int'), y_train.astype('int'))

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Predicting Results for XTest:
    X = sc_x.transform(X)
    Y = classifier.predict(X)
   
    # Writing Predictions in a file:
    df = pd.DataFrame(Y)
    df.to_csv('Diabetes_YPred.csv',index=False)
        
if __name__ == "__main__":
    main()