import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



def download_dataset():
    iris=load_iris()
    iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
    iris_df['target']=iris.target
    return iris_df

def split_dataset(df):
    y=df['target']
    x=df.drop(['target'],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=111)
    return x_train,x_test,y_train,y_test

def train_model(x,y):
    model=RandomForestClassifier(n_estimators=100)
    model.fit(x,y)
    return model

def model_inference(model,x_test,y_test):
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    # disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true", cmap=plt.cm.Blues)
    # plt.savefig("plot.png")
    return accuracy,cm



if __name__=="__main__":
    df=download_dataset()
    x_train,x_test,y_train,y_test=split_dataset(df)
    model=train_model(x_train,y_train)
    accuracy,cm=model_inference(model,x_test,y_test)
    with open("metrics.txt","w") as fp:
        fp.write(str(accuracy))



