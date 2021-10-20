# Databricks notebook source
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn

# COMMAND ----------

!/databricks/python3/bin/python -m pip install --upgrade pip

# COMMAND ----------

data  = load_iris()

# COMMAND ----------

data

# COMMAND ----------

data.target

# COMMAND ----------

X=data.data
y = data.target

# COMMAND ----------

X

# COMMAND ----------

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=10)

# COMMAND ----------

y_train

# COMMAND ----------

with mlflow.start_run():
    dtc = DecisionTreeClassifier(random_state=10)
    dtc.fit(X_train,y_train)
    y_pred_class = dtc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred_class)
    
    print(accuracy)
    mlflow.log_param("random_state",10)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.sklearn.log_model(dtc,"model")
    modelpath = "/dbfs/mlflow/iris2/model-%s-%f" %("decision_tree",1)
    mlflow.sklearn.save_model(dtc,modelpath)
    
    

# COMMAND ----------

with mlflow.start_run():
    dtc = DecisionTreeClassifier(max_depth=1,random_state=10)
    dtc.fit(X_train,y_train)
    y_pred_class = dtc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred_class)
    
    print(accuracy)
    mlflow.log_param("random_state",10)
    mlflow.log_param("max_depth",1)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.sklearn.log_model(dtc,"model")
    modelpath = "/dbfs/mlflow/iris2/model-%s-%f" %("decision_tree",2)
    mlflow.sklearn.save_model(dtc,modelpath)
    
    

# COMMAND ----------

with mlflow.start_run():
    dtc = DecisionTreeClassifier(max_depth=1,min_samples_split=5,random_state=10)
    dtc.fit(X_train,y_train)
    y_pred_class = dtc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred_class)
    
    print(accuracy)
    mlflow.log_param("random_state",10)
    mlflow.log_param("max_depth",1)
    mlflow.log_param("min_samples_split",5)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.sklearn.log_model(dtc,"model")
    modelpath = "/dbfs/mlflow/iris2/model-%s-%f" %("decision_tree",3)
    mlflow.sklearn.save_model(dtc,modelpath)
    
    

# COMMAND ----------

with mlflow.start_run():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    
    print(accuracy)
    mlflow.log_param("n_neighbors",5)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.sklearn.log_model(knn,"model")
    modelpath = "/dbfs/mlflow/iris2/model-%s-%f" %("KNN",4)
    mlflow.sklearn.save_model(knn,modelpath)
    
    

# COMMAND ----------

with mlflow.start_run():
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    
    print(accuracy)
    mlflow.log_param("n_neighbors",2)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.sklearn.log_model(knn,"model")
    modelpath = "/dbfs/mlflow/iris2/model-%s-%f" %("KNN",5)
    mlflow.sklearn.save_model(knn,modelpath)
    
    

# COMMAND ----------

mlflow.search_runs()

# COMMAND ----------

run_idl = "5bcc9e7531044f6ead9ef5b15b7b493e"
model_url = "runs:/" + run_idl + "/model"

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri = model_url)

# COMMAND ----------

model

# COMMAND ----------

model.get_params()

# COMMAND ----------

