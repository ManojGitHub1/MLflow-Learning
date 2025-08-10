import mlflow
print(mlflow.get_tracking_uri())

# output - file:///C:/Users/HP/Downloads/ML/MLflow-Learning/mlruns 
# and not http://....
# that's why error is raised when trying to run the code in file1.py because it expects a remote tracking server

print("\n")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(mlflow.get_tracking_uri())
