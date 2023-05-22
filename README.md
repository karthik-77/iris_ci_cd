# iris_ci_cd
Example for CI/CD using github actions. The sample script iris_model.py downloads the iris data and splits it into train,test.
Train data is used to train a random forest classifier model with 100 Trees. The model is tested with test data and results are
published as comment for the workflow. The results include accuracy score and confusion matrix for the model.
