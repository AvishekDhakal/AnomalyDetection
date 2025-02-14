import joblib

model1 = joblib.load("logistic_reg.pkl")
model2 = joblib.load("random_forest.pkl")
model3 = joblib.load("xgb.pkl")
if hasattr(model1, "feature_names_in_"):
    print("logistic",model1.feature_names_in_)
if hasattr(model2, "feature_names_in_"):
    print("random",model1.feature_names_in_)
if hasattr(model3, "feature_names_in_"):
    print("xgb",model1.feature_names_in_)
