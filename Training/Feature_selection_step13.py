
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

y_np = data.train_y.cpu().numpy()
clf = ExtraTreesClassifier(n_estimators=50, random_state=42)  # fewer trees to save RAM
clf.fit(X_reduced, y_np)

selector_model = SelectFromModel(clf, prefit=True, threshold="median")
X_selected = selector_model.transform(X_reduced)

print("Final selected feature shape:", X_selected.shape)
