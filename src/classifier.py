from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

clf = RandomForestClassifier(n_estimators=200, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
# compute accuracy/precision/recall/F1