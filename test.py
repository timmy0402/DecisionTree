import pandas as pd
from DecisionTree import *
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("./train.csv")
train_data.describe()

y = train_data["Survived"]

features = ["Age", "SibSp"]
X = train_data[features]

imp = SimpleImputer(missing_values=pd.NA, strategy="mean")

X_imputed = imp.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

tree = DecisionTree()
tree.fit(X_imputed_df, y.to_numpy())
ans = tree.predict(X_imputed_df)

print(y.sum())
print(np.sum(ans))

# print(ans)


model = DecisionTreeClassifier()
model.fit(X_imputed_df, y)
prediction = model.predict(X_imputed_df)
# print(prediction)
