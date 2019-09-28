train_df = pd.read_csv(r"C:\Users\PV140\Desktop\python_tut\Mango\Mango\datainfo.csv")
test_df = pd.read_csv(r"C:\Users\PV140\Desktop\python_tut\Mango\Mango\test.csv")

train_df.info()
train_df.describe()
train_df.head()

total = train_df.isnull().sum().sort_values(ascending=False)
print(total)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
print(percent_1)
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
print(percent_2)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

train_df.columns.values


X_train = train_df.drop("taste", axis=1)
print(X_train)
Y_train = train_df["taste"]
print(Y_train)
X_test  = test_df.drop("sr no", axis=1).copy()
print(X_test)

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(round(acc_sgd,2,), "%")


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")