#Pkg.add("Conda")
using Conda
Conda.add("pandas")

using LinearAlgebra
using Statistics
using Optim
using PyCall
using Plots
using Random
using DataFrames

tree = pyimport("sklearn.tree")
gb = pyimport("sklearn.ensemble")
pd = pyimport("pandas")
#dc = tree.DecisionTreeClassifier()
data = pyimport("sklearn.datasets")
tts = pyimport("sklearn.model_selection")
metrics = pyimport("sklearn.metrics")
#plt = pyimport("matplotlib.pyplot")
#joblib = pyimport("joblib")
r2 = metrics.r2_score
np = pyimport("numpy")
#xgb = pyimport("xgboost")

train = tts.train_test_split
boston = data.load_boston()
#diabetes = data.load_diabetes()
california = data.fetch_california_housing()
df4 = pd.read_csv("Datasets/Auto.csv")
df6 = pd.read_csv("Datasets/concrete.csv")
df7 = pd.read_csv("Datasets/diamond.csv")
df8 = pd.read_csv("Datasets/energy.csv")
df9 = pd.read_csv("Datasets/gold.csv")
df10 = pd.read_csv("Datasets/insurance.csv")
df11 = pd.read_csv("Datasets/weatherHistory.csv")

X_1 = boston["data"]
X_2 = california["data"]
y_1 = boston["target"]
y_2 = california["target"]
X_3 = df4.drop("price", axis=1).to_numpy()
y_3 = df4["price"].to_numpy()
X_4 = df6.drop("strength",axis=1).to_numpy()
y_4 = df6["strength"].to_numpy()
X_5 = df7.drop("Price",axis=1).to_numpy()
y_5 = df7["Price"].to_numpy()
X_6 = df8.drop(["Heating Load","Cooling Load"],axis=1).to_numpy()
y_6 = df8["Cooling Load"].to_numpy()
X_7 = df9.drop("Total Turnover",axis=1).to_numpy()
y_7 = df9["Total Turnover"].to_numpy()
X_8 = df10.drop("charges",axis=1).to_numpy()
y_8 = df10["charges"].to_numpy()
X_9 = df11.drop("Temperature (C)",axis=1).to_numpy()
y_9 = df11["Temperature (C)"].to_numpy()

X_train_1, X_test_1, y_train_1, y_test_1 = train(X_1, y_1,test_size=0.2, random_state=0)
X_train_2, X_test_2, y_train_2, y_test_2 = train(X_2, y_2,test_size=0.2, random_state=0)
X_train_3, X_test_3, y_train_3, y_test_3 = train(X_3, y_3,test_size=0.2, random_state=0)
X_train_4, X_test_4, y_train_4, y_test_4 = train(X_4, y_4,test_size=0.2, random_state=0)
X_train_5, X_test_5, y_train_5, y_test_5 = train(X_5, y_5*1.0,test_size=0.2, random_state=0)
X_train_6, X_test_6, y_train_6, y_test_6 = train(X_6, y_6,test_size=0.2, random_state=0)
X_train_7, X_test_7, y_train_7, y_test_7 = train(X_7, y_7,test_size=0.2, random_state=0)
X_train_8, X_test_8, y_train_8, y_test_8 = train(X_8, y_8,test_size=0.2, random_state=0)
X_train_9, X_test_9, y_train_9, y_test_9 = train(X_9, y_9,test_size=0.2, random_state=0)


X_train = Any[[X_train_1] ;[X_train_2] ;[X_train_3] ;[X_train_4] ;[X_train_5] ;[X_train_6] ;[X_train_7]; [X_train_8]; [X_train_9]]
y_train = Any[[y_train_1] ;[y_train_2] ;[y_train_3] ;[y_train_4] ;[y_train_5] ;[y_train_6] ;[y_train_7]; [y_train_8]; [y_train_9]]
X_test = Any[[X_test_1]; [X_test_2]; [X_test_3]; [X_test_4] ;[X_test_5] ;[X_test_6] ;[X_test_7] ;[X_test_8] ;[X_test_9]]
y_test = Any[[y_test_1]; [y_test_2]; [y_test_3]; [y_test_4]; [y_test_5] ;[y_test_6] ;[y_test_7] ;[y_test_8] ;[y_test_9]]
