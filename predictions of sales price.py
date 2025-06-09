import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

print("Rows and columns of training csv:", train.shape)
print("Rows and columns of testing csv:", test.shape)

print("Testing the missing values for test csv:\n", test.isnull().sum().sort_values(ascending=False).head(49))

cat_cols = train.select_dtypes(include=['object']).columns.tolist()
test[cat_cols] = test[cat_cols].fillna('Missing')
train[cat_cols] = train[cat_cols].fillna('Missing')

num = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']

train[num] = train[num].fillna(train[num].median())
test[num] = test[num].fillna(test[num].median())

x = train[num]
y = train['SalePrice']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

pred = model.predict(x_val)

testing = test[num]
test_prediction = model.predict(testing)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_val, pred)
r2 = r2_score(y_val, pred)
print("Validation MSE:", mse)
print("Validation R² score:", r2)

output = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_prediction})
output.to_csv('submission.csv', index=False)
print("Test predictions saved!")
