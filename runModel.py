import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, date
from collections import OrderedDict

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


def runModel(models, weights, mydf, test_size, confidence, future_mon):
    features_to_be_removed = ["dom", "isHoliday", "holidayNearby"]
    mydf = mydf.drop(features_to_be_removed, axis=1)
    X = mydf.drop(["Count", "ArrivalDate"], axis=1)
    y = mydf["Count"]

    '''split data'''
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    '''fit model'''
    for model, w in zip(models, weights):
        model.fit(X_train, y_train)
        try:
            y_pred += w*model.predict(X_test)
        except:
            y_pred = w*model.predict(X_test)
    y_pred /= sum(weights)

    ######
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    '''Generate PLot data'''
    avg_val = y_test.mean()

    outp = f"**RMSE**: {sqrt(mean_squared_error(y_test, y_pred)):.2f}"
    outp += f"\n\n**R^2**: {r2_score(y_test, y_pred):.2f}"
    relative_errors = abs((y_test - y_pred)/avg_val) # y_test.replace(0, avg_val)
    correct_preds = (relative_errors <= confidence)
    mydic = dict(correct_preds.value_counts())
    outp += f"\n\n**Average percent Error**: {relative_errors.values.mean()*100:.2f}%"

    try: true = mydic[True]
    except: true = 0
    try: false = mydic[False]
    except: false = 0
    accuracy = 100*true/(true+false)

    outp += f"\n\n**Prediction Accuracy**: {accuracy:.2f}%  {str(mydic)}"

    '''feature importance'''
    model = LinearRegression()
    model.fit(X_train, y_train)
    var_imp = list(X_train.columns), list(model.coef_)
    dic = {var_imp[0][i]: var_imp[1][i] for i in range(len(list(model.coef_)))}
    var_imp = OrderedDict(sorted(dic.items(), key=lambda x: x[1], reverse=True))

    '''Future Forecast'''
    date_range = pd.date_range(start=date(2020, 1, 1), end=date(2020, 12, 31))
    X_forecast = pd.DataFrame(date_range, columns=["ArrivalDate"])
    X_forecast["ArrivalDate"] = pd.to_datetime(X_forecast["ArrivalDate"])
    X_forecast = genDerivedFeatures(X_forecast)
    X_forecast = X_forecast[X_forecast[future_mon] == 1]

    model.fit(X, y)
    y_forecast = model.predict(X_forecast[list(X.columns)])


    ## return
    return outp, y_test, y_pred, mydf.ArrivalDate[-test_size:], var_imp, y_forecast, X_forecast
    


def getEDA(property):
    prop_details = {
"Americas Best Value Inn Charlotte, NC" : '''**PropertyType** : Franchise\n
**PropertyCity** : Charlotte\n
**PropertyState **: NC\n
**PropertyCountry** : USA
''',
"Americas Best Value Inn & Suites Bismarck": '''**PropertyType** : Franchise\n
**PropertyCity** : Bismarck\n
**PropertyState** : ND\n
**PropertyCountry** : USA
''',
"Americas Best Value Inn Midlothian Cedar Hill": '''**PropertyType** : Franchise\n
**PropertyCity** : Midlothian\n
**PropertyState** : TX\n
**PropertyCountry** : USA
''',
"Americas Best Value Inn & Suites Boise": '''**PropertyType** : Franchise\n
**PropertyCity** : Boise\n
**PropertyState** : ID\n
**PropertyCountry** : USA
''',
"Red Lion Inn & Suites Modesto": '''**PropertyType** : Franchise\n
**PropertyCity** : Modesto\n
**PropertyState** : CA\n
**PropertyCountry** : USA
''',
"Knights Inn Virginia Beach on 29th St": '''**PropertyType** : Franchise\n
**PropertyCity** : VirginiaBeach\n
**PropertyState** : VA\n
**PropertyCountry** : USA
''',
'Red Lion Hotel on the River Jantzen Beach': '''**PropertyType** : Franchise\n
**PropertyCity** : Portland\n
**PropertyState** : OR\n
**PropertyCountry** : USA
''',
'Americas Best Value Inn Scarborough Portland' : '''**PropertyType** : Franchise\n
**PropertyCity** : Scarborough\n
**PropertyState** : ME\n
**PropertyCountry** : USA
''',
'Americas Best Value Inn Nashville Downtown' : '''**PropertyType** : Franchise\n
**PropertyCity** : Nashville\n
**PropertyState** : TN\n
**PropertyCountry** : USA
''',
'Americas Best Value Inn Nashville Airport S' : '''**PropertyType** : Franchise\n
**PropertyCity** : Nashville\n
**PropertyState** : TN\n
**PropertyCountry** : USA
''',
'Americas Best Value Inn & Suites Melbourne' : '''**PropertyType** : Licensed\n
**PropertyCity** : Melbourne\n
**PropertyState** : FL\n
**PropertyCountry** : USA
''',
'Americas Best Value Inn Lake Tahoe-Tahoe City' : '''**PropertyType** : Franchise\n
**PropertyCity** : Tahoe City\n
**PropertyState** : CA\n
**PropertyCountry** : USA
'''
}
    return prop_details[property]



def getHolidayProximity(ind, holidays_ind):
    distances = [abs(i-ind) for i in holidays_ind]
    least_2_dist = sorted(distances)[:2]
    avg_dist = 2*np.prod(least_2_dist)/sum(least_2_dist)
    avg_dist = min(distances)
    return 1/(1+avg_dist)


def genDerivedFeatures(mydf):
    ## Create derived features
    mydf["doy"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%j')).astype("int")
    mydf["dom"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%d')).astype("int")
    mydf["dow"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%a'))
    mydf["isWeekend"] = mydf.dow.isin(['Sat','Sun']).astype("int")
    mydf["month"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%b'))
    mydf["woy"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%U')).astype("int")

    mydf["presentYear"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%y') in ["19", "20"]).astype("int")
    mydf["previousYear"] = mydf.ArrivalDate.apply(lambda x: x.strftime('%y')=="18").astype("int")

    ## One hot encode DOW and Months
    mydf = pd.concat([mydf, pd.get_dummies(mydf["dow"])], axis=1).drop("dow", axis=1)
    mydf = pd.concat([mydf, pd.get_dummies(mydf["month"])], axis=1).drop("month", axis=1)

    ## isHoliday
    holidays = pd.read_csv("US_holidays.csv")
    holidays.date = pd.to_datetime(holidays.date)
    mydf["isHoliday"] = mydf.ArrivalDate.apply(lambda x: x in list(holidays.date)).astype("int")

    ## holidayProximity
    holidays_ind = list(mydf[mydf.isHoliday == 1].index)
    mydf["ind"] = mydf.index
    mydf["holidayProximity"] = mydf["ind"].apply(lambda x: getHolidayProximity(x, holidays_ind))
    mydf = mydf.drop("ind", axis=1)
    mydf["holidayNearby"] = (mydf.holidayProximity > 0.3).astype("int")

    return mydf


def prepData(property):
    df = pd.read_csv(property+".csv")

    df["ArrivalDate"] = pd.to_datetime(df["ArrivalDate"])
    df = df.sort_values(by=['ArrivalDate'])
    df = df[df.ArrivalDate <= "2019-12-01"]
    '''Fill missing dates'''
    r = pd.date_range(start=df.ArrivalDate.min(), end=df.ArrivalDate.max())
    df = df.set_index('ArrivalDate').reindex(r).fillna(0).rename_axis('ArrivalDate').reset_index()

    ''''Create derived features'''
    df = genDerivedFeatures(df)

    '''Do feature Engineering'''
    
    return df


if __name__ == "__main__":
    pass




