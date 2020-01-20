import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


def runModel(models, weights, mydf, test_size, confidence):
    mydf = mydf.drop(["doy", "isHoliday", "holidayNearby"], axis=1)
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
    avg_val = y_test.mean()
    
    outp = "**RMSE**: {:.2f}".format(sqrt(mean_squared_error(y_test, y_pred)))
    outp += "\n\n**R^2**: {:.2f}".format(r2_score(y_test, y_pred))
    relative_errors = abs((y_test - y_pred)/avg_val) # y_test.replace(0, avg_val)
    correct_preds = (relative_errors <= confidence)
    mydic = dict(correct_preds.value_counts())
    outp += "\n\n**Average percent Error**: {:.2f}%".format(relative_errors.values.mean()*100)

    try: true = mydic[True]
    except: true = 0
    try: false = mydic[False]
    except: false = 0

    accuracy = 100*true/(true+false)

    outp += "\n\n**Prediction Accuracy**: {:.2f}%  {}".format(accuracy, str(mydic))

    '''feature importance'''
    model = LinearRegression()
    model.fit(X_train, y_train)
    var_imp = list(X_train.columns), list(model.coef_)
    dic = {var_imp[0][i]: var_imp[1][i] for i in range(len(list(model.coef_)))}
    var_imp = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))

    return outp, y_test, y_pred, mydf.ArrivalDate[-test_size:], var_imp
    


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
'''}
    return prop_details[property]



def getHolidayProximity(ind, holidays_ind):
    distances = [abs(i-ind) for i in holidays_ind]
    least_2_dist = sorted(distances)[:2]
    avg_dist = 2*np.prod(least_2_dist)/sum(least_2_dist)
    avg_dist = min(distances)
    return 1/(1+avg_dist)

def prepData(property):
    df = pd.read_csv("../ReservationData/"+property+".csv")

    df["ArrivalDate"] = pd.to_datetime(df["ArrivalDate"])
    df = df.sort_values(by=['ArrivalDate'])
    df = df[df.ArrivalDate <= "2019-12-01"]
    '''Fill missing dates'''
    r = pd.date_range(start=df.ArrivalDate.min(), end=df.ArrivalDate.max())
    df = df.set_index('ArrivalDate').reindex(r).fillna(0).rename_axis('ArrivalDate').reset_index()

    ## Create derived features
    df["doy"] = df.ArrivalDate.apply(lambda x: x.strftime('%j')).astype("int")
    df["dom"] = df.ArrivalDate.apply(lambda x: x.strftime('%d')).astype("int")
    df["dow"] = df.ArrivalDate.apply(lambda x: x.strftime('%a'))
    df["isWeekend"] = df.dow.isin(['Sat','Sun']).astype("int")
    df["month"] = df.ArrivalDate.apply(lambda x: x.strftime('%b'))
    df["woy"] = df.ArrivalDate.apply(lambda x: x.strftime('%U')).astype("int")

    df["presentYear"] = df.ArrivalDate.apply(lambda x: x.strftime('%y')=="19").astype("int")
    df["previousYear"] = df.ArrivalDate.apply(lambda x: x.strftime('%y')=="18").astype("int")

    ## One hot encode DOW and Months
    df = pd.concat([df, pd.get_dummies(df["dow"])], axis=1).drop("dow", axis=1)
    df = pd.concat([df, pd.get_dummies(df["month"])], axis=1).drop("month", axis=1)

    ## isHoliday
    holidays = pd.read_csv("../ReservationData/US_holidays.csv")
    holidays.date = pd.to_datetime(holidays.date)
    df["isHoliday"] = df.ArrivalDate.apply(lambda x: x in list(holidays.date)).astype("int")
    ## holidayProximity
    holidays_ind = list(df[df.isHoliday == 1].index)
    df["ind"] = df.index
    df["holidayProximity"] = df["ind"].apply(lambda x: getHolidayProximity(x, holidays_ind))
    df = df.drop("ind", axis=1)
    df["holidayNearby"] = (df.holidayProximity > 0.3).astype("int")

    '''Do feature Engineering'''
    
    return df


if __name__ == "__main__":
    pass




