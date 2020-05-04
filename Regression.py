# -*- coding: utf-8 -*-
import matplotlib.pyplot as plot
import numpy
import scipy.stats as st
from prettytable import PrettyTable

def func(data):
    dataX, dataY = [],[]
    for item in data:
        dataX.append(item[0])
        dataY.append(item[1])
    SumXSq = sum( map(lambda x: x**2,dataX) )
    SumXY = sum ( map(lambda x: x[0]*x[1],data))
    a = (sum(dataY) * SumXSq) - (sum(dataX) * SumXY)
    a/= ( (len(dataX) * SumXSq) - (sum(dataX) * sum(dataX)) )
    b = len(dataX) * SumXY - sum(dataX)*sum(dataY)
    b /= len(dataX)*SumXSq -  (sum(dataX) * sum(dataX))
    CalcY = [] #у расчетное
    for x in dataX: CalcY.append(a+b*x)
    SrAriph = sum(CalcY)/len(CalcY)
    CalcF = 0
    for item in CalcY: CalcF += (item-SrAriph)**2
    CalcF *= (len(dataX) - 2)
    temp = 0
    for counter in range(0,len(dataY)): 
        temp += (dataY[counter]-CalcY[counter])**2
    CalcF /= temp #F расчетное из критерия Фишера

    ############ table
    table = PrettyTable(['Значение оценки параметра', 'Среднее отклонение', 'Доверительный интервал','P-значение','Значимость'])
    SStot = 0 #общая сумма квадратов https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%B4%D0%B5%D1%82%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D0%B0%D1%86%D0%B8%D0%B8
    yp = (1/len(dataY))*sum(dataY)
    for c in range(len(dataY)): SStot += (dataY[c]-yp)**2
    Yi = []
    for item in range(0,len(dataY)):
        Yi.append((dataY[item]-CalcY[item])**2)
    print('R2 = ', 1-sum(Yi)/SStot)
    kDet = 1-sum(Yi)/SStot
    Fstat = (kDet/(len(dataX)-1) / ((1-kDet)/(len(dataX)-len(dataX))))  #https://ru.wikipedia.org/wiki/F-%D1%82%D0%B5%D1%81%D1%82
    hiQuadr = 0
    for counter in range(0,len(CalcY)):
        hiQuadr += ((dataY[counter]-CalcY[counter])**2)/CalcY[counter] # p-value
    interval = st.t.interval(0.95, len(CalcY)-1, loc=numpy.mean(CalcY), scale=st.sem(CalcY))
    table.add_row([a,numpy.std(CalcY),interval,hiQuadr,Fstat])
    table.add_row([b,numpy.std(CalcY),interval,hiQuadr,Fstat])
    print(table)
    print('end')
    #res = numpy.linalg.lstsq(data_x, data_y, rcond=None)
    #print(res)
#    gettedValues = []
  #  for c in data_x:
    #    gettedValues.append(tmp1[2]+tmp1[0]*c[0]+tmp1[1]*c[1])
  #  fig = plot.figure()
   # plot.plot(data_y)
   # plot.plot(gettedValues)
   # plot.show()


import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('abalone.csv', sep=',', encoding='utf-8')
data['Sex'] = data['Sex'].replace("M", 0)
data['Sex'] = data['Sex'].replace('F', 1)
data['Sex'] = data['Sex'].replace('I', 2)

X = data[['Length']]
y = data['Diameter']
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())

func(list(map( lambda x: [x[1],x[2]],data.to_numpy())))