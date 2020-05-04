# -*- coding: utf-8 -*-
import matplotlib.pyplot as plot
import numpy
import scipy.stats as st



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
#значение оценки параметра - CalcY
#numpy.std(CalcY[i]) #среднее отклонение
#st.t.interval(0.95, len(CalcY)-1, loc=numpy.mean(CalcY), scale=st.sem(CalcY)) #доверительный интервал
#hiQuadr+=math.pow((dataY[c]-CalcY[c]), 2)/gettedValues[c] # p-value

    

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
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('abalone.csv', sep=',', encoding='utf-8')
data['Sex'] = data['Sex'].replace("M", 0)
data['Sex'] = data['Sex'].replace('F', 1)
data['Sex'] = data['Sex'].replace('I', 2)

X = data[['Length']]
y = data['Diameter']
est = sm.OLS(y, X).fit()
print(est.summary())

func(list(map( lambda x: [x[1],x[2]],data.to_numpy())))