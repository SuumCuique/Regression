# -*- coding: utf-8 -*-
from prettytable import PrettyTable
import matplotlib.pyplot as plot
from scipy.stats import shapiro
import statsmodels.api as sm
import scipy.stats as st
import pandas as pd
import numpy
import math
import os

def cls():
    os.system(['clear','cls'][os.name == 'nt'])

def func(data):
    #шапиро-уилка
    stat = shapiro(data)
    cls()
    Yi = []
    temp = 0
    CalcF = 0
    dataX, dataY = [],[]
    hiQuadr,SStot = 0,0
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

    for item in CalcY: CalcF += (item-SrAriph)**2
    CalcF *= (len(dataX) - 2)
    for counter in range(0,len(dataY)): 
        temp += (dataY[counter]-CalcY[counter])**2
    CalcF /= temp #F расчетное из критерия Фишера

    #-----------------table------------------------------
    table = PrettyTable(['Значение оценки параметра', 'Среднее отклонение', 'Доверительный интервал','P-значение','Значимость'])
    yp = (1/len(dataY))*sum(dataY)
    for c in range(len(dataY)): SStot += (dataY[c]-yp)**2
    for item in range(0,len(dataY)):
        Yi.append((dataY[item]-CalcY[item])**2)
    print('R2 = ', 1-sum(Yi)/SStot)
    kDet = 1-sum(Yi)/SStot
    Fstat = (kDet/(len(dataX)-1) / ((1-kDet)/len(dataX)))
    print('F-stat = ',Fstat)
    
    for counter in range(0,len(CalcY)):
        hiQuadr += ((dataY[counter]-CalcY[counter])**2)/CalcY[counter] # p-value
    print('Хи-квадрат для P-value = ', hiQuadr)

    ostDispers = sum(Yi)/(len(dataX)-1)
    print('Остаточная дисперсия = ', ostDispers)

 #построение диаграммы остатков
    xOstatki, resOstatki, ostatki  = [],[],[]
    for c in range(len(dataY)):
        ostatki.append(dataY[c] - CalcY[c])
    uchastki = numpy.array_split(ostatki, 10)
    i=1
    for c in uchastki:
        resOstatki.append(math.fabs(sum(c)))
        xOstatki.append(len(c)*i)
        i+=1

    

    print("Значение критерия Шапиро-Уилки", stat)
    
    fig = plot.figure() 
    ax = plot.subplot(221)
    plot.title('Сумма остатков на 10 равных участках')
    plot.bar([0,1,2,3,4,5,6,7,8,9], resOstatki  )
    ax = plot.subplot(222)
    plot.title('Остатки vs полученные значения, \nY полученное red')
    plot.plot(dataY, color='r', label='Y полученное red')
    plot.plot(ostatki)
    plot.show()

    interval = st.t.interval(0.95, len(CalcY)-1, loc=numpy.mean(CalcY), scale=st.sem(CalcY))
    table.add_row([a,numpy.std(CalcY),interval,hiQuadr,Fstat])
    table.add_row([b,numpy.std(CalcY),interval,hiQuadr,Fstat])
    print(table)

data = pd.read_csv('abalone.csv', sep=',', encoding='utf-8')
func(list(map( lambda x: [x[1],x[2]],data.to_numpy())))