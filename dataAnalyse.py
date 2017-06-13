# 1、鼠标移动速度，平均及整体？
# 2、轨迹范围？，是否一直在向目标靠近？下一步和上一步的关系？（有效移动比例？）
# 3、移动数量？
import dataTraitor
import math
import numpy
from scipy.stats import mode

pathName = "D:\\bigData\\dsjtzs_txfz_training.txt"

maxColone,listData_humain,listData_robot,listData_humain_time,listData_robot_time = dataTraitor.parseTrainData(pathName)
listData_humain = numpy.array(listData_humain)
listData_robot = numpy.array(listData_robot)

def speedAnalyse_total(listData_point,listData_time):
    listLenth = []
    listTime = []
    listSpeed = []
    for data_humain in listData_point:
        lenth = 0
        pre_data = data_humain[0]
        for sequence in data_humain:
           cha = numpy.subtract(sequence,pre_data)
           pingfang = numpy.square(cha)
           lenth = lenth + math.sqrt(numpy.sum(pingfang)) 
           pre_data = sequence
        listLenth.append(lenth)
    for data_humain_time in listData_time:
        timeConsume =  data_humain_time[-1]-data_humain_time[0]
        listTime.append(timeConsume)
    totalSpeed = []
    for i in range(len(listLenth)):
        if listLenth[i]!=0.0 and listTime[i]!=0.0:
            totalSpeed.append(listLenth[i]/listTime[i])
        elif listLenth[i]==0.0:
            print ("错误数据Lenth:",i)
        elif listTime[i]==0.0:
            print ("错误数据Time:",i) 
    return totalSpeed

def speedAnalyse_split(listData_point,listData_time):
    listLenth = []
    listTime = []
    listSpeed = []
    for i in range(len(listData_point)):
        speed_split = []
        pre_data = listData_point[0]
        pre_time = listData_time[0][0]
        for j in range(len(listData_point[i])):
            cha = numpy.subtract(listData_point[i][j],pre_data)
            pingfang = numpy.square(cha)
            lenth = math.sqrt(numpy.sum(pingfang))
            speed = lenth/((listData_time[i][j]-pre_time)+0.0001)
            pre_data = listData_point[i][j]
            pre_time = listData_time[i][j]
            speed_split.append(speed)
        listSpeed.append(speed_split[1:])
    return listSpeed
    


# resultSpeed_humain = speedAnalyse_total(listData_humain,listData_humain_time)
# resultSpeed_robot = speedAnalyse_total(listData_robot,listData_robot_time)
# print(len(resultSpeed_humain))
# print(len(resultSpeed_robot))
# print ("人类速度平均：",numpy.mean(resultSpeed_humain))
# print ("人类速度中位数：",numpy.median(resultSpeed_humain))
# print ("人类速度众数：",mode(resultSpeed_humain))
# print ("机器速度平均：",numpy.mean(resultSpeed_robot))
# print ("机器速度中位数：",numpy.median(resultSpeed_robot))
# print ("机器速度众数：",mode(resultSpeed_robot))

resultSpeed_humain_split = speedAnalyse_split(listData_humain,listData_humain_time)
list_Speed_split_max = []
for i in range(len(resultSpeed_humain_split)):
    if(len(resultSpeed_humain_split[i])>0):
        list_Speed_split_max.append(numpy.max(resultSpeed_humain_split[i]))
print(list_Speed_split_max)
     
