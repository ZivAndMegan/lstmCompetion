# 1、鼠标移动速度，平均及整体？
# 2、轨迹范围？，是否一直在向目标靠近？下一步和上一步的关系？（有效移动比例？）
# 3、移动数量？
import math

import numpy
from scipy.stats import mode

import dataTraitor

pathName = "D:\\bigData\\dsjtzs_txfz_training.txt"

maxColone,listData_humain,listData_robot,listData_humain_time,listData_robot_time,lenthToFinal_humain,lenthToFinal_robot = dataTraitor.parseTrainData(pathName)
listData_humain = numpy.array(listData_humain)
listData_robot = numpy.array(listData_robot)

def speed_total_traitor(listData_point,listData_time):
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

def speed_split_traitor(listData_point,listData_time):
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
            time_minus = (listData_time[i][j]-pre_time)+0.0001
            speed = 0.6
            if time_minus!=0:
                speed = lenth/time_minus
            pre_data = listData_point[i][j]
            pre_time = listData_time[i][j]
            speed_split.append(speed)
        listSpeed.append(speed_split[1:])
    return listSpeed

def track_data_traitor(listData_point):
    track_scope = []
    for one_data_point in listData_point:
        maxOneLigne_x = 0
        minOneLigne_x = 999999999
        maxOneLigne_y = 0
        minOneLigne_y = 999999999
        for every_point in one_data_point:
            if maxOneLigne_x<every_point[0]:
                maxOneLigne_x = every_point[0]
            if minOneLigne_x>every_point[0]:
                minOneLigne_x = every_point[0]
            if maxOneLigne_y<every_point[1]:
                maxOneLigne_y = every_point[1]
            if minOneLigne_y>every_point[1]:
                minOneLigne_y = every_point[1]
        track_scope.append([maxOneLigne_x-minOneLigne_x,maxOneLigne_y-minOneLigne_y])
    return track_scope
        
    

    

def analyse_total_humain_robot(): 
    resultSpeed_humain = speed_total_traitor(listData_humain,listData_humain_time)
    resultSpeed_robot = speed_total_traitor(listData_robot,listData_robot_time)
    print(len(resultSpeed_humain))
    print(len(resultSpeed_robot))
    print ("人类速度平均：",numpy.mean(resultSpeed_humain))
    print ("人类速度中位数：",numpy.median(resultSpeed_humain))
    print ("人类速度众数：",mode(resultSpeed_humain))
    print ("机器速度平均：",numpy.mean(resultSpeed_robot))
    print ("机器速度中位数：",numpy.median(resultSpeed_robot))
    print ("机器速度众数：",mode(resultSpeed_robot))

def analyse_split_humain_3():
    resultSpeed_humain_split = speed_split_traitor(listData_humain,listData_humain_time)
    list_median = []
    for one_speed_humain_split in resultSpeed_humain_split:
        if(len(one_speed_humain_split)>3):
            list_medianTmp = []
            one_speed_humain_split_1 = one_speed_humain_split[0:int(len(one_speed_humain_split)/3)]
            one_speed_humain_split_2 = one_speed_humain_split[int(len(one_speed_humain_split)/3):int(2*len(one_speed_humain_split)/3)]
            one_speed_humain_split_3 = one_speed_humain_split[int(2*len(one_speed_humain_split)/3):]
            list_medianTmp.append(numpy.median(one_speed_humain_split_1))
            list_medianTmp.append(numpy.median(one_speed_humain_split_2))
            list_medianTmp.append(numpy.median(one_speed_humain_split_3))
            list_median.append(list_medianTmp)
    print (list_median)
    return list_median


def analyse_split_robot_3():
    resultSpeed_robot_split = speed_split_traitor(listData_robot,listData_robot_time)
    list_median = []
    for one_speed_robot_split in resultSpeed_robot_split:
        if(len(one_speed_robot_split)>3):
            list_medianTmp = []
            one_speed_robot_split_1 = one_speed_robot_split[0:int(len(one_speed_robot_split)/3)]
            one_speed_robot_split_2 = one_speed_robot_split[int(len(one_speed_robot_split)/3):int(2*len(one_speed_robot_split)/3)]
            one_speed_robot_split_3 = one_speed_robot_split[int(2*len(one_speed_robot_split)/3):]
            list_medianTmp.append(numpy.median(one_speed_robot_split_1))
            list_medianTmp.append(numpy.median(one_speed_robot_split_2))
            list_medianTmp.append(numpy.median(one_speed_robot_split_3))
            list_median.append(list_medianTmp)
    print (list_median)
    return list_median

def analyse_track_ulti(result_track_data_humain):
    result_traited_x = []
    result_traited_y = []
    num_1500_x = 0
    num_1000_x = 0
    num_500_x = 0
    num_0_x = 0
    num_1500_y = 0
    num_1000_y = 0
    num_500_y = 0
    num_0_y = 0
    for one_result in result_track_data_humain:
        result_traited_x.append(one_result[0])
        result_traited_y.append(one_result[1])
        if one_result[0]>1500:
            num_1500_x = num_1500_x+1
        elif one_result[0]>1000:
            num_1000_x = num_1000_x+1
        elif one_result[0]>500:
            num_500_x = num_500_x+1
        else:
            num_0_x = num_0_x+1
        if one_result[1]>1500:
            num_1500_y = num_1500_y+1
        elif one_result[1]>1000:
            num_1000_y = num_1000_y+1
        elif one_result[1]>500:
            num_500_y = num_500_y+1
        else:
            num_0_y = num_0_y+1
    return result_traited_x,result_traited_y,[num_1500_x,num_1000_x,num_500_x,num_0_x],[num_1500_y,num_1000_y,num_500_y,num_0_y]

def analyse_track():
    result_track_data_humain = track_data_traitor(listData_humain)
    result_traited_x,result_traited_y,scope_result_x,scope_result_y = analyse_track_ulti(result_track_data_humain)
    print ("人移动的横坐标范围的中位数是:", numpy.median(result_traited_x))
    print ("人移动的纵坐标范围的中位数是:", numpy.median(result_traited_y))
    print ("人移动的横坐标范围的>1500,1000-1500,500-1000,<500:", scope_result_x)
    print ("人移动的纵坐标范围的>1500,1000-1500,500-1000,<500:", scope_result_y)
    result_track_data_robot = track_data_traitor(listData_robot)
    result_traited_x,result_traited_y,scope_result_x,scope_result_y = analyse_track_ulti(result_track_data_robot)
    print ("机器移动的横坐标范围的中位数是:", numpy.median(result_traited_x))
    print ("机器移动的纵坐标范围的中位数是:", numpy.median(result_traited_y))
    print ("机器移动的横坐标范围的>1500,1000-1500,500-1000,<500:", scope_result_x)
    print ("机器移动的纵坐标范围的>1500,1000-1500,500-1000,<500:", scope_result_y)

    print ("如果机器的轨迹同比例放大，横纵坐标范围如下:")
    print ("机器移动的横坐标范围的>1500,1000-1500,500-1000,<500:", numpy.array(scope_result_x)*(2600/400))
    print ("机器移动的纵坐标范围的>1500,1000-1500,500-1000,<500:", numpy.array(scope_result_y)*(2600/400))

def analyse_num_userful_util(list_lenth_data):
    list_userful = []
    for one_data_list in list_lenth_data:
        perLenth = 99999999
        one_userful_list = []
        for one_data in one_data_list:
            if one_data[2]<perLenth:
                one_userful_list.append(1)
            else:
                one_userful_list.append(0)
            perLenth = one_data[2]
        list_userful.append(one_userful_list)
    return list_userful


def analyse_num_userful():
    result_useful = analyse_num_userful_util(lenthToFinal_humain)
    result_useful = numpy.array(result_useful)
    
    print (result_useful)
    print ("人移动的有效的次数为:",result_useful.count(1))
    print ("人移动的无效的次数为:",result_useful.count(0))
    print ("人移动的有效次数比例为:",result_useful.count(1)/len(result_useful))

    result_useful_robot = analyse_num_userful_util(lenthToFinal_robot)
    print ("机器移动的有效的次数为:",result_useful_robot.count(1))
    print ("机器移动的无效的次数为:",result_useful_robot.count(0))
    print ("机器移动的有效次数比例为:",result_useful_robot.count(1)/len(result_useful))    
# analyse_track()
analyse_num_userful()