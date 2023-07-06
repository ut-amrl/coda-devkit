import math
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
def main():
    # #input a directory (assuming it contains the frames from all the trajectories)

    # os.chdir("/robodata/CODa_test/3d_bbox/os1/2/")
    # list_files = os.listdir("/robodata/CODa_test/3d_bbox/os1/2/")
    # # index = 0
    # all_x = []
    # all_y = []
    # for filename in list_files:
    #     # index += 1
    #     labels = labeling(filename)
    #     # print(filename)
    #     # break
    #     print("\nhello world 1\n")
    #     all_x.extend(labels[1])
    #     all_y.extend(labels[2])
    label = labeling()


    # print(labeling(filename))
    # heatmap = sns.kdeplot(x=labels[0], y=labels[1], cmap="Reds", shade=True, bw_adjust=.5)
    # heatmap.figure.savefig("/home/christinaz/treesonly/%s.png"%filename, format='png')
    # heatmap.figure.clf()

    # heatmap = sns.kdeplot(x=all_x, y=all_y, cmap="Reds", shade=True, bw_adjust=.5)
    # heatmap.figure.savefig("/home/christinaz/labeling2/optset_dist.png", format='png')
    # heatmap.figure.clf()


def labeling_with_hist(filename, traj_num, frame_num, class_weather_count):
    #dictionary: jsonfilename : [weather, list of distance/obj, other label]
    weather = "cloudy"
    #if generalize: 
    #frame = get_info_filename(json filename)
    #changedir, open timestamps, (for the correct trajectory), index, returns unix time
    #look up weather
    #distance:
    distance5m = []
    distance15m = []
    distance30m = []
    theta = {}
    obj_count = {}
    f = open(filename)
    data = json.load(f)
    list_x = []
    list_y = []
    coord_list = []
    tree_x = []
    tree_y = []
    # obj_names = ["Pedestrian", "Car", "Bike", "Tree", "Chair", "Table"]
    obj_names = ["Tree"]
    specified_objs_x = []
    specified_objs_y = []
    dynamic_obj = ["Pedestrian", "Horse", "Car", "Pickup Truck", "Delivery Truck", "Service Vehicle", "Utility Vehicle", "Bike", "Scooter", "Motorcycle"]
    dynamic_x = []
    dynamic_y = []
    static_x = []
    static_y = []
    # os.chdir("/home/christinaz")
    # f_2 = open("traj_bbox_count.json")
    # all_bbox_counts = json.load(f_2)

    # os.chdir("/home/christinaz")
    # f_3 = open("segment_bbox_count.json")
    # all_bbox_counts_seg = json.load(f_2)
    ped_distance = {}
    os.chdir("/home/christinaz")
    traj_weather_temp = open("weather_data.json")
    traj_weather = json.load(traj_weather_temp)
    weather_num = 0
    if (traj_weather[traj_num] == "rainy"): 
        weather_num = 1
    elif (traj_weather[traj_num] == "cloudy"):
        weather_num = 2
    elif (traj_weather[traj_num] == "dark"):
        weather_num = 3
    

    for i in data["3dbbox"]:
        
        ins_id = i["instanceId"]
        class_id = i["classId"]
        x_coord = i["cX"]
        y_coord = i["cY"]
        z_coord = i["cZ"]
        # counting bbox (traj)
        # traj = "traj_"+traj_num
        # all_bbox_counts[traj][class_id] = all_bbox_counts[traj][class_id] + 1
        # with open("traj_bbox_count.json", "w") as jsonFile:
        #     jsonFile.seek(0)
        #     json.dump(all_bbox_counts, jsonFile)

        # counting bbox (traj seg)
        # if traj..., then all_bbox_counts_seg[segment]
        
        # all_bbox_counts[traj][class_id] = all_bbox_counts[traj][class_id] + 1
        # with open("traj_bbox_count.json", "w") as jsonFile:
        #     jsonFile.seek(0)
        #     json.dump(all_bbox_counts, jsonFile)

        
        if (class_id == "Tree"):
            tree_x.append(x_coord)
            tree_y.append(y_coord)
        if (class_id in obj_names):
            specified_objs_x.append(x_coord)
            specified_objs_y.append(y_coord)
        if (class_id in dynamic_obj):
            if (class_id == "Pedestrian"):
                ped_distance[ins_id] = [math.sqrt(x_coord ** 2 + y_coord ** 2), x_coord, y_coord]
            else:
                dynamic_x.append(x_coord)
                dynamic_y.append(y_coord)
        else:
            static_x.append(x_coord)
            static_y.append(y_coord)



        # sunny, rainy, cloudy, dark
        class_weather_count[class_id][weather_num] += 1


        list_x.append(x_coord)
        list_y.append(y_coord)
        distance = (x_coord ** 2 + y_coord ** 2 + z_coord ** 2) ** (1/2)
        single_obj = [x_coord, y_coord, distance]
        coord_list.append(single_obj)
        if (distance <= 5) :
            distance5m.append(ins_id)
        elif (distance <= 15) :
            distance15m.append(ins_id)
        elif (distance <= 30) :
            distance30m.append(ins_id)
        radian = math.atan2(y_coord, x_coord)
        theta[ins_id] = radian * 180 / math.pi
        
        allKeys = obj_count.keys()
        if class_id in allKeys:
            obj_count[class_id] += 1
        else:
            obj_count[class_id] = 1
    ped_distance = sorted(ped_distance.items(), key=lambda x:x[1][0])
    if ped_distance:
        ped_distance.pop(0)
    if ped_distance:
        ped_distance.pop(0)
    for ped in ped_distance:
        dynamic_x.append(ped[1][1])
        dynamic_y.append(ped[1][2])
    


# {classid:[distance, xoord, ycoord], classid:[distance, xoord, ycoord], ...}
    # for x_c, y_c in ped_d

    # heatmap = sns.kdeplot(x=list_x, y=list_y, cmap="Reds", shade=True, bw_adjust=.5)
    # heatmap.figure.savefig("/home/christinaz/labeling2/%s.png"%filename, format='png')
    # heatmap.figure.clf()

    labels = {filename: {"weatherCondition":weather, "distance5m": distance5m, "distance15m": distance15m, "distance30m": distance30m, "theta":theta, "objCount":obj_count, "x_coord_list":list_x, "y_coord_list":list_y, "specified_objs_x":specified_objs_x, "specified_objs_y":specified_objs_y, "dynamic_x":dynamic_x, "dynamic_y":dynamic_y, "static_x": static_x, "static_y":static_y}}
    # result = [labels, list_x, list_y]
    # return result
    return labels
    # return tree_x, tree_y

def labeling(filename):
    #dictionary: jsonfilename : [weather, list of distance/obj, other label]
    weather = "cloudy"
    #if generalize: 
    #frame = get_info_filename(json filename)
    #changedir, open timestamps, (for the correct trajectory), index, returns unix time
    #look up weather
    #distance:
    distance5m = []
    distance15m = []
    distance30m = []
    theta = {}
    obj_count = {}
    f = open(filename)
    data = json.load(f)
    list_x = []
    list_y = []
    coord_list = []
    tree_x = []
    tree_y = []
    # obj_names = ["Pedestrian", "Car", "Bike", "Tree", "Chair", "Table"]
    obj_names = ["Tree"]
    specified_objs_x = []
    specified_objs_y = []
    dynamic_obj = ["Horse", "Car", "Pickup Truck", "Delivery Truck", "Service Vehicle", "Utility Vehicle", "Bike", "Scooter", "Motorcycle"]
    dynamic_x = []
    dynamic_y = []
    static_x = []
    static_y = []
    # os.chdir("/home/christinaz")
    # f_2 = open("obj_labels.json")
    # all_obj_counts = json.load(f_2)
    ped_distance = {}
    for i in data["3dbbox"]:
        
        ins_id = i["instanceId"]
        class_id = i["classId"]
        x_coord = i["cX"]
        y_coord = i["cY"]
        z_coord = i["cZ"]
        # counting obj
        # all_obj_counts[class_id] = all_obj_counts[class_id] + 1
        # with open("obj_labels.json", "w") as jsonFile:
        #     jsonFile.seek(0)
        #     json.dump(all_obj_counts, jsonFile)


        if (class_id == "Tree"):
            tree_x.append(x_coord)
            tree_y.append(y_coord)

        if (class_id in obj_names):
            specified_objs_x.append(x_coord)
            specified_objs_y.append(y_coord)

        if (class_id == "Pedestrian"):
            ped_distance[ins_id] = [math.sqrt(x_coord ** 2 + y_coord ** 2), x_coord, y_coord]

        if (class_id in dynamic_obj):
            dynamic_x.append(x_coord)
            dynamic_y.append(y_coord)
            
        if(class_id != "Pedestrian" and class_id not in dynamic_obj):
            static_x.append(x_coord)
            static_y.append(y_coord)


        list_x.append(x_coord)
        list_y.append(y_coord)
        distance = (x_coord ** 2 + y_coord ** 2 + z_coord ** 2) ** (1/2)
        single_obj = [x_coord, y_coord, distance]
        coord_list.append(single_obj)
        if (distance <= 5) :
            distance5m.append(ins_id)
        elif (distance <= 15) :
            distance15m.append(ins_id)
        elif (distance <= 30) :
            distance30m.append(ins_id)
        radian = math.atan2(y_coord, x_coord)
        theta[ins_id] = radian * 180 / math.pi
        
        allKeys = obj_count.keys()
        if class_id in allKeys:
            obj_count[class_id] += 1
        else:
            obj_count[class_id] = 1
    ped_distance = sorted(ped_distance.items(), key=lambda x:x[1][0])
    if ped_distance:
        ped_distance.pop(0)
    if ped_distance:
        ped_distance.pop(0)
    
    for ped in ped_distance:
        dynamic_x.append(ped[1][1])
        dynamic_y.append(ped[1][2])
    # print(dynamic_x)
    # print(dynamic_y)
    #[('Pedestrian:1', [14.583245381589942, 14.05547551443648, 3.887756926910548]), ('Pedestrian:2', [16.06564636838515, 10.018461082294351, -12.559276682060968]), ('Pedestrian:10', [16.548493300679354, 4.434599022603862, -15.943241892141959]), ('Pedestrian:4', [24.437298581799016, -22.121402279988516, -10.383887669987345])]
    


# {classid:[distance, xoord, ycoord], classid:[distance, xoord, ycoord], ...}
    # for x_c, y_c in ped_d

    # heatmap = sns.kdeplot(x=list_x, y=list_y, cmap="Reds", shade=True, bw_adjust=.5)
    # heatmap.figure.savefig("/home/christinaz/labeling2/%s.png"%filename, format='png')
    # heatmap.figure.clf()

    labels = {filename: {"weatherCondition":weather, "distance5m": distance5m, "distance15m": distance15m, "distance30m": distance30m, "theta":theta, "objCount":obj_count, "x_coord_list":list_x, "y_coord_list":list_y, "specified_objs_x":specified_objs_x, "specified_objs_y":specified_objs_y, "dynamic_x":dynamic_x, "dynamic_y":dynamic_y, "static_x":static_x, "static_y": static_y}}
    # result = [labels, list_x, list_y]
    # return result
    return labels
    # return tree_x, tree_y


if __name__ == '__main__':
    main()


