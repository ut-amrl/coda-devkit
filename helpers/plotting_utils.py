import math
import os
import json
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import numpy as np
import seaborn as sns
import pandas as pd

from helpers.constants import *
from helpers.sensors import set_filename_dir, get_filename_info

def labeling_with_hist(filepath, traj_num, frame_num, class_weather_count):
    # print("Opening filepath %s" % filepath)
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
    f = open(filepath)
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

    traj_weather_temp = open("%s/helpers/helper_utils/weather_data.json"% os.getcwd())
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

    filename = filepath.split('/')[-1]
    labels = {filename: {"weatherCondition":weather, "distance5m": distance5m, "distance15m": distance15m, 
                "distance30m": distance30m, "theta":theta, "objCount":obj_count, "x_coord_list":list_x, 
                "y_coord_list":list_y, "specified_objs_x":specified_objs_x, "specified_objs_y":specified_objs_y, 
                "dynamic_x":dynamic_x, "dynamic_y":dynamic_y, "static_x": static_x, "static_y":static_y}}
    return labels

def sum_labels(indir, input_dict, list_files, class_weather_count):

    for anno_subpath in list_files:
        filename = anno_subpath.split('/')[-1]
        modality, sensor_name, traj_num, frame_num =get_filename_info(anno_subpath.split('/')[-1])
        filepath = set_filename_dir(indir, modality, sensor_name, traj_num, frame_num)

        labels = labeling_with_hist(filepath, traj_num, frame_num, class_weather_count)
        input_dict['all_labels'].update(labels)
        labels = labels[filename]

        input_dict['distance_total'][0] += len(labels["distance5m"])
        input_dict['distance_total'][1] += len(labels["distance15m"])
        input_dict['distance_total'][2] += len(labels["distance30m"])

        per_frame_dist = [len(labels["distance5m"]), len(labels["distance15m"]), len(labels["distance30m"])]
        input_dict['list_distance_vec'].append(per_frame_dist)

        per_frame_theta = [0, 0, 0, 0]
        for obj in labels["theta"].keys():
            if (labels["theta"][obj] >= 0 and labels["theta"][obj] <= 90):
                input_dict['theta_total'][0] += 1
                per_frame_theta[0] += 1
            elif (labels["theta"][obj] <= 180 and labels["theta"][obj] > 90):
                input_dict['theta_total'][1] += 1
                per_frame_theta[1] += 1
            elif (labels["theta"][obj] <= -90 and labels["theta"][obj] >= -180):
                input_dict['theta_total'][2] += 1
                per_frame_theta[2] += 1
            else:
                input_dict['theta_total'][3] += 1
                per_frame_theta[3] += 1
        input_dict['list_theta_vec'].append(per_frame_theta)

        per_frame_obj = [0] * len(class_weather_count) # NUMCLASSES x 1
        for obj in labels["objCount"]:
            assert obj in BBOX_CLASS_TO_ID, "Object %s not found in bbox classes" % obj
            obj_idx = BBOX_CLASS_TO_ID[obj]
            input_dict['class_total'][obj_idx] += labels["objCount"][obj]
            per_frame_obj[obj_idx] += labels["objCount"][obj]

        input_dict['list_class_vec'].append(per_frame_obj)
    return input_dict

def kdeplot_set(training_files, outdir, all_labels, split="training", object_type="dynamic"):
    x_coord = []
    y_coord = []
    for file in training_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
    
        x_coord.extend(file_labels["%s_x"%object_type])
        y_coord.extend(file_labels["%s_y"%object_type])
    
    fig = plt.figure(figsize=(8, 8))
    grid_ratio = 5
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])

    # sns.kdeplot(data=df, x='x', y='y', bw_adjust=0.7, linewidths=1, ax=ax_joint)
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.5, clip=[-30, 30])

    ax_joint.set_aspect('equal', adjustable='box')  # equal aspect ratio is needed for a polar plot
    ax_joint.axis('off')
    xmin, xmax = ax_joint.get_xlim()
    xrange = max(-xmin, xmax)
    ax_joint.set_xlim(-xrange, xrange)  # force 0 at center
    ymin, ymax = ax_joint.get_ylim()
    yrange = max(-ymin, ymax)
    ax_joint.set_ylim(-yrange, yrange)  # force 0 at center

    ax_polar = fig.add_subplot(projection='polar')
    ax_polar.set_facecolor('none')  # make transparent
    ax_polar.set_position(pos=ax_joint.get_position())
    ax_polar.set_rlim(0, max(xrange, yrange))
    name = "%s_%s_all" % (object_type, split)
    plt.savefig("%s/%s.png"%(outdir, name), format='png')

def labeling(filename, dynamic_class_list=["Horse", "Car", "Pickup Truck", "Delivery Truck", 
    "Service Vehicle", "Utility Vehicle", "Bike", "Scooter", "Motorcycle"]):
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
    dynamic_obj = dynamic_class_list
    dynamic_x = []
    dynamic_y = []
    static_x = []
    static_y = []
    ped_distance = {}
    for i in data["3dbbox"]:
        
        ins_id = i["instanceId"]
        class_id = i["classId"]
        x_coord = i["cX"]
        y_coord = i["cY"]
        z_coord = i["cZ"]

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

    labels = {filename: {"weatherCondition":weather, "distance5m": distance5m, "distance15m": distance15m, 
        "distance30m": distance30m, "theta":theta, "objCount":obj_count, "x_coord_list":list_x, 
        "y_coord_list":list_y, "specified_objs_x":specified_objs_x, "specified_objs_y":specified_objs_y, 
        "dynamic_x":dynamic_x, "dynamic_y":dynamic_y, "static_x":static_x, "static_y": static_y}}
    return labels


