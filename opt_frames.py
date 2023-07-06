import labeling
import os
import json
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

#sunny, cloudy, rainy, dark
weather_total = [0, 0, 0, 0]
list_weather_vec = []
#5m, 15m, 30m
distance_total = [0, 0, 0]
list_distance_vec = []
#I, II, III, IV
theta_total = [0, 0, 0, 0]
list_theta_vec = []
#tree, pole, person, car, bike, chair, table
class_total = [0, 0, 0, 0, 0, 0, 0]
list_class_vec = []
#cost
cost_matrix = []
#index:[filename, cost]
index_to_cost = {}

all_labels = {}
file_to_cost = {}
index_to_file_cost = {}
user_num_frames = 10

def main():
    #when reading from metadata file, split filename (find trajectory# and chdir)
    #reading in user input (json)
    f = open("input.json")
    input_file = json.load(f)
    

    os.chdir("/robodata/arthurz/Datasets/CODa/metadata/")
    list_metadata_files = os.listdir("/robodata/arthurz/Datasets/CODa/metadata/")
    all_files = []
    traj_to_frame = {}

    training_files = []
    validation_files = []
    testing_files = []
    
    counter = 0
    print(list_metadata_files)
    for metadata_file in list_metadata_files:
        # if(metadata_file != "6.json" and metadata_file != "13.json"):
        if(metadata_file != "6.json"):
        # if(metadata_file != "13.json"):
            print("##########")
            print(metadata_file)
            print("##########")
            f = open(metadata_file)
            metadata = json.load(f)
            # trajectory = metadata["trajectory"]
            # for json_file in metadata["ObjectTracking"]["training"]:
            trajectory_files = metadata["ObjectTracking"]["training"]
            # print(trajectory_files)
            temp = metadata["ObjectTracking"]["training"]
            # training_files.extend(temp.split("/")[3])
            training_files.extend(temp)

            trajectory_files.extend(metadata["ObjectTracking"]["validation"])
            # print(trajectory_files)
            temp = metadata["ObjectTracking"]["validation"]
            # validation_files.extend(temp.split("/")[3])
            validation_files.extend(temp)

            trajectory_files.extend(metadata["ObjectTracking"]["testing"])
            temp = metadata["ObjectTracking"]["testing"]
            # testing_files.extend(temp.split("/")[3])
            testing_files.extend(temp)

            all_files.extend(trajectory_files)
            traj_to_frame[metadata["trajectory"]] = len(trajectory_files)
            sum_labels(trajectory_files)
            os.chdir("/robodata/arthurz/Datasets/CODa/metadata/")
            counter+=1
            # break
    # cost(all_files)
    # print(validation_files)
    # print(opt_set(file_to_cost))
    # weather_distribution(traj_to_frame)
    # print(all_files[50:60])
    # temp_files = ['3d_bbox_os1_11_2183.json', '3d_bbox_os1_11_2041.json', '3d_bbox_os1_11_2047.json', '3d_bbox_os1_11_2033.json', '3d_bbox_os1_11_2029.json', '3d_bbox_os1_11_1841.json', '3d_bbox_os1_11_1802.json', '3d_bbox_os1_11_1811.json', '3d_bbox_os1_11_2181.json', '3d_bbox_os1_11_1989.json']

    # check_dist(temp_files, all_labels)

    training_set(training_files)
    validation_set(validation_files)
    testing_set(testing_files)

def training_set(training_files):
    x_coord = []
    y_coord = []
    # print(training_files)
    for file in training_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
        # x_coord.extend(file_labels["dynamic_x"])
        # y_coord.extend(file_labels["dynamic_y"])
        x_coord.extend(file_labels["static_x"])
        y_coord.extend(file_labels["static_y"])
    
    # r_list, theta_list = cart2pol(x_coord, y_coord)
    # N = len(r_list)
    # colors = theta_list
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    # c = ax.scatter(theta_list, r_list, c="k", alpha=0.75)
    # name = "polarscatter3"
    # plt.savefig("/home/christinaz/polar_plots/%s.png"%name, format='png')

    # heatmap = sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", shade=True, bw_adjust=.5)
    # a = [-0.5, -0.3]
    # b = [-0.5, 0.3]
    # c = [0.5, -0.3]
    # d = [0.5, 0.3]
    # width = c[0] - a[0]
    # height = d[1] - a[1]
    # name = "specified_objs"
    # heatmap = sns.kdeplot(x=[-0.5, -0.5, 0.5, 0.5], y=[-0.3, 0.3, -0.3, 0.3], cmap="Blues", bw_adjust=.5)
    # # filename = "traj11_test"
    # heatmap.figure.savefig("/home/christinaz/traj11/%s.png"%name, format='png')
    # heatmap.figure.clf()

    fig = plt.figure(figsize=(10, 10))
    fig2 = plt.figure(figsize=(10,10))
    grid_ratio = 8
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])
    print("x_coords" + str(len(x_coord)))
    print("y_coords" + str(len(y_coord)))
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.4)
    plt.rcParams.update({'font.size': 22})
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d m'))

    ax_joint.set_aspect('equal', adjustable='box')  # equal aspect ratio is needed for a polar plot
    plt.gca().axis('off')
    xmin, xmax = ax_joint.get_xlim()
    xrange = max(-xmin, xmax)
    plt.gca().set_xlim(-30, 30)  # force 0 at center
    ymin, ymax = ax_joint.get_ylim()
    yrange = max(-ymin, ymax)
    plt.gca().set_ylim(-30, 30)  # force 0 at center
    

    ax_polar = fig2.add_subplot(projection='polar')
    ax_polar.set_facecolor('none')  # make transparent
    # ax_polar.set_position(pos=ax_joint.get_position())
    # ax_polar.set_rlim(0, max(xrange, yrange))
    ax_polar.set_yticks(np.arange(0,35,5))
    name = "StaticTraining"
    plt.savefig("/home/arshgamare/polar_plots/%s.png"%name, format='png')
    return

def cart2pol(x, y):
    r_result = []
    theta_result = []
    for x_pt, y_pt in zip(x, y):
        r = np.sqrt(x_pt**2 + y_pt**2)
        theta = np.arctan2(y_pt, x_pt)
        r_result.append(r)
        theta_result.append(theta)
    return(r_result, theta_result)

def validation_set(validation_files):
    x_coord = []
    y_coord = []
    for file in validation_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
        # x_coord.extend(file_labels["dynamic_x"])
        # y_coord.extend(file_labels["dynamic_y"])
        x_coord.extend(file_labels["static_x"])
        y_coord.extend(file_labels["static_y"])
    # r_list, theta_list = cart2pol(x_coord, y_coord)
    # N = len(r_list)
    # colors = theta_list
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    # c = ax.scatter(theta_list, r_list, c="k", alpha=0.75)
    # name = "polarscatter3"
    # plt.savefig("/home/christinaz/polar_plots/%s.png"%name, format='png')

    # heatmap = sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", shade=True, bw_adjust=.5)
    # a = [-0.5, -0.3]
    # b = [-0.5, 0.3]
    # c = [0.5, -0.3]
    # d = [0.5, 0.3]
    # width = c[0] - a[0]
    # height = d[1] - a[1]
    # name = "specified_objs"
    # heatmap = sns.kdeplot(x=[-0.5, -0.5, 0.5, 0.5], y=[-0.3, 0.3, -0.3, 0.3], cmap="Blues", bw_adjust=.5)
    # # filename = "traj11_test"
    # heatmap.figure.savefig("/home/christinaz/traj11/%s.png"%name, format='png')
    # heatmap.figure.clf()

    fig = plt.figure(figsize=(10, 10))
    fig2 = plt.figure(figsize=(10,10))
    grid_ratio = 8
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.4)
    plt.rcParams.update({'font.size': 22})

    ax_joint.set_aspect('equal', adjustable='box')  # equal aspect ratio is needed for a polar plot
    plt.gca().axis('off')
    xmin, xmax = ax_joint.get_xlim()
    xrange = max(-xmin, xmax)
    plt.gca().set_xlim(-30, 30)  # force 0 at center
    ymin, ymax = ax_joint.get_ylim()
    yrange = max(-ymin, ymax)
    plt.gca().set_ylim(-30, 30)  # force 0 at center
    

    ax_polar = fig2.add_subplot(projection='polar')
    ax_polar.set_facecolor('none')  # make transparent
    # ax_polar.set_position(pos=ax_joint.get_position())
    # ax_polar.set_rlim(0, max(xrange, yrange))
    ax_polar.set_yticks(np.arange(0,35,5))
    name = "StaticValidation"
    plt.savefig("/home/arshgamare/polar_plots/%s.png"%name, format='png')
    return

def testing_set(testing_files):
    x_coord = []
    y_coord = []
    for file in testing_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
        # x_coord.extend(file_labels["dynamic_x"])
        # y_coord.extend(file_labels["dynamic_y"])
        x_coord.extend(file_labels["static_x"])
        y_coord.extend(file_labels["static_y"])
    # r_list, theta_list = cart2pol(x_coord, y_coord)
    # N = len(r_list)
    # colors = theta_list
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    # c = ax.scatter(theta_list, r_list, c="k", alpha=0.75)
    # name = "polarscatter3"
    # plt.savefig("/home/christinaz/polar_plots/%s.png"%name, format='png')

    # heatmap = sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", shade=True, bw_adjust=.5)
    # a = [-0.5, -0.3]
    # b = [-0.5, 0.3]
    # c = [0.5, -0.3]
    # d = [0.5, 0.3]
    # width = c[0] - a[0]
    # height = d[1] - a[1]
    # name = "specified_objs"
    # heatmap = sns.kdeplot(x=[-0.5, -0.5, 0.5, 0.5], y=[-0.3, 0.3, -0.3, 0.3], cmap="Blues", bw_adjust=.5)
    # # filename = "traj11_test"
    # heatmap.figure.savefig("/home/christinaz/traj11/%s.png"%name, format='png')
    # heatmap.figure.clf()

    fig = plt.figure(figsize=(10, 10))
    fig2 = plt.figure(figsize=(10,10))
    grid_ratio = 8
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.4)
    plt.rcParams.update({'font.size': 22})

    ax_joint.set_aspect('equal', adjustable='box')  # equal aspect ratio is needed for a polar plot
    plt.gca().axis('off')
    xmin, xmax = ax_joint.get_xlim()
    xrange = max(-xmin, xmax)
    plt.gca().set_xlim(-30, 30)  # force 0 at center
    ymin, ymax = ax_joint.get_ylim()
    yrange = max(-ymin, ymax)
    plt.gca().set_ylim(-30, 30)  # force 0 at center
    

    ax_polar = fig2.add_subplot(projection='polar')
    ax_polar.set_facecolor('none')  # make transparent
    # ax_polar.set_position(pos=ax_joint.get_position())
    # ax_polar.set_rlim(0, max(xrange, yrange))
    ax_polar.set_yticks(np.arange(0,35,5))
    name = "StaticTesting"
    plt.savefig("/home/arshgamare/polar_plots/%s.png"%name, format='png')
    return


def opt_set(file_to_cost):
    # print(file_to_cost)
    file_to_cost = sorted(file_to_cost.items(), key=lambda x:x[1])
    # print(dict(file_to_cost).values())
    opt_set = list(dict(file_to_cost).keys())[:user_num_frames]

    # check_dist(opt_set, all_labels)
    # print(opt_set)
    # check_dist(temp_5_files, all_labels)
    # print(temp_5_files)
    # print("-------")
    # print(temp_5_files)
    return opt_set

#add dark / rainy
def weather_distribution(traj_to_frame):
    os.chdir("/home/christinaz/")
    f = open("weather_data.json")
    weather_data = json.load(f)
    distr = {"sunny":0, "cloudy":0, "clear":0}
    for traj in traj_to_frame:
        # print(traj)
        distr[weather_data[str(traj)]] += traj_to_frame[traj]
    # print(list(distr.keys()))
    # print(list(distr.values()))
    plt.bar(list(distr.keys()), list(distr.values()))
    plt.savefig("/home/christinaz/weather_distribution/weather.png", format='png')

    return 

def cost(list_files):
    index_num = 1
    for frame in range(len(list_files)):
        cost = 0
        #assuming target distribution = x
        # user_weather = [0.0, 0.25, 0.0, 0.0]

        # user_distance = [0.1, 0.6, 0.3]
        # user_theta = [0.5, 0.15, 0.15, 0.2]
        # user_class = [0.25, 0.25, 0.3, 0.1, 0.1]

        user_distance = [0.07518797, 0.43609023, 0.4887218]
        user_theta = [0.15288221, 0.20551378, 0.29072682, 0.35087719]
        user_class = [0.31937173, 0.5078534, 0, 0.08376963, 0.08900524, 0, 0] #added Chair, Table
        
        # weather_total = [10, 20, 5, 15]
        # weather_vec = list_weather_vec[frame]

        # weather_vec = list(map(truediv, list_weather_vec[frame], weather_total))

        # weather_vec = np.subtract(np.array(weather_vec), np.array(user_weather))
        # cost += np.linalg.norm(weather_vec)
        

        # distance_vec = list(map(truediv, list_distance_vec[frame], distance_total))
        distance_vec = np.array(list_distance_vec[frame]) / sum(list_distance_vec[frame])
        distance_vec = np.subtract(np.array(distance_vec), np.array(user_distance))
        cost += np.linalg.norm(distance_vec)
        # print(list_distance_vec[frame])

        # theta_vec = list(map(truediv, list_theta_vec[frame], theta_total))
        theta_vec = np.array(list_theta_vec[frame]) / sum(list_theta_vec[frame])
        theta_vec = np.subtract(np.array(theta_vec), np.array(user_theta))
        cost += np.linalg.norm(theta_vec)
        # print(list_theta_vec[frame])

        # class_vec = list(map(truediv, list_class_vec[frame], class_total))
        class_vec = np.array(list_class_vec[frame]) / (sum(list_class_vec[frame]) + 1)
        class_vec = np.subtract(np.array(class_vec), np.array(user_class))
        cost += np.linalg.norm(class_vec)


        # index_to_cost[frame].append(cost)
        file_to_cost[list_files[frame]] = cost
        index_to_file_cost[index_num] = [list_files[frame], cost]
        index_num += 1
    return

def sum_labels(list_files):
    for filename in list_files:
        #strip the name from the full path passed in
        traj_num = filename.split("/")[2]
        filename = filename.split("/")[3]

        os.chdir("/robodata/arthurz/Datasets/CODa/3d_bbox/os1/%s/" % traj_num)
        labels = labeling.labeling(filename)
        all_labels.update(labels)
        labels = labels[filename]
        # if(labels['weatherCondition'] == 'sunny'):
        #     weather_total[0] += 1
        #     list_weather_vec.append([1, 0, 0, 0])
        # elif (labels['weatherCondition'] == 'cloudy'):
        #     weather_total[1] += 1
        #     list_weather_vec.append([0, 1, 0, 0])
        # elif (labels['weatherCondition'] == 'rainy'):
        #     weather_total[2] += 1
        #     list_weather_vec.append([0, 0, 1, 0])
        # else:
        #     weather_total[3] += 1
        #     list_weather_vec.append([0, 0, 0, 1])
        
        #list_distance_vec : [[0,0,0],[1,1,1],[2,2,2]]

        distance_total[0] += len(labels["distance5m"])
        distance_total[1] += len(labels["distance15m"])
        distance_total[2] += len(labels["distance30m"])

        per_frame_dist = [len(labels["distance5m"]), len(labels["distance15m"]), len(labels["distance30m"])]
        list_distance_vec.append(per_frame_dist)

        per_frame_theta = [0, 0, 0, 0]
        for obj in labels["theta"].keys():
            if (labels["theta"][obj] >= 0 and labels["theta"][obj] <= 90):
                theta_total[0] += 1
                per_frame_theta[0] += 1
            elif (labels["theta"][obj] <= 180 and labels["theta"][obj] > 90):
                theta_total[1] += 1
                per_frame_theta[1] += 1
            elif (labels["theta"][obj] <= -90 and labels["theta"][obj] >= -180):
                theta_total[2] += 1
                per_frame_theta[2] += 1
            else:
                theta_total[3] += 1
                per_frame_theta[3] += 1
        list_theta_vec.append(per_frame_theta)

        per_frame_obj = [0, 0, 0, 0, 0, 0, 0]
        for obj in labels["objCount"]:
            if(obj == "Tree"):
                class_total[0] += labels["objCount"][obj]
                per_frame_obj[0] += labels["objCount"][obj]
            elif (obj == "Pole"):
                class_total[1] += labels["objCount"][obj]
                per_frame_obj[1] += labels["objCount"][obj]
            elif (obj == "Person"):
                class_total[2] += labels["objCount"][obj]
                per_frame_obj[2] += labels["objCount"][obj]
            elif (obj == "Car"):
                class_total[3] += labels["objCount"][obj]
                per_frame_obj[3] += labels["objCount"][obj]
            elif (obj == "Bike"):
                class_total[4] += labels["objCount"][obj]
                per_frame_obj[4] += labels["objCount"][obj]
            elif (obj == "Chair"):
                class_total[5] += labels["objCount"][obj]
                per_frame_obj[5] += labels["objCount"][obj]
            elif (obj == "Table"):
                class_total[6] += labels["objCount"][obj]
                per_frame_obj[6] += labels["objCount"][obj]
        list_class_vec.append(per_frame_obj)
    return




def check_dist(opt_set, all_labels):
        weather_count = [0, 0, 0, 0]
        distance_count = [0, 0, 0]
        theta_count = [0, 0, 0, 0]
        class_count = [0, 0, 0, 0, 0, 0, 0]
        for filename in opt_set:
            # labels = labeling.labeling(filename)
            labels = all_labels[filename]
            # print(labels)
            # weather_count[1] += 1
            # labels = labels[filename]
            distance_count[0] += len(labels['distance5m'])
            distance_count[1] += len(labels["distance15m"])
            distance_count[2] += len(labels["distance30m"])
            for obj in labels["theta"].keys():
                if (labels["theta"][obj] >= 0 and labels["theta"][obj] <= 90):
                    theta_count[0] += 1
                elif (labels["theta"][obj] <= 180 and labels["theta"][obj] > 90):
                    theta_count[1] += 1
                elif (labels["theta"][obj] <= -90 and labels["theta"][obj] >= -180):
                    theta_count[2] += 1
                else:
                    theta_count[3] += 1
            for obj in labels["objCount"].keys():
                if(obj == "Tree"):
                    class_count[0] += labels["objCount"][obj]
                elif (obj == "Pole"):
                    class_count[1] += labels["objCount"][obj]
                elif (obj == "Person"):
                    class_count[2] += labels["objCount"][obj]
                elif (obj == "Car"):
                    class_count[3] += labels["objCount"][obj]
                elif (obj == "Bike"):
                    class_count[4] += labels["objCount"][obj]
                elif (obj == "Chair"):
                    class_count[5] += labels["objCount"][obj]
                elif (obj == "Table"):
                    class_count[6] += labels["objCount"][obj]
         

        # print(" ")
        # print(" ")
        # print("selected_files count-------")
        # print("weather count: " + str(weather_count))
        # print("distance count: " + str(distance_count)  )
        # print("theta count: " + str(theta_count)   ) 
        # print("class count" + str(class_count) )
        # print("-------")
        # print(" ")
        # weather_count = list(map(truediv, weather_count, weather_total)) 
        # distance_count = list(map(truediv, distance_count, distance_total))
        # theta_count = list(map(truediv, theta_count, theta_total)) 
        #class_count = list(map(truediv, class_count, class_total))  
        distance_count = np.array(distance_count) / sum(distance_count)
        theta_count = np.array(theta_count) / sum(theta_count) 
        class_count = np.array(class_count) / sum(class_count)

        # print("selected_files distribution-------")
        # print("weather distribution" + str(weather_count))
        # print("distance distribution" + str(distance_count)  )
        # print("theta distribution" + str(theta_count)    )
        # print("class distribution" + str(class_count) )
        # print("-------")
        # print(" ")
        # print("total count---------")
        # print("weather total count" + str(weather_total))
        # print("distance total count" + str(distance_total))
        # print("theta total count" + str(theta_total))
        # print("class total count" + str(class_total))
        # print("-------")


if __name__ == '__main__':
    main()
