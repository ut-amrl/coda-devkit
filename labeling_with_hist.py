import math
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import itertools
import cv2

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
#[sunny, rainy, cloudy, dark]
class_weather_count = {"Pedestrian": [0,0,0,0], "Horse": [0,0,0,0], "Car": [0,0,0,0], "Pickup Truck": [0,0,0,0], "Delivery Truck": [0,0,0,0], "Service Vehicle": [0,0,0,0], "Utility Vehicle": [0,0,0,0], "Bike": [0,0,0,0], "Scooter": [0,0,0,0], "Motorcycle": [0,0,0,0], "Fire Hydrant": [0,0,0,0], "Fire Alarm": [0,0,0,0], "Parking Kiosk": [0,0,0,0], "Mailbox": [0,0,0,0], "Newspaper Dispenser": [0,0,0,0], "Sanitizer Dispenser": [0,0,0,0], "Condiment Dispenser": [0,0,0,0], "ATM": [0,0,0,0], "Vending Machine": [0,0,0,0], "Door Switch": [0,0,0,0], "Emergency Aid Kit": [0,0,0,0], "Fire Extinguisher": [0,0,0,0], "Emergency Phone": [0,0,0,0], "Computer": [0,0,0,0], "Television": [0,0,0,0], "Dumpster": [0,0,0,0], "Trash Can": [0,0,0,0], "Vacuum Cleaner": [0,0,0,0], "Cart": [0,0,0,0], "Chair": [0,0,0,0], "Couch": [0,0,0,0], "Bench": [0,0,0,0], "Table": [0,0,0,0], "Bollard": [0,0,0,0], "Construction Barrier": [0,0,0,0], "Fence": [0,0,0,0], "Railing": [0,0,0,0], "Cone": [0,0,0,0], "Stanchion": [0,0,0,0], "Traffic Light": [0,0,0,0], "Traffic Sign": [0,0,0,0], "Traffic Arm": [0,0,0,0], "Canopy": [0,0,0,0], "Bike Rack": [0,0,0,0], "Pole": [0,0,0,0], "Informational Sign": [0,0,0,0], "Wall Sign": [0,0,0,0], "Door": [0,0,0,0], "Floor Sign": [0,0,0,0], "Room Label": [0,0,0,0], "Freestanding Plant": [0,0,0,0], "Tree": [0,0,0,0], "Other": [0,0,0,0]}

outdir="/robodata/arthurz/Research/coda"

def sum_labels(list_files):
    for filename in list_files:
        #strip the name from the full path passed in
        # print(filename)
        traj_num = filename.split("/")[2]
        filename = filename.split("/")[3]
        frame_num = (filename.split("_")[4]).split(".")[0]
        # print(frame_num)
        
        os.chdir("/robodata/arthurz/Datasets/CODa/3d_bbox/os1/%s/" % traj_num)
        labels = labeling.labeling(filename, traj_num, frame_num, class_weather_count)
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

def main():
    #when reading from metadata file, split filename (find trajectory# and chdir)
    #reading in user input (json)
    # os.chdir("/home/christinaz/paper_barcharts/")
    # combine_images()
    # return
    # f = open("input.json")
    # input_file = json.load(f)
    

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
        if (metadata_file != "6.json"):
            print(metadata_file)
            counter += 1
            # print(metadata_file)
            # if (metadata_file == "14.json"):
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
            # if (counter == 13): break
        
    # cost(all_files)

    # print(opt_set(file_to_cost))
    # weather_distribution(traj_to_frame)
    # print(all_files[50:60])
    # temp_files = ['3d_bbox_os1_11_2183.json', '3d_bbox_os1_11_2041.json', '3d_bbox_os1_11_2047.json', '3d_bbox_os1_11_2033.json', '3d_bbox_os1_11_2029.json', '3d_bbox_os1_11_1841.json', '3d_bbox_os1_11_1802.json', '3d_bbox_os1_11_1811.json', '3d_bbox_os1_11_2181.json', '3d_bbox_os1_11_1989.json']

    # check_dist(temp_files, all_labels)

    # training_set(training_files)
    # validation_set(validation_files)
    # testing_set(testing_files)
    print("-----------------")
    # print(class_weather_count)
    training_set(training_files)
    validation_set(validation_files)
    testing_set(testing_files)


def class_and_weather():
    class_weather_count = {'Pedestrian': [21908, 14521, 14511, 9862], 'Chair': [8563, 0, 25076, 559], 'Table': [2356, 0, 8503, 148], 'Railing': [6103, 447, 8035, 3094], 'Pole': [9089, 2675, 13489, 6552], 'Tree': [7966, 2942, 9947, 3889], 'Horse': [0, 0, 470, 0], 'Car': [1542, 91, 2550, 390], 'Pickup Truck': [0, 0, 156, 0], 'Delivery Truck': [459, 0, 138, 0], 'Service Vehicle': [0, 0, 1549, 0], 'Utility Vehicle': [0, 112, 1019, 33], 'Bike': [5020, 2151, 2720, 555], 'Scooter': [838, 576, 555, 0], 'Motorcycle': [90, 0, 0, 0], 'Fire Hydrant': [0, 199, 192, 377], 'Fire Alarm': [93, 0, 186, 0], 'Parking Kiosk': [0, 0, 0, 200], 'Mailbox': [0, 0, 0, 0], 'Newspaper Dispenser': [0, 479, 0, 0], 'Sanitizer Dispenser': [162, 0, 276, 0], 'Condiment Dispenser': [0, 0, 0, 0], 'ATM': [61, 0, 0, 0], 'Vending Machine': [0, 0, 0, 0], 'Door Switch': [0, 0, 371, 0], 'Emergency Aid Kit': [0, 0, 146, 0], 'Fire Extinguisher': [0, 0, 345, 0], 'Emergency Phone': [220, 0, 426, 231], 'Computer': [275, 0, 0, 0], 'Television': [0, 0, 0, 0], 'Dumpster': [255, 0, 0, 0], 'Trash Can': [1495, 320, 2430, 3062], 'Vacuum Cleaner': [0, 0, 0, 0], 'Cart': [0, 0, 39, 0], 'Couch': [816, 0, 324, 0], 'Bench': [378, 148, 457, 1339], 'Bollard': [521, 600, 551, 374], 'Construction Barrier': [0, 0, 0, 0], 'Fence': [150, 0, 199, 0], 'Cone': [200, 0, 0, 0], 'Stanchion': [0, 0, 0, 0], 'Traffic Light': [20, 0, 101, 0], 'Traffic Sign': [1584, 0, 2463, 1668], 'Traffic Arm': [180, 0, 207, 0], 'Canopy': [0, 0, 1788, 0], 'Bike Rack': [1895, 563, 1625, 644], 'Informational Sign': [271, 237, 0, 394], 'Wall Sign': [0, 0, 219, 0], 'Door': [0, 0, 2921, 0], 'Floor Sign': [278, 0, 1569, 400], 'Room Label': [0, 0, 313, 0], 'Freestanding Plant': [674, 0, 2256, 0], 'Other': [672, 626, 1225, 666]}
    weather = ["sunny", "rainy", "cloudy", "dark"]
    # print(class_weather_count.keys())
    # return


    # for key in class_weather_count:
    #     fig = plt.figure()
    #     name = key
    #     plt.bar(weather, class_weather_count[key], color ='blue', width = 0.4)
    #     plt.xlabel("weather")
    #     plt.ylabel("obj count")
    #     plt.title(key)
    #     plt.savefig("/home/christinaz/class_weather/%s.png"%name, format='png')

    sunny_list = []
    rainy_list = []
    cloudy_list = []
    dark_list = []
    #3x3
    obj_name = ['Pedestrian', 'Tree', 'Pole', 'Car', 'Chair', 'Table', 'Railing', 'Horse', 'Pickup Truck', 'Delivery Truck', 'Service Vehicle', 'Utility Vehicle', 'Bike', 'Scooter', 'Motorcycle', 'Fire Hydrant', 'Fire Alarm', 'Parking Kiosk', 'Mailbox', 'Newspaper Dispenser', 'Sanitizer Dispenser', 'Condiment Dispenser', 'ATM', 'Vending Machine', 'Door Switch', 'Emergency Aid Kit', 'Fire Extinguisher', 'Emergency Phone', 'Computer', 'Television', 'Dumpster', 'Trash Can', 'Vacuum Cleaner', 'Cart', 'Couch', 'Bench', 'Bollard', 'Construction Barrier', 'Fence', 'Cone', 'Stanchion', 'Traffic Light', 'Traffic Sign', 'Traffic Arm', 'Canopy', 'Bike Rack', 'Informational Sign', 'Wall Sign', 'Door', 'Floor Sign', 'Room Label', 'Freestanding Plant', 'Other']
    #[sunny, rainy, cloudy, dark]

    for index in range(53):
        sunny_list.append(class_weather_count[obj_name[index]][0])
        rainy_list.append(class_weather_count[obj_name[index]][1])
        cloudy_list.append(class_weather_count[obj_name[index]][2])
        dark_list.append(class_weather_count[obj_name[index]][3])


    # fig, axs = plt.subplots(3, 1)

    first = 0
    last = 7
    counter = 1
    for num in range(8):
        fig = plt.figure()
        # for x_index in range(3):
            # if (last < 54):
        df = pd.DataFrame({'object_name': obj_name[first:last], 'sunny': sunny_list[first:last], 'rainy': rainy_list[first:last], 'cloudy' : cloudy_list[first:last], 'dark' : dark_list[first:last]})
        plot = df.plot(x="object_name", y=['sunny', 'rainy', 'cloudy', 'dark'], kind="bar")
            
        first += 7
        if (last + 7) > 53:
            last = 53
        else:
            last += 7

        # axs[0].bar(obj_name[0:10], sunny_list[0:10], label="sunny")
        # axs[0].bar(obj_name[0:10], rainy_list[0:10], label="rainy")
        # axs[0].bar(obj_name[0:10], cloudy_list[0:10], label="cloudy")
        # axs[0].bar(obj_name[0:10], dark_list[0:10], label="dark")
        # axs[0].set_title('Chart 1')
        # axs[0].legend()
        
                
                

                
        # fig.set_figheight(5)
        # print(obj_name[:9])
        fig = plot.get_figure()
        name = "combined" + str(counter)
        fig.savefig("%s/%s.png"%(outdir, name), format='png', bbox_inches='tight')
        counter += 1
        #53   53-6 = 47 
        #Pedestrian, Chair, Table, Railing, Pole, Tree 
    
    return

def combine_images():
    os.chdir("/home/christinaz/paper_barcharts/")
    img1 = cv2.imread("combined1.png")
    img2 = cv2.imread("combined2.png")
    img3 = cv2.imread("combined3.png")
    img4 = cv2.imread("combined4.png")
    img5 = cv2.imread("combined5.png")
    img6 = cv2.imread("combined6.png")
    img7 = cv2.imread("combined7.png")
    img8 = cv2.imread("combined8.png")
    img_list = [img1, img2, img3, img4, img5, img6, img7, img8]
    new_list = []
    for img in img_list:
        img = cv2.resize(img, (1000, 1000))
        new_list.append(img)
    
    img12 = np.hstack((new_list[0], new_list[1], new_list[2], new_list[3]))
    img34 = np.hstack((new_list[4], new_list[5], new_list[6], new_list[7]))
    result = np.vstack((img12, img34))
    cv2.imwrite("final_image.png", result)
    return

def training_set(training_files):
    x_coord = []
    y_coord = []
    for file in training_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
        # x_coord.extend(file_labels["x_coord_list"])
        # y_coord.extend(file_labels["y_coord_list"])
        # x_coord.extend(file_labels["specified_objs_x"])
        # y_coord.extend(file_labels["specified_objs_y"])

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
    # name = "dynamic"
    # # heatmap = sns.kdeplot(x=[-0.5, -0.5, 0.5, 0.5], y=[-0.3, 0.3, -0.3, 0.3], cmap="Blues", bw_adjust=.5)
    # # filename = "traj11_test"
    # heatmap.figure.savefig("/home/christinaz/traj11/%s.png"%name, format='png')
    # heatmap.figure.clf()

    # r_list, theta_list = cart2pol(x_coord, y_coord)
    # interp = gaussian_kde(np.vstack((theta_list, r_list)))
    # mesh = np.stack(np.meshgrid(theta_list, r_list), 0)
    # colors = interp(mesh.reshape(2, -1)).reshape(len(r_list), len(r_list))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(projection='polar')
    # # ax1.contourf(theta_list, r_list, colors, alpha = 0.75)
    # ax1.contourf(mesh[0], mesh[1], colors, alpha = 0.75)
    # name = "temp2"
    # plt.savefig("/home/christinaz/polar_plots/%s.png"%name, format='png')

    
    fig = plt.figure(figsize=(8, 8))
    grid_ratio = 5
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])

    # sns.kdeplot(data=df, x='x', y='y', bw_adjust=0.7, linewidths=1, ax=ax_joint)
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.5)

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
    name = "static_training_all"
    plt.savefig("%s/%s.png"%(outdir, name), format='png')



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
        x_coord.extend(file_labels["x_coord_list"])
        y_coord.extend(file_labels["y_coord_list"])

        x_coord.extend(file_labels["static_x"])
        y_coord.extend(file_labels["static_y"])

    # heatmap = sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", shade=True, bw_adjust=.5)
    # filename = "validation_distr"
    # heatmap.figure.savefig("/home/christinaz/traj_0_distribution/%s.png"%filename, format='png')
    # heatmap.figure.clf()

    fig = plt.figure(figsize=(8, 8))
    grid_ratio = 5
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])

    # sns.kdeplot(data=df, x='x', y='y', bw_adjust=0.7, linewidths=1, ax=ax_joint)
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.5)

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
    name = "static_validation_all"
    plt.savefig("%s/%s.png"%(outdir, name), format='png')

    return

def testing_set(testing_files):
    x_coord = []
    y_coord = []
    for file in testing_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
        x_coord.extend(file_labels["x_coord_list"])
        y_coord.extend(file_labels["y_coord_list"])

        x_coord.extend(file_labels["static_x"])
        y_coord.extend(file_labels["static_y"])
    # heatmap = sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", shade=True, bw_adjust=.5)
    # filename = "testing_files"
    # heatmap.figure.savefig("/home/christinaz/traj_0_distribution/%s.png"%filename, format='png')
    # heatmap.figure.clf()

    fig = plt.figure(figsize=(8, 8))
    grid_ratio = 5
    gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])

    # sns.kdeplot(data=df, x='x', y='y', bw_adjust=0.7, linewidths=1, ax=ax_joint)
    sns.kdeplot(x=x_coord, y=y_coord, cmap="Reds", fill=True, bw_adjust=0.5)

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
    name = "static_testing"
    plt.savefig("%s/%s.png"%(outdir, name), format='png')

    return


def opt_set(file_to_cost):
    # print(file_to_cost)
    file_to_cost = sorted(file_to_cost.items(), key=lambda x:x[1])
    # print(dict(file_to_cost).values())
    opt_set = list(dict(file_to_cost).keys())[:user_num_frames]



def labeling(filename, traj_num, frame_num, class_weather_count):
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


if __name__ == '__main__':
    main()


