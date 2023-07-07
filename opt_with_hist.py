import labeling
import math
import os
import json
import numpy as np
from operator import truediv
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
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

outdir = "/robodata/arthurz/Research/coda_package/src/coda-devkit/plots"
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
        # if (metadata_file != "6.json"):
        if metadata_file!="13.json":
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
    object_type="dynamic"
    file_list = [training_files, validation_files, testing_files]
    object_type_list = ["static", "dynamic", "combined"]
    split_type_list = ["training", "validation", "testing"]
    for object_type in ["static", "dynamic", "combined"]:
        for i in range(len(split_type_list)):
            kdeplot_set(file_list[i], split=split_type_list[i], object_type=object_type)
    combine_kdeplots(object_type_list, split_type_list)

    # Uncomment to generate histograms
    # num_images = class_and_weather()
    # combine_images(num_images)

def combine_kdeplots(object_type_list, split_type_list):
    img_list = []

    combined_img = None
    for row_idx, object_type in enumerate(object_type_list):

        horizontal_img = None
        for col_idx, split_type in enumerate(split_type_list):
            img_np = cv2.imread("%s/%s_%s_all.png"%(outdir, object_type, split_type))
            img_list.append(img_np)

            if horizontal_img is None:
                horizontal_img = img_list[row_idx*col_idx+col_idx]
            else:
                horizontal_img = np.hstack((horizontal_img, img_list[row_idx*col_idx+col_idx]))
        
        if combined_img is None:
            combined_img = horizontal_img
        else:
            combined_img = np.vstack((combined_img, horizontal_img))

    cv2.imwrite("%s/final_kdeplot.png"%outdir, combined_img)
    

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

    # Presort objects by frequency
    combined_list = np.array(sunny_list) + np.array(rainy_list) + np.array(cloudy_list) + np.array(dark_list)
    combined_list_sorted_indices = np.argsort(-combined_list)

    sunny_list = np.array(sunny_list)[combined_list_sorted_indices].tolist()
    rainy_list = np.array(rainy_list)[combined_list_sorted_indices].tolist()
    cloudy_list = np.array(cloudy_list)[combined_list_sorted_indices].tolist()
    dark_list = np.array(dark_list)[combined_list_sorted_indices].tolist()
    obj_name = np.array(obj_name)[combined_list_sorted_indices].tolist()

    total_num_classes = len(obj_name)
    num_images = 2
    num_objects_per_plot = total_num_classes // num_images
    first = 0
    last = num_objects_per_plot
    counter = 1
    for num in range(2):
        fig = plt.figure()
        # for x_index in range(3):
            # if (last < 54):
        labels = np.array([['sunny', 'rainy'],['cloudy', 'dark']])
        labels = np.repeat(labels, (last-first))

        obj_name_two_line = [name.replace(' ', '\n', 1) for name in obj_name[first:last]]
        full_obj_list = np.tile(obj_name_two_line, 4)

        df = pd.DataFrame({
            'Type': full_obj_list,
            'Proportion': sunny_list[first:last]+rainy_list[first:last]+cloudy_list[first:last]+dark_list[first:last],
            'Labels': labels.tolist()
        })
        
        # df = pd.DataFrame({
        #     'Labels': ['sunny', 'rainy', 'cloudy', 'dark']
        #     'object_name': obj_name[first:last], 
        #     'sunny': sunny_list[first:last], 
        #     'rainy': rainy_list[first:last], 
        #     'cloudy' : cloudy_list[first:last], 
        #     'dark' : dark_list[first:last]})
        # plot = df.plot(x="object_name", y=['sunny', 'rainy', 'cloudy', 'dark'], kind="bar")
        
        sns.set(style='white', font_scale=2.5)
        print(num==1)
        ax = sns.catplot(x='Type', y='Proportion', hue='Labels', data=df, 
            kind='bar',
            height=10,
            aspect=3.5,
            width=0.75,
            legend_out=False,
            legend=num==1
        )
        ax.set(yscale="log")
        ax.set(xlabel=None)
        ax.set(ylabel="Counts")
        plt.legend(loc='upper right')
        ax.set_xticklabels(rotation=90)

        name = "combined" + str(counter)
        plt.subplots_adjust(bottom=0.3)
        plt.savefig("%s/%s.png"%(outdir, name), format='png', dpi=300)
        first = last
        if (last + num_objects_per_plot) > 53:
            last = 53
        else:
            last += num_objects_per_plot

        # # axs[0].bar(obj_name[0:10], sunny_list[0:10], label="sunny")
        # # axs[0].bar(obj_name[0:10], rainy_list[0:10], label="rainy")
        # # axs[0].bar(obj_name[0:10], cloudy_list[0:10], label="cloudy")
        # # axs[0].bar(obj_name[0:10], dark_list[0:10], label="dark")
        # # axs[0].set_title('Chart 1')
        # # axs[0].legend()
        

        # # fig.set_figheight(5)
        # # print(obj_name[:9])
        # fig = plot.get_figure()
        # name = "combined" + str(counter)
        # fig.savefig("%s/%s.png"%(outdir, name), format='png', bbox_inches='tight')
        counter += 1
        #53   53-6 = 47 
        #Pedestrian, Chair, Table, Railing, Pole, Tree 
    
    return num_images

def combine_images(num_images, num_rows=2):
    # os.chdir("/home/christinaz/paper_barcharts/")
    img_list = []
    total_width, total_height = 0, 0
    for i in range(1, num_images+1):
        img_np = cv2.imread("%s/combined%i.png"%(outdir, i))
        total_width += img_np.shape[0]
        total_height += img_np.shape[1]

        img_list.append(img_np)
    # img_list = [img1, img2, img3, img4, img5, img6, img7, img8]
    # import pdb; pdb.set_trace()
    # new_list = []
    # for img in img_list:
    #     img = cv2.resize(img, (800, 300))
    #     new_list.append(img)
    

    # img12 = np.hstack((new_list[0], new_list[1], new_list[2], new_list[3]))
    # img34 = np.hstack((new_list[4], new_list[5], new_list[6], new_list[7]))
    result = np.vstack((img_list[0], img_list[1]))
    cv2.imwrite("%s/final_image.png"%outdir, result)
    return

def kdeplot_set(training_files, split="training", object_type="dynamic"):
    x_coord = []
    y_coord = []
    for file in training_files:
        file = file.split("/")[3]
        file_labels = all_labels[file]
    
        if object_type=="combined":
            object_type_list = ["static", "dynamic"]
        else:
            object_type_list = [object_type]

        for object_type_single in object_type_list:
            x_coord.extend(file_labels["%s_x"%object_type_single])
            y_coord.extend(file_labels["%s_y"%object_type_single])
    
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
    font_size = 30
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)

    name = "%s_%s_all" % (object_type, split)
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
        print(traj)
        distr[weather_data[str(traj)]] += traj_to_frame[traj]
    print(list(distr.keys()))
    print(list(distr.values()))
    plt.bar(list(distr.keys()), list(distr.values()))
    plt.savefig("%s/weather.png"%outdir, format='png')

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
        # print(filename)
        traj_num = filename.split("/")[2]
        filename = filename.split("/")[3]
        frame_num = (filename.split("_")[4]).split(".")[0]
        # print(frame_num)
        
        os.chdir("/robodata/arthurz/Datasets/CODa/3d_bbox/os1/%s/" % traj_num)
        labels = labeling.labeling_with_hist(filename, traj_num, frame_num, class_weather_count)
        all_labels.update(labels)
        labels = labels[filename]

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

        print("selected_files distribution-------")
        print("weather distribution" + str(weather_count))
        print("distance distribution" + str(distance_count)  )
        print("theta distribution" + str(theta_count)    )
        print("class distribution" + str(class_count) )
        print("-------")
        print(" ")
        # print("total count---------")
        # print("weather total count" + str(weather_total))
        # print("distance total count" + str(distance_total))
        # print("theta total count" + str(theta_total))
        # print("class total count" + str(class_total))
        # print("-------")


if __name__ == '__main__':
    main()
