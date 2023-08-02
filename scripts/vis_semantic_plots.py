import os
import itertools
from os.path import join
import seaborn as sns
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import math
import time
import json
import copy
import cv2

sys.path.append(os.getcwd())

from helpers.metadata import OBJECT_DETECTION_TASK, SEMANTIC_SEGMENTATION_TASK
from helpers.constants import *
from helpers.sensors import read_sem_label
from helpers.plotting_utils import sum_labels, kdeplot_set
import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot_type', default="semantichist",
                    help="Select a histogram type to show: semantichit,  ")

#Done with help of Chaitanya

#Get all file paths to iterate through
def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

#Update the dictionary of labels every time a label is found.
def count_labels(labels_list, labels_dictionary, annotated_data):
    classes, counts = np.unique(annotated_data, return_counts=True)

    classes_to_update = labels_list[classes]
    for cls_index, cls_name in enumerate(classes_to_update):
        labels_dictionary[cls_name] += counts[cls_index]
    return labels_dictionary

def plot_counts(indir, outdir):
    semantic_dir = join(indir, SEMANTIC_LABEL_DIR, "os1")
    files = (absoluteFilePaths(semantic_dir))
    labels_list = np.array(list(SEM_CLASS_TO_ID.keys()))

    cached_path = join(outdir, "GEN_semlabeldict.json")
    if not os.path.exists(cached_path):
        print("Cached label dictionary counts not found %s, generating..." % cached_path)
        labels_dictionary = {name: 0 for name in SEM_CLASS_TO_ID.keys() }

        start_time = time.time()
        for file in files:
            annotated_data = read_sem_label(file)
            labels_dictionary = count_labels(labels_list, labels_dictionary, annotated_data)
        end_time = time.time()
        print("Time to count labels ", end_time - start_time)

        with open(cached_path, "w") as label_dict_file:
            labels_dictionary_str = {k: str(v) for k, v in labels_dictionary.items()}
            json.dump(labels_dictionary_str, label_dict_file)
    else: 
        label_dict_file = open(cached_path, "r")
        labels_dictionary_str = json.load(label_dict_file)
        labels_dictionary = {k: np.int64(v) for k, v in labels_dictionary_str.items()}

    #Calculate percentages
    labels_counts_dictionary = copy.deepcopy(labels_dictionary)
    total = sum(labels_dictionary.values())
    proportion = total - labels_dictionary["Unlabeled"]
    for key in labels_dictionary:
        key_sum = labels_dictionary[key]
        labels_dictionary[key] = key_sum/proportion

    for key in labels_dictionary:
        labels_dictionary[key] *=100 # conert to percentage

    #Sort dictionary in descending order
    labels_dictionary_keys = np.array(list(labels_dictionary.keys()))
    labels_dictionary_values = np.array(list(labels_dictionary.values()))
    sorted_labels_indices = np.argsort(-labels_dictionary_values)

    labels_counts_dictionary_keys = np.array(list(labels_counts_dictionary.keys()))
    labels_counts_dictionary_values = np.array(list(labels_counts_dictionary.values()))

    labels_dictionary = { labels_dictionary_keys[index]: labels_dictionary_values[index] for index in sorted_labels_indices }
    labels_counts_dictionary = {labels_counts_dictionary_keys[index]: labels_counts_dictionary_values[index] for index in sorted_labels_indices}
    colors = np.array([ '#{:02x}{:02x}{:02x}'. format(SEM_ID_TO_COLOR[idx][2], SEM_ID_TO_COLOR[idx][1], SEM_ID_TO_COLOR[idx][0]) for idx in sorted_labels_indices if idx!=0])
    
    #  np.array([ '#{:02x}{:02x}{:02x}'. format(c[2], c[1], c[0]) for i, c in enumerate(SEM_ID_TO_COLOR) if i!=0])
    # sorted_colors_descending = colors[sorted_labels_indices]

    # sorted_labels_descending = 
    # sorted_labels_counts_descending = sorted(labels_counts_dictionary.items(), key=lambda x:x[1], reverse=True)
    #Clean and get labels and counts 
    # labels_dictionary = dict(sorted_labels_descending)
    # labels_counts_dictionary = dict(sorted_labels_counts_descending)
    del labels_dictionary["Unlabeled"]
    del labels_counts_dictionary["Unlabeled"]
    keys = list(labels_dictionary.keys())
    values = [labels_dictionary[k] for k in keys]

    #Plot data in bargraph (Using Seaborn)
    #Run export DISPLAY=:0.0 before plotting.

    sns.set(rc={'figure.figsize':(25,25)})

    # colors = ["firebrick", "gold", "orange", "purple", "hotpink", "palegreen", "darkcyan", "darkblue", "khaki", "lightcoral", "lawngreen", "teal", "sienna", "plum", "slateblue", "darkorchid", "slategray", "aqua", "magenta", "thistle", "peachpuff", "navy", "skyblue", "mediumvioletred"]
    sns.set_style("darkgrid")
    # sns.set_palette(colors)

    class_to_type_dict ={
        'Road Pavement':    'Road', 
        'Concrete':         'Walkways', 
        'Speedway Bricks':  'Walkways', 
        'Dirt Paths':       'Unstructured', 
        'Short Vegetation': 'Unstructured', 
        'Carpet':           'Indoor Floor', 
        'Porcelain Tile':   'Tiling', 
        'Pebble Pavement':  'Walkways', 
        'Patterned Tile':   'Tiling', 
        'Dark Marble Tiling':   'Tiling', 
        'Grass':            'Unstructured', 
        'Red Bricks':       'Walkways', 
        'Crosswalk':        'Road', 
        'Rocks':            'Unstructured',  
        'Stairs':           'Boundary', 
        'Wood Panel':       'Walkways', 
        'Light Marble Tiling':  'Tiling', 
        'Unknown':          'Other',
        'Metal Grates':     'Metal', 
        'Door Mat':         'Boundary', 
        'Dome Mat':         'Boundary',
        'Blond Marble Tiling':  'Tiling',
        'Metal Floor':      'Metal',
        'Threshold':        'Boundary'
    }
    floor_type_list = []
    for floor in keys:
        floor_type_list.append(class_to_type_dict[floor])

    floor_counts_list = []
    for floor in keys:
        floor_counts_list.append(labels_counts_dictionary[floor])

    #Create dataframe with each label names, group, and counts.
    df = pd.DataFrame({'Labels': keys,
        'Type': floor_type_list,
        'Proportion': values,
        'Counts': floor_counts_list,
        'Colors': colors})
    df = df.sort_values(by=['Type'], ascending=False, kind='mergesort').reset_index()
    #Plot and set fields.
    print("Plotting")
    cats = ['Outdoor', 'Indoor', 'Both']
    # sns.set_theme(style='white')
    sns.set(style='white', font_scale=4)

    # Set the figure size
    # ax = sns.barplot(x='Type', y='Proportion', hue='Labels', data=df, palette=colors, width=1)
    # sns.set_palette(df['Colors'])
    ax = sns.barplot(x='Labels', y='Proportion', data=df, palette=df['Colors'], width=1)

    # # Create legend handles
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(df['Colors'], df['Labels'])]

    # Remove x-ticks
    plt.xticks([])
    # plt.legend(labels=df['Labels'])
    # plt.legend(handles=legend_handles,  ncol=2, fontsize=32)
    plt.legend(handles=legend_handles, title='Terrain Class', bbox_to_anchor=(1.145, 1.17), loc='upper right', ncol=1, fontsize=32)
    # ax.legend(loc='upper right', fontsize=36, ncol=2)
    ax.set(xlabel=None)
    ax.set(ylabel="% Total Annotations")

    for idx, i in enumerate(ax.patches):
        p = i.get_height()
        p_text = f'{p:.3f}'
        x = i.get_x() + i.get_width()/2
        y = i.get_y() + p + 0.001

        plt.text(x, y, df['Counts'][idx], ha='center', va='bottom', size=40, rotation=90)
        print("counts ", df['Counts'][idx], " idx ", idx)

    plt.grid(color='gray', linestyle='solid', axis='both')
    print(labels_dictionary)
    sns.despine()
    #Save locally
    plt.savefig(os.path.join(outdir,"GEN_semlabelhist.png"), format='png', dpi=300)

def plot_object_heatmap(indir, outdir):
    metadata_dir = join(indir, METADATA_DIR)
    list_metadata_files = next(os.walk(metadata_dir))[2]
    all_files = []
    traj_to_frame = {}

    training_files = []
    validation_files = []
    testing_files = []
    counter = 0
    print(list_metadata_files)
    num_weather = 4
    class_weather_count = {objcls: [0]*num_weather for objcls in BBOX_CLASS_TO_ID.keys()}

    input_dict = {
        "distance_total": [0, 0, 0], 
        "all_labels": {}, 
        "list_distance_vec": [],
        "list_theta_vec": [],
        "list_weather_vec": [],
        "theta_total": [0, 0, 0, 0],
        "class_total": [0] * len(BBOX_CLASS_TO_ID),
        "list_class_vec": [] 
    }

    for metadata_file in list_metadata_files:
        metadata_path = join(metadata_dir, metadata_file)
        print(metadata_path)
        counter += 1
        # print(metadata_file)
        # if (metadata_file == "14.json"):
        f = open(metadata_path, 'r')
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
        input_dict = sum_labels(indir, input_dict, trajectory_files, class_weather_count)

    print("-----------------")
    # print(class_weather_count)
    for object_type in ["static", "dynamic", "combined"]:
        kdeplot_set(training_files, outdir, input_dict['all_labels'], split="training", object_type=object_type)
        kdeplot_set(validation_files, outdir, split="validation", object_type=object_type)
        kdeplot_set(testing_files, outdir, split="testing", object_type=object_type)
    class_and_weather()
    combine_images()

def plot_weather_time_frequency(indir, outdir):
    metadata_dir = join(indir, METADATA_DIR)
    metadata_files = file_paths = [file_name for file_name in os.listdir(metadata_dir)
              if os.path.isfile(os.path.join(metadata_dir, file_name))]

    for metadata_file in metadata_files:
        pass # TODO later

def plot_label_counts(indir, outdir, splits=["training", "validation", "testing"], 
    counts_file="GEN_object_counts.json"):
    metadata_dir = join(indir, METADATA_DIR)
    metadata_files = file_paths = [file_name for file_name in os.listdir(metadata_dir)
              if os.path.isfile(os.path.join(metadata_dir, file_name))]

    
    counts_cache = join(outdir, counts_file)
    if not os.path.exists(counts_cache):
    
        counts_dict = {className: 0 for className in BBOX_CLASS_TO_ID.keys()}
        for metadata_file in metadata_files:
            meta_path = join(metadata_dir, metadata_file)
            obj_splits = json.load(open(meta_path, 'r'))['ObjectTracking']

            # Load all annotation paths and count labels
            obj_anno_subpaths = []
            for split in splits:
                obj_anno_subpaths.extend(obj_splits[split])

            print("Started meta file %s" % meta_path)
            for obj_anno_subpath in obj_anno_subpaths:
                obj_anno_path = join(indir, obj_anno_subpath)
                obj_anno_dict = json.load(open(obj_anno_path, 'r'))["3dbbox"]

                for obj_dict in obj_anno_dict:
                    counts_dict[obj_dict["classId"]] += 1
            print("Finished meta file %s" % meta_path)

        # Save cached label counts
        with open(counts_cache, "w") as counts_cache_file:
            counts_dict_str = {k: str(v) for k, v in counts_dict.items()}
            json.dump(counts_dict_str, counts_cache_file)
    else:
        counts_dict = json.load(open(counts_cache, 'r'))

    # Divide into categories
    class_to_cat = {
        "Tree"                  : "Vegetation",
        "Freestanding Plant"    : "Vegetation",
        "Pole"                  : "Structure",
        "Traffic Sign"          : "Structure",
        "Bike Rack"             : "Structure",
        "Door"                  : "Structure",
        "Floor Sign"            : "Structure",
        "Canopy"                : "Structure",
        "Informational Sign"    : "Structure",
        "Door Switch"           : "Structure",
        "Room Label"            : "Structure",
        "Wall Sign"             : "Structure",
        "Traffic Light"         : "Structure",
        "Horse"                 : "Structure",
        "Railing"               : "Barrier",
        "Bollard"               : "Barrier",
        "Traffic Arm"           : "Barrier",
        "Fence"                 : "Barrier",
        "Cone"                  : "Barrier",
        "Construction Barrier"  : "Barrier",
        "Stanchion"             : "Barrier",
        "Trash Can"             : "Container",
        "Dumpster"              : "Container",
        "Cart"                  : "Container",
        "Newspaper Dispenser"   : "Service Machine",
        "Sanitizer Dispenser"   : "Service Machine",
        "Parking Kiosk"         : "Service Machine",
        "ATM"                   : "Service Machine",
        "Mailbox"               : "Service Machine",
        "Condiment Dispenser"   : "Service Machine",
        "Vending Machine"       : "Service Machine",
        "Water Fountain"        : "Service Machine",
        "Bike"                  : "Transportation",
        "Car"                   : "Transportation",
        "Scooter"               : "Transportation",
        "Service Vehicle"       : "Transportation",
        "Utility Vehicle"       : "Transportation",
        "Delivery Truck"        : "Transportation",
        "Pickup Truck"          : "Transportation",
        "Motorcycle"            : "Transportation",
        "Golf Cart"             : "Transportation",
        "Truck"                 : "Transportation",
        "Bus"                   : "Transportation",
        "Segway"                : "Transportation",
        "Skateboard"            : "Transportation",
        "Emergency Phone"       : "Emergency Device",
        "Fire Hydrant"          : "Emergency Device",
        "Fire Extinguisher"     : "Emergency Device",
        "Fire Alarm"            : "Emergency Device",
        "Emergency Aid Kit"     : "Emergency Device",
        "Pedestrian"            : "Human",
        "Chair"                 : "Furniture/ Appliance",
        "Table"                 : "Furniture/ Appliance",
        "Bench"                 : "Furniture/ Appliance",
        "Couch"                 : "Furniture/ Appliance",
        "Computer"              : "Furniture/ Appliance",
        "Television"            : "Furniture/ Appliance",
        "Vacuum Cleaner"        : "Furniture/ Appliance",
        "Other"                 : "Other"
    }

    # Sort class counts dictionary
    sorted_counts_dict = dict(sorted(counts_dict.items(), key=lambda item: int(item[1]), reverse=True ))

    # Remove classes with no counts
    sorted_counts_dict = { key: int(val) for key, val in sorted_counts_dict.items() if int(val)>0}

    # Create new dictionary from sorted class counts dictionary into multiple smaller dictionaries
    cat_list = []
    for cat in class_to_cat.values():
        if cat not in cat_list:
            cat_list.append(cat)
    cat_list_dict = {cat: {} for cat in cat_list }

    for cls_name, counts in sorted_counts_dict.items():
        if cls_name in class_to_cat:
            cat = class_to_cat[cls_name]
            cat_list_dict[cat][cls_name] = counts
        else:
            print("Class %s not in labels" % cls_name)

    # Merge separate dictionaries into one
    separated_counts_dict = {}
    colors = []
    for cat in cat_list_dict:
        for cls_name, cls_count in cat_list_dict[cat].items():
            colors.append(BBOX_ID_TO_COLOR[BBOX_CLASS_TO_ID[cls_name]])
            new_cls_name = cls_name.replace(' ', '\n')
            separated_counts_dict[new_cls_name] = int(cls_count)

    colors = [ '#{:02x}{:02x}{:02x}'. format(c[2], c[1], c[0]) for i, c in enumerate(colors)]
    sns.set(style='white', font_scale=2.25)
    sns.set_palette(colors)

    # Plot expects dict in the form of x="cat name" y="cat counts"
    df = pd.DataFrame({
        'Counts': list(separated_counts_dict.values()),
        'Labels': list(separated_counts_dict.keys())
    })

    ax = sns.catplot(x='Labels', y='Counts', data=df, 
        kind='bar',
        height=10,
        aspect=3.5,
        width=0.75,
        legend=True,
        palette=colors
    )

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, df['Labels'])]
    name = "GEN_object_counts_plot"
    
    # plt.legend(handles=legend_handles, title='Semantic Class', ncol=7, fontsize=20)
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, 2.2), loc='upper center', ncol=11, fontsize=22)
    plt.yscale('log')
    ax.set(xlabel="")
    ax.set(ylabel="# Labels (Log Scale)")
    ax.set_xticklabels([])
    plt.grid(color='gray', linestyle='solid')
    plt.subplots_adjust(top=0.55)
    # plt.subplots_adjust(top=0.1)
    plt.savefig("%s/%s.png"%(outdir, name), format='png', dpi=300)
        
def plot_label_weather(indir, outdir, splits=["training", "validation", "testing"], 
    counts_file="GEN_object_weather.json"):
    metadata_dir = join(indir, METADATA_DIR)
    metadata_files = file_paths = [file_name for file_name in os.listdir(metadata_dir)
              if os.path.isfile(os.path.join(metadata_dir, file_name))]
    traj_weather_path = join(os.getcwd(), "helpers", "helper_utils", "weather_data_multi.json")
    traj_weather_dict = json.load(open(traj_weather_path, 'r'))
    weather_list = []
    for traj_weather_list in traj_weather_dict.values():
        for weather in traj_weather_list:
            if weather not in weather_list:
                weather_list.append(weather)

    counts_cache = join(outdir, counts_file)
    if not os.path.exists(counts_cache):
    
        counts_dict = {className: {weather: 0 for weather in weather_list} for className in BBOX_CLASS_TO_ID.keys()}
        for metadata_file in metadata_files:
            traj = metadata_file.split('.')[0]
            meta_path = join(metadata_dir, metadata_file)
            obj_splits = json.load(open(meta_path, 'r'))['ObjectTracking']

            print("Started meta file %s" % meta_path)
            for weather in traj_weather_dict[traj]:
                # Load all annotation paths and count labels
                obj_anno_subpaths = []
                for split in splits:
                    obj_anno_subpaths.extend(obj_splits[split])

                for obj_anno_subpath in obj_anno_subpaths:
                    obj_anno_path = join(indir, obj_anno_subpath)
                    obj_anno_dict = json.load(open(obj_anno_path, 'r'))["3dbbox"]

                    # Count occurrences of each object under each weather condition
                    for obj_dict in obj_anno_dict:
                        counts_dict[obj_dict["classId"]][weather] += 1
            print("Finished meta file %s" % meta_path)

        # Save cached label counts
        with open(counts_cache, "w") as counts_cache_file:
            counts_dict_str = {k: str(v) for k, v in counts_dict.items()}
            json.dump(counts_dict_str, counts_cache_file)

    counts_dict = json.load(open(counts_cache, 'r'))

    def str_to_dict(string):
        # remove the curly braces from the string
        string = string.strip('{}')
    
        # split the string into key-value pairs
        pairs = string.split(', ')
    
        # use a dictionary comprehension to create
        # the dictionary, converting the values to
        # integers and removing the quotes from the keys
        return {key[1:-1]: int(value) for key, value in (pair.split(': ') for pair in pairs)}

    def dict_sort_helper(weather_counts_dict):
        sum = 0
        for count in str_to_dict(weather_counts_dict[1]).values():
            sum += int(count)
        return sum


    # Sort object classes by summing all counts across all weather conditions
    sorted_counts_dict_str = dict(sorted(counts_dict.items(), key=dict_sort_helper, reverse=True ))
    sorted_counts_dict = {}
    for clsname, val in sorted_counts_dict_str.items():
        new_cls_name = clsname.replace(' ', '\n', 1)
        weather_counts_dict = str_to_dict(val)
        
        any_counts_nonzero = False
        for _, weather_counts_val in weather_counts_dict.items():
            if weather_counts_val > 0:
                any_counts_nonzero = True
        
        if not any_counts_nonzero: # Dont append objects with zero annotation instances here
            continue
        sorted_counts_dict[new_cls_name] = weather_counts_dict

    num_plots = 2
    num_classes = len(sorted_counts_dict.keys())
    classes_per_plot = num_classes // num_plots

    start_class_index = 0
    end_class_index = classes_per_plot
    # Plot grouped histogram in seaborn
    sns.set_style('whitegrid')
    plot_idx = 0
    plot_paths = []

    class_abbr_map = {
        "Freestanding Plant": "Plant",
        "Newspaper Dispenser": "News. Disp.",
        "Informational Sign": "Info. Sign",
        "Emergency Phone": "Emer. Phone",
        "Sanitizer Dispenser": "Sani. Disp",
        "Fire Extinguisher": "Fire Extin.",
        "Emergency Aid Kit": "Emer. Aid Kit"
    }
    while start_class_index < end_class_index:
        data_list = []
        colors = []

        class_idx = 0
        for cls_name, weather_dict in sorted_counts_dict.items():
            short_cls_name = cls_name
            if cls_name.replace('\n', ' ', 1) in class_abbr_map.keys():
                short_cls_name = class_abbr_map[cls_name.replace('\n', ' ', 1)]
                short_cls_name = short_cls_name.replace(' ', '\n', 1)
            
            if class_idx >= start_class_index and class_idx < end_class_index:
                weather_counts_list = list(weather_dict.values())
                df = pd.DataFrame(
                    {
                    'Counts': list(weather_dict.values()),
                    'Weather': list(weather_dict.keys())
                    }
                ).assign(Location=short_cls_name)
                single_line_cls_name = cls_name.replace('\n', ' ', 1)
                colors.append(BBOX_ID_TO_COLOR[BBOX_CLASS_TO_ID[single_line_cls_name]])
                data_list.append(df)
            class_idx += 1
        if len(data_list)==0:
            print("Finished generating subplots...")
            break

        cdf = pd.concat(data_list)
        # mdf = pd.melt(cdf, id_vars=['Location'], var_name=['ClassName'])
        mdf = cdf
        colors = [ '#{:02x}{:02x}{:02x}'. format(c[2], c[1], c[0]) for i, c in enumerate(colors)]

        plt.figure(figsize=(19, 5))  # Adjust the width and height as needed
        plt.subplots_adjust(bottom=0.30, top=0.97)  # Adjust the margin bottom as needed
        sns.set(style='white', font_scale=1.5)
        ax = sns.barplot(x="Location", y="Counts", hue="Weather", data=mdf, 
            errwidth=0
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        num_locations = len(mdf.Location.unique())

        hatches = itertools.cycle(['*', '//', 'o', '-'])
        for i, bar in enumerate(ax.patches):
            if i % num_locations == 0:
                hatch = next(hatches)
            bar.set_hatch(hatch)

        # Iterate over the bars and modify the colors and line styles
        for i, patch in enumerate(ax.patches):
            # Calculate the index within each group
            index_within_group = i % len(colors)
            # Set the same color for each group
            patch.set_facecolor(colors[index_within_group])    

        plt.yscale('log')
        ax.set(xlabel="")
        ax.set(ylabel="# Labels (Log Scale)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=4, fancybox=True, shadow=False)
        plt.grid(color='gray', linestyle='solid')

        name="GEN_object_weather_plot%i" % plot_idx
        plot_path = "%s/%s.png"%(outdir, name)
        plot_paths.append(plot_path)
        plt.savefig(plot_path, format='png', dpi=300)
        plt.clf()

        plot_idx += 1
        start_class_index = end_class_index
        end_class_index = max(end_class_index + classes_per_plot, end_class_index)

    # Combine subsplots to vertically
    img_list = [] 
    for plot_path in plot_paths:
        img_np = cv2.imread(plot_path)
        img_list.append(img_np)

    combined_img_np = np.vstack((img_np for img_np in img_list))
    combined_img_path = join(outdir, "GEN_object_weather_plot_combined.png")
    cv2.imwrite(combined_img_path, combined_img_np)

def check_num_annotations(indir, task=SEMANTIC_SEGMENTATION_TASK):
    meta_dir = join(indir, METADATA_DIR)
    meta_files = [meta_file for meta_file in os.listdir(meta_dir) if meta_file.endswith(".json")]

    annotation_files = {}
    for meta_file in meta_files:
        meta_path = join(meta_dir, meta_file)
        obj_splits = json.load(open(meta_path, 'r'))[task]
        for split in obj_splits.keys():
            if split not in annotation_files:
                annotation_files[split] = obj_splits[split]
            else:
                annotation_files[split].extend(obj_splits[split])

    num_frames = 0
    for split, split_list in annotation_files.items():
        print("Split %s number of frames %i" % (split, len(split_list)))
        num_frames += len(split_list)
    print("Total number of frames across all splits %i"%num_frames)
    return annotation_files

def convert_ts_to_military(ts_np):
    # Converts unix epoch timestamp to 24 hour military time
    ts_local = [datetime.datetime.fromtimestamp(ts) for ts in ts_np]
    # Format as 24-hour time strings
    time_ints = [int(dt.strftime("%H%M%S")) for dt in ts_local]
    return time_ints

def plot_label_location(indir, outdir, counts_file="GEN_location_counts.json", do_time=False):
    meta_dir = join(indir, METADATA_DIR)
    meta_files = [meta_file for meta_file in os.listdir(meta_dir) if meta_file.endswith(".json")]

    timestamp_dir       = join(indir, TIMESTAMPS_DIR)
    traj_weather_path = join(os.getcwd(), "helpers", "helper_utils", "weather_data_multi.json")
    traj_weather_dict = json.load(open(traj_weather_path, 'r'))
    weather_list = [] # Set of available weather conditions
    for weather in traj_weather_dict.values():
        if type(weather) == list:
            for subweather in weather:
                if subweather not in weather_list:
                    weather_list.append(subweather)
        elif weather not in weather_list:
            weather_list.append(weather)

    location_frame_path = join(os.getcwd(), "helpers", "helper_utils", "locations.json")
    location_frame_dict = json.load(open(location_frame_path, 'r'))

    locations_densities_dict = {}
    annotation_files = {}

    # Load location counts if cached
    cache_path = join(outdir, counts_file)
    if not os.path.exists(cache_path):
        # Loop through all available trajectories
        for traj in range(23):
            traj = str(traj)
            timestamp_path = join(timestamp_dir, "%s_frame_to_ts.txt"%traj)
            timestamp_np = np.loadtxt(timestamp_path).reshape(-1,)
            time_ints = convert_ts_to_military(timestamp_np)
            print("Start processing traj %s"%traj)
            start_time = time.time()
            for location_name, frames in location_frame_dict[traj].items():
                if location_name not in locations_densities_dict:
                    locations_densities_dict[location_name] = {weather: [] for weather in weather_list}

                for frame_set in frames:
                    start_frame, end_frame = frame_set[0], frame_set[1]
                    end_frame = len(time_ints) if end_frame == -1 else end_frame+1
                    for weather in traj_weather_dict[traj]:
                        locations_densities_dict[location_name][weather].extend(time_ints[start_frame:end_frame])
            print("Done processing took", time.time() - start_time)

        
        with open(cache_path, "w") as counts_cache_file:
            json.dump(locations_densities_dict, counts_cache_file)
    else:
        locations_densities_dict = json.load(open(cache_path, 'r'))

    # Plotting
    locations = list(locations_densities_dict.keys())

    # Create a grid of subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)

    color_palette = ['#FD8F0F', '#0157E9', '#88E910', '#E11846', '#CC8899'] # currently only support four weather types
    weather_color_palette = {
        weather: color_palette[weather_idx] for weather_idx, weather in enumerate(weather_list)
    }

    if not do_time:         
        # fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
        fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharex=True)
        weather_label_count_dict = { weather: [0]*3 for weather in weather_list }
        weather_label_count_dict['Time of Day'] = ['Morning', 'Afternoon', 'Evening']
        discrete_location_densities_dict = {location: copy.deepcopy(weather_label_count_dict) for location in locations}

        morning_thres, afternoon_thres = 123000, 170000, 
        # Process weather into morning, afternoon evening categories
        for location in locations_densities_dict.keys():
            for weather, times in locations_densities_dict[location].items():
                times_np = np.array(times)

                counts_np = [0]*3
                counts_np[0] = np.sum(times_np<morning_thres)
                counts_np[1] = np.sum(np.logical_and(times_np>=morning_thres, times_np<afternoon_thres))
                counts_np[2] = np.sum(times_np>=afternoon_thres)

                for time_idx in range(len(counts_np)):
                    discrete_location_densities_dict[location][weather][time_idx] += counts_np[time_idx]
    
        sns.set(font_scale=1.3)
        # sns.set(font_scale=1.2)
        decimal_places = 0
        for i, ax in enumerate(axes):
            location = locations[i]
            print("Plotting location %s"%location)

            df = pd.DataFrame(discrete_location_densities_dict[location])

            # Set seaborn style
            sns.set(style="whitegrid")

            # Melt the DataFrame to transform it into long format for stacked bar plot
            df_melted = df.melt(id_vars='Time of Day', var_name='Weather Condition', value_name='Count')

            # Plot the stacked barplot
            sns.barplot(data=df_melted, x='Time of Day', y='Count', hue='Weather Condition', ax=ax, palette="pastel", dodge=False)

            axes[i].set_title(location, fontsize=24)
            ax.set_ylabel('   ', fontsize=18)
            ax.set_xlabel('   ', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=20)
            axes[i].get_legend().remove()

            axes[i].set_axisbelow(True)
            axes[i].grid(color='gray', linestyle='solid')
            # axes[i].grid(True, zorder=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=24)
        # Add labels and title
        # ax.set_xlabel('Time of Day', fontsize=20)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.ylabel('# Frames')
        # plt.title('Weather Conditions by Time of Day')

        # fig.text(0.02, 0.5, '# Frames', va='center', rotation='vertical', fontsize=20)
        fig.text(0.5, 0.03, 'Time of Day', va='center', rotation='horizontal', fontsize=30)
        fig.text(0.001, 0.5, '# Frames', va='center', rotation='vertical', fontsize=30)
        # Adjust spacing between subplots
        plt.tight_layout()

        plt.savefig(join(outdir, 'GEN_location_counts.png'), format='png', dpi=300)

    else: # Plot KDEs with time based x axis
        sns.set(font_scale=1.2)
        decimal_places = 0
        for i, ax in enumerate(axes):
            location = locations[i]
            print("Plotting location %s"%location)
            # for weather, times in locations_densities_dict[location].items():
                # if len(times) == 0:
                #     continue

            all_times = []
            all_weather = []
            for weather, times in locations_densities_dict[location].items():
                ds_times = times

                all_times.extend([time/10000 for time in ds_times])
                all_weather.extend([weather]*len(ds_times))
            data = pd.DataFrame({
                'Time': np.array(all_times),
                'Weather': np.array(all_weather, dtype=str)
            })
            custom_palette = [weather_color_palette[weather] for weather in data['Weather'].unique()]
            # Use low bandwidth to show all spots
            # sns.kdeplot(data=data, x='Time', hue='Weather', fill=True, ax=ax, bw_method=0.2, palette=custom_palette, legend=True)
            sns.histplot(data=data, x='Time', hue='Weather', fill=True, ax=ax, palette=custom_palette, legend=True, multiple='stack')
                # Normalize times to be in hours insetad of seconds
                # times = [t/10000 for t in times]
                # sns.kdeplot(data=times, fill=True, ax=ax, bw_method=0.5)

            ax.set_title(location, fontsize=18)
            ax.set_ylabel('   ', fontsize=18)
            ax.set_xlabel('Time (HH:MM:SS)', fontsize=20)
            legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(custom_palette, data['Weather'].unique())]
            ax.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=18)
            y_min, y_max = ax.get_ylim()
            ax.set_yticks(np.linspace(start=int(y_min), stop=math.ceil(y_max), num=4).tolist())
            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimal_places}f'))
            ax.tick_params(axis='both', which='major', labelsize=18)
            # legend_handles, legend_labels = ax.get_legend_handles_labels()
            # ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        xtick_times = [str(int(t)).rjust(2, '0').ljust(6, '0') for t in ax.get_xticks()]
        xtick_times = [f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" for time_str in xtick_times]
        # plt.yticks(fontsize=18)
        plt.xticks(ax.get_xticks(), xtick_times, rotation=45)

        # Set common x label for the entire figure
        # fig.text(0.5, 0.01, 'Time (HH:MM:SS)', ha='center')
        fig.text(0.02, 0.5, '# Frames', va='center', rotation='vertical', fontsize=20)
        # Adjust spacing between subplots
        plt.tight_layout()

        plt.savefig(join(outdir, 'GEN_location_counts.png'), format='png', dpi=300)

def main(args):
    #Get file paths and loop throught to sum each label
    indir = "/robodata/arthurz/Datasets/CODa_dev"
    outdir = "%s/plots" % os.getcwd()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    check_num_annotations(indir)

    if args.plot_type=="semantichist":
        #List of labels to use to get label name from index, Dictionary to keep track of total for each label
        plot_counts(indir, outdir)
    elif args.plot_type=="objheatmap":
        # broken fix later
        plot_object_heatmap(indir, outdir)
    elif args.plot_type=="weathertime":
        plot_weather_time_frequency(indir, outdir)
    elif args.plot_type=="labelcounts":
        plot_label_counts(indir, outdir)
    elif args.plot_type=="labelweather":
        plot_label_weather(indir, outdir)
    elif args.plot_type=="labellocation":
        plot_label_location(indir, outdir)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
