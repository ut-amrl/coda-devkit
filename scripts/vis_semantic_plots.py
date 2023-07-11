import os
from os.path import join
import seaborn as sns
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

import time
import json
import copy

sys.path.append(os.getcwd())

from helpers.constants import *
from helpers.sensors import read_sem_label
from helpers.plotting_utils import sum_labels, kdeplot_set

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
    sorted_labels_descending = sorted(labels_dictionary.items(), key=lambda x:x[1], reverse=True)
    sorted_labels_counts_descending = sorted(labels_counts_dictionary.items(), key=lambda x:x[1], reverse=True)
    #Clean and get labels and counts 
    labels_dictionary = dict(sorted_labels_descending)
    labels_counts_dictionary = dict(sorted_labels_counts_descending)
    del labels_dictionary["Unlabeled"]
    del labels_counts_dictionary["Unlabeled"]
    keys = list(labels_dictionary.keys())
    values = [labels_dictionary[k] for k in keys]

    #Plot data in bargraph (Using Seaborn)
    #Run export DISPLAY=:0.0 before plotting.

    sns.set(rc={'figure.figsize':(25,25)})

    # colors = ["firebrick", "gold", "orange", "purple", "hotpink", "palegreen", "darkcyan", "darkblue", "khaki", "lightcoral", "lawngreen", "teal", "sienna", "plum", "slateblue", "darkorchid", "slategray", "aqua", "magenta", "thistle", "peachpuff", "navy", "skyblue", "mediumvioletred"]
    colors = [ '#{:02x}{:02x}{:02x}'. format(c[2], c[1], c[0]) for i, c in enumerate(SEM_ID_TO_COLOR) if i!=0]
    sns.set_style("darkgrid")
    sns.set_palette(colors)

    #Create dataframe with each label names, group, and counts.
    df = pd.DataFrame({'Labels': keys,
        'Type': ['Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Indoor Floor', 'Indoor Floor', 'Indoor Floor', 'Outdoor Floor', 'Hybrid Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Hybrid Floor', 'Hybrid Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Hybrid Floor', 'Hybrid Floor', 'Hybrid Floor'],
        'Proportion': values,
        'Counts': labels_counts_dictionary.values()})
    df = df.sort_values(by=['Type','Proportion'], ascending=False)

    #Plot and set fields.
    print("Plotting")
    cats = ['Outdoor Floor', 'Indoor Floor', 'Hybrid Floor']
    # sns.set_theme(style='white')
    sns.set(style='white', font_scale=4)

    # Set the figure size
    # ax = sns.barplot(x='Type', y='Proportion', hue='Labels', data=df, palette=colors, width=1)
    ax = sns.barplot(x='Labels', y='Proportion', data=df, palette=colors, width=1)

    # # Create legend handles
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, df['Labels'])]

    # Remove x-ticks
    plt.xticks([])
    # plt.legend(labels=df['Labels'])
    plt.legend(handles=legend_handles, title='Semantic Class', ncol=2, fontsize=32)
    # ax.legend(loc='upper right', fontsize=36, ncol=2)
    ax.set(xlabel=None)
    ax.set(ylabel="% Total Annotations")

    for idx, i in enumerate(ax.patches):
        p = i.get_height()
        p_text = f'{p:.3f}'
        x = i.get_x() + i.get_width()/2
        y = i.get_y() + p + 0.001

        plt.text(x, y, df['Counts'][idx], ha='center', va='bottom', size=30, rotation=90)

    # plt.title('UT CODA Terrain Segmentation Class Breakdown')

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
        "Pedestrian"            : "Mammal",
        "Horse"                 : "Mammal",
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
    plt.subplots_adjust(top=0.55)
    # plt.subplots_adjust(top=0.1)
    plt.savefig("%s/%s.png"%(outdir, name), format='png', dpi=300)
        

def main(args):
    #Get file paths and loop throught to sum each label
    indir = "/robodata/arthurz/Datasets/CODa_dev"
    outdir = "%s/plots" % os.getcwd()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

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

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

""""""
    # cats = ['Outdoor Floor', 'Indoor Floor', 'Hybrid Floor']
    # hue = labels_list
    # center_positions = np.arange(len(cats)) * len(hue) + (len(hue) - 1) / 2
    # plot = sns.catplot(x='Type', y='Proportion', hue='Labels', data=df, palette=colors, kind="bar", 
    #     height=15, 
    #     legend_out=False, # move legend onto plot
    #     dodge=True, # Set to false to stack class bars
    #     aspect=1.25,
    #     width=1,
    #     center = center_positions
    # )
    # # plot._legend.set_bbox_to_anchor((0.75, 1))
    # plt.legend(loc='upper right', ncol=2)
    # plot.set(xlabel=None)
    # plot.set(ylabel="Proportion %")

    # # ax = plot.facet_axis(0, 0)  # or ax = g.axes.flat[0]
    # # # x = np.arange(3)
    # # # plot.set_xticks(x + 0.25,("Outdoor Flooring", "Indoor Flooring", "Hybrid Flooring"))
    # # for c in ax.containers:
    # #     class_counts = [str(labels_counts_dictionary[cls_name]) for cls_name in df['Labels']]

    # #     ax.bar_label(c, labels=class_counts, label_type='edge')
    # for ax in plot.axes.flat:
    #     for bar_idx, bar in enumerate(ax.containers):
    #         cls_name = df['Labels'][bar_idx]
    #         cls_count = labels_counts_dictionary[cls_name]
    #         ax.bar_label(bar, fmt='%0.3f', rotation=90)  # Format the label with desired precision

    # for i, bar in enumerate(plot.axes.patches): 

    #     # move the missing to the centre
    #     current_width = bar.get_width()
    #     current_pos = bar.get_x()
    #     if i == 5:
    #         bar.set_x(current_pos-(current_width/2))
    #         # move also the std mark
    #         plot.axes.lines[i].set_xdata(current_pos)
""""""