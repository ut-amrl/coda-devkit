import os
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

from helpers.constants import SEM_CLASS_TO_ID, SEM_ID_TO_COLOR
from helpers.sensors import read_sem_label

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

def plot_counts(indir, cached_path="./GEN_semlabeldict.json"):
    files = (absoluteFilePaths(indir))
    labels_list = np.array(list(SEM_CLASS_TO_ID.keys()))

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

    colors = ["firebrick", "gold", "orange", "purple", "hotpink", "palegreen", "darkcyan", "darkblue", "khaki", "lightcoral", "lawngreen", "teal", "sienna", "plum", "slateblue", "darkorchid", "slategray", "aqua", "magenta", "thistle", "peachpuff", "navy", "skyblue", "mediumvioletred"]
    sns.set_style("darkgrid")
    # sns.set_palette(SEM_ID_TO_COLOR)
    sns.set(font_scale=4)

    #Create dataframe with each label names, group, and counts.
    df = pd.DataFrame({'Labels': keys,
        'Type': ['Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Indoor Floor', 'Indoor Floor', 'Indoor Floor', 'Outdoor Floor', 'Hybrid Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Hybrid Floor', 'Hybrid Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Hybrid Floor', 'Hybrid Floor', 'Hybrid Floor'],
        'Proportion': values,
        'Counts': labels_counts_dictionary.values()})
    df = df.sort_values(by=['Type','Proportion'], ascending=False)

    #Plot and set fields.
    print("Plotting")
    cats = ['Outdoor Floor', 'Indoor Floor', 'Hybrid Floor']

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
    ax.set(ylabel="Proportion %")

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
    import pdb; pdb.set_trace()
    plt.savefig("GEN_semlabelhist.png", format='png', dpi=300)

def main():
    #Get file paths and loop throught to sum each label
    indir = "/robodata/arthurz/Datasets/CODa_dev/3d_semantic/os1"
    #List of labels to use to get label name from index, Dictionary to keep track of total for each label
    plot_counts(indir)

if __name__ == "__main__":
    main()

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