import os
import seaborn as sns
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#Done with help of Chaitanya

#Get all file paths to iterate through
def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

#List of labels to use to get label name from index
labels_list = [
    "Unlabeled",
    "Concrete",
    "Grass",
    "Rocks",
    "Speedway Bricks",
    "Red Bricks",
    "Pebble Pavement",
    "Light Marble Tiling",
    "Dark Marble Tiling",
    "Dirt Paths",
    "Road Pavement",
    "Short Vegetation",
    "Porcelain Tile",
    "Metal Grates",
    "Blond Marble Tiling",
    "Wood Panel",
    "Patterned Tile",
    "Carpet",
    "Crosswalk",
    "Dome Mat",
    "Stairs",
    "Door Mat",
    "Threshold",
    "Metal Floor",
    "Unknown"
]

#Dictionary to keep track of total for each label
labels_dictionary = {
    "Unlabeled":0,  
    "Concrete":0,  
    "Grass":0,                          
    "Rocks":0,                          
    "Speedway Bricks":0,                   
    "Red Bricks":0,
    "Pebble Pavement":0,
    "Light Marble Tiling":0,
    "Dark Marble Tiling":0,
    "Dirt Paths":0,
    "Road Pavement":0,
    "Short Vegetation":0,
    "Porcelain Tile":0,
    "Metal Grates":0,
    "Blond Marble Tiling":0,
    "Wood Panel":0,
    "Patterned Tile":0,
    "Carpet":0,
    "Crosswalk":0,
    "Dome Mat":0,
    "Stairs":0,
    "Door Mat":0,
    "Threshold":0,
    "Metal Floor":0,
    "Unknown":0
}

#Update the dictionary of labels every time a label is found.
def count_labels(annotated_data):
    for label in annotated_data:
        labels_dictionary[labels_list[label]] += 1

#Get file paths and loop throught to sum each label
files = (absoluteFilePaths("/robodata/arthurz/Datasets/CODa_dev/3d_semantic/os1"))
for file in files:
    with open(file, "rb") as annotated_file:
        annotated_data = list(annotated_file.read())  # [] 130,000 labels(indices)
        count_labels(annotated_data)

#Calculate percentages
total = sum(labels_dictionary.values())
proportion = total - labels_dictionary["Unlabeled"]
for key in labels_dictionary:
    key_sum = labels_dictionary[key]
    labels_dictionary[key] = key_sum/proportion

#Sort dictionary in descending order
sorted_labels_descending = sorted(labels_dictionary.items(), key=lambda x:x[1], reverse=True)

#Clean and get labels and counts 
labels_dictionary = dict(sorted_labels_descending)
del labels_dictionary["Unlabeled"]
keys = list(labels_dictionary.keys())
values = [labels_dictionary[k] for k in keys]

#Plot data in bargraph (Using Seaborn)
#Run export DISPLAY=:0.0 before plotting.

sns.set(rc={'figure.figsize':(20,15)})

colors = ["firebrick", "gold", "orange", "purple", "hotpink", "palegreen", "darkcyan", "darkblue", "khaki", "lightcoral", "honeydew", "teal", "sienna", "plum", "slateblue", "darkorchid", "slategray", "aqua", "magenta", "thistle", "peachpuff", "navy", "skyblue", "linen"]
sns.set_style("darkgrid")
sns.set(font_scale=2)

#Create dataframe with each label names, group, and counts.
df = pd.DataFrame({'Labels': keys,
    'Type': ['Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Indoor Floor', 'Indoor Floor', 'Indoor Floor', 'Outdoor Floor', 'Hybrid Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Outdoor Floor', 'Outdoor Floor', 'Hybrid Floor', 'Hybrid Floor', 'Outdoor Floor', 'Outdoor Floor', 'Indoor Floor', 'Hybrid Floor', 'Hybrid Floor', 'Hybrid Floor'],
    'Proportion': values})

#Plot and set fields.
ax = sns.barplot(x='Type', y='Proportion', hue='Labels', data=df, palette=colors)
ax.legend(fontsize=17)
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.title('UT CODA Terrain Segmentation Class Breakdown')

#Save locally
plt.savefig("plot.png", format='png', dpi=300)