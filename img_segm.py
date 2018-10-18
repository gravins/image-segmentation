from sklearn.cluster import KMeans
import random
from PIL import Image
import pandas as pd
from sklearn import preprocessing
import sys
import skimage
from skimage import segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import numpy as np

if len(sys.argv) not in range(4, 6):
    raise SyntaxError("You must specify 4 arguments value:\n\tnumber of cluster\n\tpath for input file\n\tpath for output file\n\tuse -norm to enabled normalization (default disabled)")

try:
    cluster_number = int(sys.argv[1])

    # Open image
    img = Image.open(sys.argv[2])
except:
    raise SyntaxError("You must specify 4 arguments value:\n\tnumber of cluster\n\tpath for input file\n\tpath for output file\n\tuse -norm to enabled normalization (default disabled)")

if len(sys.argv) == 5 and sys.argv[4] not in ["-norm"]:
    raise SyntaxError("You must specify normalization mode with \"-norm\". By default is not enabled")
else:
    normalize = True if sys.argv[4] == "-norm" and len(sys.argv) == 5 else False

outputName = sys.argv[3]

def colors(n):
    """
    Generate n random distinct rgb colors
    :param n: number of color to generate
    :return: list of rgb colors
    """
    ret = []
    red = int(random.random() * 256)
    green = int(random.random() * 256)
    blue = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        red += step
        green += step
        blue += step
        red = int(red) % 256
        green = int(green) % 256
        blue = int(blue) % 256
        ret.append((red, green, blue))
    return ret


def getAverageRGB(clrs):
    """
    Given set of RGB colors, return average value of color as (r, g, b)
    :param clrs: set of RGB colors
    :return: average between rgb color
    """
    # no. of pixels in set
    npixels = len(clrs)

    sumRGB = [0, 0, 0]
    for c in clrs:
        for i in range(3):
            sumRGB[i] += c[i]

    avg = (round(sumRGB[0]/npixels), round(sumRGB[1]/npixels), round(sumRGB[2]/npixels))

    return avg


# Define random color for clusters
cluster_color = colors(cluster_number)

# Create k-means model
kmean = KMeans(n_clusters=cluster_number)

# Insert information of all pixels (rgb color and x,y position) into Pandas DataFrame
imageW = img.size[0]
imageH = img.size[1]


# Convert image into Lab color space
LABimg = skimage.color.rgb2lab(img)

data = {"r": [], "g": [], "b": [], "L": [], "A": [], "B": [], "x": [], "y": []}
for y in range(0, imageH):
    for x in range(0, imageW):

        rgb = img.getpixel((x, y))

        data["r"].append(rgb[0])
        data["g"].append(rgb[1])
        data["b"].append(rgb[2])
        data["L"].append(LABimg[y][x][0])
        data["A"].append(LABimg[y][x][1])
        data["B"].append(LABimg[y][x][2])
        data["x"].append(x)
        data["y"].append(y)

df = pd.DataFrame(data={"r": data["r"], "g": data["g"], "b": data["b"]})
df_lab = pd.DataFrame(data={"L": data["L"], "A": data["A"], "B": data["B"]})
df_pos = pd.DataFrame(data={"r": data["r"], "g": data["g"], "b": data["b"], "x": data["x"], "y": data["y"]})
df_lab_pos = pd.DataFrame(data={"L": data["L"], "A": data["A"], "B": data["B"], "x": data["x"], "y": data["y"]})

if normalize:
    # Standarize the values of features
    df = pd.DataFrame(data=preprocessing.normalize(df))
    df_pos = pd.DataFrame(data=preprocessing.normalize(df_pos))
    df_lab = pd.DataFrame(data=preprocessing.normalize(df_lab))
    df_lab_pos = pd.DataFrame(data=preprocessing.normalize(df_lab_pos))

# Run k-means
res = kmean.fit_predict(df)
res_pos = kmean.fit_predict(df_pos)
res_lab = kmean.fit_predict(df_lab)
res_lab_pos = kmean.fit_predict(df_lab_pos)

# Average color for each cluster
j = 0
avg_color = [[] for _ in range(cluster_number)]
avg_color_pos = [[] for _ in range(cluster_number)]
avg_color_lab = [[] for _ in range(cluster_number)]
avg_color_lab_pos = [[] for _ in range(cluster_number)]
for y in range(0, imageH):
    for x in range(0, imageW):
        avg_color[res[j]].append(img.getpixel((x, y)))
        avg_color_pos[res_pos[j]].append(img.getpixel((x, y)))
        avg_color_lab[res_lab[j]].append(img.getpixel((x, y)))
        avg_color_lab_pos[res_lab_pos[j]].append(img.getpixel((x, y)))
        j += 1

avg_color = [getAverageRGB(avg_c) for avg_c in avg_color]
avg_color_pos = [getAverageRGB(avg_c) for avg_c in avg_color_pos]
avg_color_lab = [getAverageRGB(avg_c) for avg_c in avg_color_lab]
avg_color_lab_pos = [getAverageRGB(avg_c) for avg_c in avg_color_lab_pos]

# Save segmented image
image = []
for i in range(0, 8):
    image.append(Image.new("RGB", (imageW, imageH)))

j = 0
for y in range(0, imageH):
    for x in range(0, imageW):
        # random color for:
        # rgb
        image[0].putpixel((x, y), cluster_color[res[j]])
        # rgb + position
        image[1].putpixel((x, y), cluster_color[res_pos[j]])
        # lab
        image[2].putpixel((x, y), cluster_color[res_lab[j]])
        # lab + position
        image[3].putpixel((x, y), cluster_color[res_lab_pos[j]])

        # avg color for:
        # rgb
        image[4].putpixel((x, y), avg_color[res[j]])
        # rgb + position
        image[5].putpixel((x, y), avg_color_pos[res_pos[j]])
        # lab
        image[6].putpixel((x, y), avg_color_lab[res_lab[j]])
        # lab + position
        image[7].putpixel((x, y), avg_color_lab_pos[res_lab_pos[j]])

        j += 1

fig, ax = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(20, 15))

j = 0
for i in range(len(ax)):
    for k in range(int(len(image)/2)):
        ax[i][k].imshow(image[j])
        if k == 0 or k == 2:
            name = "RGB "
        elif k == 1 or k == 3:
            name = "LAB "
        if i == 0:
            ax[i][k].set_title(name+"Without (x, y) position")
        else:
            ax[i][k].set_title(name + "With (x, y) position")
        j += 2
    j = 1

for a in ax:
    for i in range(int(len(image)/2)):
        a[i].axis('off')

plt.tight_layout()
plt.savefig(outputName, dpi=800)


"""
    Run Normalized Cut segmentation
"""

img = np.asarray(img)

# Segment the image using SLIC algorithm
labels1 = segmentation.slic(img, compactness=30, n_segments=400)

# Replace each pixel with the average RGB color of its region
out1 = color.label2rgb(labels1, img, kind='avg')

# Crate the Region Adjacency Graphs
# Each node in the RAG represents a set of pixels within image with the same
# label in labels. The weight between two adjacent regions represents how
# similar or dissimilar two regions are depending on the mode parameter
g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[0].set_title("Superpixel view")
ax[1].imshow(out2)
ax[1].set_title("NCut segmetnation result")

for a in ax:
    a.axis('off')

plt.tight_layout()

name = outputName[:-4]
outputName = name + "_ncut.png"
plt.savefig(outputName, dpi=600)
