import splitfolders
import os

input ='./images/rgb'
output = './images_split'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input, output=output, seed=1337, ratio=(.7, .15, .15), group_prefix=None) # default values

