# coding=utf-8

"""
Routine for DDDM data exploration
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
from python import DDDM
import matplotlib.pyplot as plt
plt.style.use('ggplot')

train_df = pd.read_csv(os.path.join(DDDM.FLAGS.data_dir, 'driver_imgs_list.csv'))
print('Total number of train images:', train_df.shape[0])
print('Dataset Columns:', train_df.columns)
print('Total number of drivers in training set:', train_df['subject'].drop_duplicates().shape[0])
test_img_path = os.path.join(DDDM.FLAGS.data_dir, 'imgs/test')
test_img_names = os.listdir(test_img_path)
print('Total number of test images:', len(test_img_names))

# Plot classes bar diagram
plt.figure(1)
pd.get_dummies(train_df['classname']).sum().plot.bar()
plt.xlabel('Classes')
plt.ylabel('# of Occurrence')

# Plot Drivers bar diagram
plt.figure(2)
pd.get_dummies(train_df['subject']).sum().plot.bar()
plt.xlabel('Drivers')
plt.ylabel('# of Occurrence')

plt.show()
