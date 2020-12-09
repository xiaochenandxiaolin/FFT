# -*- coding: utf-8 -*-
import shutil
import re
import numpy as np
import glob
import os

files = np.sort(glob.glob("./data/*"))
filelist = os.listdir("./data/")
fault_folder = ['0','1']

for folder_name in fault_folder:
    os.mkdir(os.path.join("./data", folder_name))
for j in range(len(filelist)):
    if re.match(filelist[j], filelist[j]):
        label_path = filelist[j].split('-')[4]
        dest = os.path.join("./data/",eval('label_path'))
        shutil.move(files[j], dest)