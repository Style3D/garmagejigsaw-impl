import os
from glob import glob
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_panels_num_info(choised_garments_dir, output_dir):
    panel_num_list = []
    max_panel_num = 0
    max_panel_num_dir = None
    min_panel_num = 9999
    min_panel_num_dir = None
    for garment_dir in choised_garments_dir:
        panel_num = len(glob(os.path.join(garment_dir, "piece_*")))
        panel_num_list.append(panel_num)
        if panel_num>max_panel_num:
            max_panel_num = panel_num
            max_panel_num_dir = garment_dir
        elif panel_num<min_panel_num:
            min_panel_num=panel_num
            min_panel_num_dir = garment_dir
    df =  pd.DataFrame(panel_num_list, columns=["Panel_Num"])
    sns.set_theme(style="darkgrid")
    sns.histplot(data=df, x="Panel_Num", kde=True)
    plt.savefig(os.path.join(output_dir, "panel_num.jpg"))
    with open(os.path.join(output_dir, "panel_num_min_max.json"), "w", encoding="utf-8") as f:
        json.dump({
            "min_panel_num":min_panel_num,
            "min_panel_num_dir":min_panel_num_dir,
            "max_panel_num":max_panel_num,
            "max_panel_num_dir":max_panel_num_dir,
            }, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    data_batchs = ["Q4"]
    dataset_dir = "data/stylexd_jigsaw/train"
    output_dir = "_my/another_script/results"

    all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))
    choised_garments_dir = []
    if "Q1" in data_batchs:
        choised_garments_dir.extend(all_garment_dir[0:890])
    if "Q2" in data_batchs:
        choised_garments_dir.extend(all_garment_dir[890:11077])
    if "Q4" in data_batchs:
        choised_garments_dir.extend(all_garment_dir[11077:12275])

    get_panels_num_info(choised_garments_dir, output_dir)