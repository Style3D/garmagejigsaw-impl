"""
STEP 3

Run In IDE

Export triangulated flatten panel from sproj file.
"""


import os
import shutil
import zipfile
import argparse
import subprocess
from glob import glob


# Constant
_CACHE_ROOT = "C:/Users/dev-1/AppData/Local/Style3DTest"
_CACHE_DIRs = [
    "Project",
    "AutoSaveProject"
    "Preference/CacheFile",
    "Preference/projectAutoSaveInfos.json",
    "Preference/ProjectCopy"
    ]
_CACHE_DIRs = [os.path.join(_CACHE_ROOT,p) for p in _CACHE_DIRs]
_STYLE_3D = "D:/Style3D_test_2024-07-03_11-53-46/Style3DTest/Style3DTest.exe"


def clear_cache():
    print('>>> Removing cache dir...')
    for cache_dir in _CACHE_DIRs:
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        elif os.path.exists(cache_dir) and os.path.isfile(cache_dir):
            os.remove(cache_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="<garmagenet-output-dir>/arrangement"
    )
    args = parser.parse_args()

    data_root = args.data_root

    sprojs_list = sorted(glob(os.path.join(data_root, "**", "*.sproj"), recursive=True))
    print(f"Total {len(sprojs_list)} data..")

    for idx, sproj_fp in enumerate(sprojs_list):

        clear_cache()

        try:
            args = [_STYLE_3D,
                    "--aiExport",
                    sproj_fp.replace(".sproj", ".zip"),
                    sproj_fp
                    ]
            result = subprocess.run(args, capture_output=True, text=True)

            zip_fp = sproj_fp.replace(".sproj", ".zip")
            unzip_dir = os.path.join(os.path.dirname(sproj_fp), os.path.basename(sproj_fp).replace(".sproj", ""))
            os.makedirs(unzip_dir, exist_ok=True)
            with zipfile.ZipFile(zip_fp, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)

            shutil.move(os.path.join(unzip_dir, "pattern.json"), os.path.join(os.path.dirname(sproj_fp), "pattern.json"))
            obj_name = os.path.basename(sproj_fp).replace(".sproj", ".obj")
            shutil.move(os.path.join(unzip_dir,obj_name ), os.path.join(os.path.dirname(sproj_fp), obj_name))

            if os.path.exists(zip_fp):
                os.remove(zip_fp)
            if os.path.exists(unzip_dir):
                shutil.rmtree(unzip_dir)
        except Exception:
            print(f"Failed processing :{sproj_fp}")
            continue


    print("Process finished")