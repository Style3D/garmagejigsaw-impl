"""
STEP 4

Run In Style3D.

Import arranged sewing pattern.
"""


import os
from glob import glob

import spy

if __name__ == "__main__":
    data_root = "<garmagenet-output-dir>/arrangement"
    data_dir_list = sorted(glob(os.path.join(data_root, "*")))

    print(f"Total {len(data_dir_list)} data..")

    for idx, dirPath in enumerate(data_dir_list):

        try:
            if True:
                patternJsonPath = dirPath + "/pattern.json"
                print(patternJsonPath)
                print(os.path.exists(patternJsonPath))
                spy.AIGP.ImportPatternJson(patternJsonPath)
                print("Finish import AIGP")
            if True:
                clothIds = spy.GetAllClothPieceIds()
                for clothId in clothIds:
                    name = spy.GetEntityName(clothId)

                    print(f"clothId: {clothId}   name: {name}")
                    # if name != "03":
                    #    continue
                    print(str(clothId) + "," + name)
                    objPath = dirPath + "/mesh/" + name + ".obj"
                    print(objPath)
                    spy.AIGP.ReplaceMesh3DByObj(clothId, objPath)

                print("Finish import objs")
            sproj_fp = os.path.join(dirPath, "Garmage_placement.sproj")
            ret = spy.SaveProject(sproj_fp)
        except Exception:
            print(f"Failed process {dirPath}")
            continue

    print("Finished all...")