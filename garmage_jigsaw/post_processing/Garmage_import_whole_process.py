
import os
import glob
import time
import shutil
import subprocess
import spy
import zipfile

ENVIRONMENT = "<Path-To-Environment>/python.exe"
ARRANGEMENT_PYTHON = "<Path-To>/arrangement_processone.py"

if __name__ == "__main__":
    data_root = "<Path-To-XXXXX>/garmagejigsaw_output"
    print(f"path exist:{os.path.exists(data_root)}")
    arrangement_root = data_root.replace("garmagejigsaw_output", "arrangement")
    os.makedirs(arrangement_root, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(data_root, '**', 'garment.json'), recursive=True))
    print(f"Total {len(all_files)} data..")

    for idx, input_json_fp in enumerate(all_files):

        # item process start time
        tic = time.perf_counter()

        # item path
        item_basename = os.path.basename(os.path.dirname(input_json_fp)).strip()
        arrangement_item_dir = os.path.join(
            arrangement_root,
            item_basename
        )
        os.makedirs(arrangement_item_dir, exist_ok=True)

        # import generated json
        _ = spy.AIGP.ImportPatternJson(input_json_fp)

        # export json and triangulated panels (.zip)
        spy.AIGP.ExportPatternJson(os.path.join(arrangement_item_dir, 'pattern.json'), True)

        # copy original garmage file
        shutil.copy(os.path.join(os.path.dirname(input_json_fp), "orig_data.pkl"), arrangement_item_dir)

        # unzip json and obj
        zip_fp = glob.glob(os.path.join(arrangement_item_dir, "*.zip"))[0]
        unzip_dir = os.path.join(arrangement_item_dir, "tmp_unzip")
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_fp, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

        shutil.move(os.path.join(unzip_dir, "pattern.json"), os.path.join(arrangement_item_dir, "pattern.json"))
        shutil.move(os.path.join(unzip_dir, "pattern.obj"), os.path.join(arrangement_item_dir, "pattern_flatten.obj"))

        if os.path.exists(zip_fp):
            os.remove(zip_fp)
        if os.path.exists(unzip_dir):
            shutil.rmtree(unzip_dir)

        # run arrangement for triangulated panels
        _ = subprocess.run([ENVIRONMENT, ARRANGEMENT_PYTHON, "--data_path", arrangement_item_dir], check=True)

        # apply arrangement result
        spy.AIGP.ImportPatternJson(os.path.join(arrangement_item_dir, "pattern.json"))
        clothIds = spy.GetAllClothPieceIds()
        for clothId in clothIds:
            name = spy.GetEntityName(clothId)
            objPath = os.path.join(arrangement_item_dir, "mesh", name + ".obj")
            spy.AIGP.ReplaceMesh3DByObj(clothId, objPath)

        # save sproj
        sproj_fp = os.path.join(arrangement_item_dir, "Arranged.sproj")
        ret = spy.SaveProject(sproj_fp)

        # item process finish time
        toc = time.perf_counter()
        print("[DONE] Processing time for %s: %.4f" % (input_json_fp, toc - tic))

    print("All items processing finished.")