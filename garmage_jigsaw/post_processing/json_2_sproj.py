"""
STEP 1

Run In Style3D.

Import GarmageJigsaw output into style3d software,
and export sproj file.
"""

import os
import glob
import time
import shutil

import spy


if __name__ == "__main__":
    data_root = "<garmagenet-output-dir>/garmagejigsaw_output"
    print(f"path exist:{os.path.exists(data_root)}")
    output_dir = data_root.replace("garmagejigsaw_output", "arrangement")
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(data_root, '**', 'garment.json'), recursive=True))
    print(f"Total {len(all_files)} data..")

    for idx, data_fp in enumerate(all_files):

        tic = time.perf_counter()

        data_output_dir = os.path.join(
            output_dir,
            os.path.basename(os.path.dirname(data_fp)).strip()
        )
        sproj_fp = os.path.join(
            data_output_dir,
            os.path.basename(os.path.dirname(data_fp)).strip() + '.sproj'
        )
        if os.path.exists(sproj_fp): continue

        ret=None
        ret = spy.AIGP.ImportPatternJson(data_fp)
        ret = spy.SaveProject(sproj_fp)

        toc = time.perf_counter()

        print("[DONE] Processing time for %s: %.4f" % (data_fp, toc - tic))
        shutil.copy(os.path.join(os.path.dirname(data_fp), "orig_data.pkl"), data_output_dir)