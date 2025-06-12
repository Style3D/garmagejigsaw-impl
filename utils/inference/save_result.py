import json, os
import pickle
import shutil
from glob import glob


def save_result(save_dir, data_id=0, garment_json=None, fig=None, g_basename=None, vis_resource=None, mesh_file_path=None):
    os.makedirs(save_dir, exist_ok=True)
    if g_basename is None:
        garment_dir = os.path.join(save_dir, f"garment_" + f"{data_id}".zfill(5) )
    else:
        garment_dir = os.path.join(save_dir, g_basename)
    os.makedirs(garment_dir, exist_ok=True)
    save_path = os.path.join(garment_dir, f"garment"+ ".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(garment_json, f, indent=4)

    if fig is not None:
        fig.write_html(os.path.join(garment_dir,"vis_comp.html"))

    if vis_resource is not None:
        with open(os.path.join(garment_dir,"vis_resource.pkl"), 'wb') as f:
            pickle.dump(vis_resource, f)

    orig_data_list = glob(os.path.join(mesh_file_path, "original_data", "*.pkl"))
    assert len(orig_data_list) == 1
    shutil.copy(orig_data_list[0], os.path.join(garment_dir, "orig_data.pkl"))