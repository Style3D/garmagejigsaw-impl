import json
from copy import deepcopy


def save_result(garment_json, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(garment_json, f, indent=4)
    # for i in range(len(garment_json["stitches"])):
    #     garment_json_ = deepcopy(garment_json)
    #     garment_json_["stitches"] = garment_json_["stitches"][:i+1]
    #
    #     save_path_ = save_path.split(".")[0]+f"_{i+1}.json"
    #
    #     with open(save_path_, 'w', encoding='utf-8') as f:
    #         json.dump(garment_json_, f, indent=4)
    #     a=1