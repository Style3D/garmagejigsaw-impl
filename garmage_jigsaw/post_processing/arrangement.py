"""
STEP 3

Run In IDE.

Arrange triangulated flatten panels by there corresponding garmage.
"""


import os
import json
import pickle
import argparse
from glob import glob
from copy import deepcopy

import cv2
import numpy as np
import trimesh
from trimesh import Trimesh


def _pad_arr(arr, pad_size=10):
    # pad size to each dimension, require tensor to have size (H,W, C)
    return np.pad(
        arr,
        ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='constant',
        constant_values=0)


def _denormalize_geo(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0 + (0, 1, 0)
    return pts * bbox_scale + bbox_offset


def _denormalize_uv(pts, bbox):
    pos_dim = pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0 + (0, 1)
    return pts * bbox_scale + bbox_offset


def normalize_pc(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    normalized_points = np.column_stack((x_norm, y_norm))
    return normalized_points


def get_mask_bbox(A):
    rows = np.any(A, axis=1)
    cols = np.any(A, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return np.array((y_min, y_max, x_min, x_max))


def dilate_garmage(mask_img, geo_img):
    """
    For each point that is False in mask_img, find the nearest boundary point
    and assign its geometry (dilation).

    :param mask_img:    RESO x RESO
    :param geo_img:     RESO x RESO x 3
    :return:            Dilated geometry image
    """

    geo_img_dilated = deepcopy(geo_img)

    mask = (mask_img * 255.0).astype(np.uint8)
    # Obtain the coordinates of boundary points
    _, thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    contour_indices = np.concatenate(list(contours)).reshape(-1, 2)

    # Geometry of the boundary points
    geo_contour = geo_img[contour_indices[:, 1], contour_indices[:, 0], :]

    # Points that require dilation (invalid points)
    unvalid_indices = np.argwhere(~mask_img)
    unvalid_indices = unvalid_indices[..., ::-1]

    # Distance matrix between all invalid points and all boundary points
    dis = np.sqrt(
        np.sum(unvalid_indices ** 2, axis=1)[:, None] +
        np.sum(contour_indices ** 2, axis=1)[None, :] -
        2 * unvalid_indices @ contour_indices.T
    )

    # Find the nearest boundary point for each invalid point
    min_dis_indices = np.argmin(dis, axis=1)
    dilation_geo = geo_contour[min_dis_indices]

    geo_img_dilated[
        unvalid_indices[:, 1],
        unvalid_indices[:, 0],
        :
    ] = dilation_geo

    return geo_img_dilated


def load_Garmage_pkl(garment_fp):
    """ load Garmage pkl file (Generated from GarmageNet)

    :param garment_fp:  file path
    :return:
        geo_wcs:    Geometry in world coordinate system
        mask:
        uv:         UV in world coordinate system
    """

    with open(garment_fp, "rb") as f:
        data = pickle.load(f)

    geo_ncs = data["surf_ncs"]
    # uv = data["surf_uv_ncs"]
    uv = None
    mask = data["surf_mask"]

    # erode mask to remove unreliable boundary points
    geo_ncs = geo_ncs.reshape(-1, 256*256, 3)
    mask = mask.reshape(-1, 256, 256)
    if mask.dtype != np.bool_:
        mask = mask>0.0

    surf_bbox = data["surf_bbox"]

    # dilate operation for each geo
    for i in range(len(geo_ncs)):
        mask[i] = erode_mask_once(mask[i])

        geo_dilated = dilate_garmage(mask[i].reshape(256, 256), geo_ncs[i].reshape(256, 256, 3))
        geo_ncs[i] = geo_dilated.reshape(-1, 3)


    # denormalize geo and uv to world space
    geo_wcs = _denormalize_geo(geo_ncs, surf_bbox)

    geo_wcs = geo_wcs.reshape(-1, 256, 256, 3)
    if uv is not None:
        uv = uv.reshape(-1, 256, 256, 2)
    mask = mask.reshape(-1, 256, 256)
    return geo_wcs, mask, uv


def load_pattern_json(pattern_json_path):
    """
    Load the JSON file exported from Style3D software
    (exported together with the OBJ file).

    :param pattern_json_path:
    :return:
    """

    pattern_json = json.load(open(pattern_json_path))
    panels_dict = {}
    id2label = {}
    for panel in pattern_json["panels"]:
        panels_dict[panel["label"]] = panel
        panel["center"] = np.array(panel["center"][:2])
        id2label[panel["id"]] = panel["label"]  # key: id, value: label
    return panels_dict, id2label


def load_garment_obj(obj_path, id2label):
    """
    Load an OBJ file exported from Style3D software
    (exported together with a corresponding JSON file).

    :param obj_path:
    :param id2label:
    :return:
    """

    with open(obj_path, 'r') as file:
        lines = file.readlines()

    _all_labels = []
    obj_dict = {}
    VTs = []
    faces = []
    for line in lines:
        line = line.strip()
        if line.startswith('vt '):
            line_split = line.split(" ")
            VTs.append([float(p) for p in line_split[1:3]])
        elif line.startswith('f '):
            faces.append([int(p.split("/")[2]) - 1 for p in line.split(" ")[1:4]])
        elif line.startswith('g '):
            id = line.split(" ")[-1]
            label = id2label[id]
            _all_labels.append(label)
            obj_dict[label] = {}
            obj_dict[label]["uv"] = np.array(VTs)
            VTs = []
        else:
            continue

    # Collect faces corresponding to each panel
    faces = np.array(sorted(faces, key=lambda x: x[0]))
    v_start = 0
    for label in _all_labels:
        v_num = len(obj_dict[label]["uv"])
        v_end = v_start+ v_num-1
        panel_faces = faces[np.bitwise_and(faces[:,0]>=v_start, faces[:,0]<=v_end)]
        panel_faces = panel_faces - v_start
        obj_dict[label]["faces"] = panel_faces
        v_start = v_end+1

    return obj_dict


def erode_mask_once(mask):
    """ Erode the mask once by CV method.

    :param mask:
    :return:
    """
    pad_size=5
    mask_img = (mask * 255.0).astype(np.uint8)
    mask_img = np.pad(mask_img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

    # erode once
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_img = cv2.erode(mask_img, kernel, iterations=1)
    mask_img[mask_img >= 150] = 255
    mask_img[mask_img < 150] = 0

    eroded_mask = mask_img[pad_size:-pad_size, pad_size:-pad_size]>=150
    return eroded_mask


def coor_sys_conversion(uv, center, RESO=256):
    """ For all vertices on a panel mesh, convert to RESOxRESO pixel coordinate system

    :param uv:          All vertices on the panel mesh
    :param center:      Center of the panel
    :param RESO:        Resolution
    :return:
    """

    # Convert global coordinates to local coordinates, normalize to [-128, 128]
    uv -= center
    uv = normalize_pc(uv) * 128

    # Get the bounding box center of the corresponding mask
    target_center = np.array([RESO - 1, RESO - 1]) / 2

    # Convert to pixel coordinates (255 / 256),
    # then further shrink according to the reduced mask
    # ((255 - 1 * 2) / 255), simplified to 253 / 256
    uv = uv * 253 / 256
    uv = uv + target_center

    return uv


def get_BI_target_weight(query_uv):
    x = query_uv[:, 0]
    y = query_uv[:, 1]

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    dx = x - x0
    dy = y - y0

    # Compute bilinear interpolation weights
    w1 = (1 - dx) * (1 - dy)
    w2 = (1 - dx) * dy
    w3 = dx * (1 - dy)
    w4 = dx * dy
    interp_weights = np.column_stack((w1, w2, w3, w4))

    # Coordinates of the interpolation points
    x_coords = np.column_stack((x0, x0, x0 + 1, x0 + 1))
    y_coords = np.column_stack((y0, y0 + 1, y0, y0 + 1))
    interp_coord = np.stack((x_coords, y_coords), axis=2)

    return interp_coord, interp_weights


def bilinear_interpolation(query_uv, geo, mask, show_every_panel=False):
    """
    For vertices transformed into the image coordinate system, we treat them as query points
    and obtain the corresponding 3D position of each point using bilinear interpolation.

    :param query_uv:    Vertices already transformed into the pixel coordinate system
    :param geo:
    :param mask:
    :param show_every_panel:
    :return:
    """

    # For each query location, compute the indices and weights of the 4 neighboring interpolation points
    interp_coord, interp_weights = get_BI_target_weight(query_uv)

    # Perform bilinear interpolation:
    # multiply the 3D positions of interpolation points by their weights to obtain target_geo
    interp_geo = geo[interp_coord[:, :, 1], interp_coord[:, :, 0]]
    interp_weights_geo = np.repeat(interp_weights, 3).reshape(interp_weights.shape[0], 4, 3)
    target_geo = np.sum(interp_weights_geo * interp_geo, axis=-2)
    return target_geo


def cal_unreliable_value(logist:np.array):
    unreliable_mask = logist<(1-1e-8)
    unreliable_value = np.sum(logist[unreliable_mask])
    return unreliable_value


def query_geo_img_one_panel(query_uv, panel_spec, geo_img, mask_img):
    """ Query the geometry image and assign initial 3D positions to all vertices of panel.

    Args:
        query_uv (np.ndarray): (N, 2). UV coordinates of all vertex in sewing pattern coordinate (i.e. software 2D coordinate system).
        panel_spec (Dict): panel JSON dict. with 'center', 'collisionLayer', 'id', 'label', 'particleDistance', 'seqEdges' fields.
        geo_img (np.ndarray): (256, 256, 3) normalized geometry image for the query panel.
        mask_img (np.ndarray): (256, 256, 1) contour mask for the query panel.
        mis-alignment, we temporarily decide to erode the panel to make sure all query uv could map to a valid coordinate
        in the geometry image (i.e. inside the mask region). Defaults to 1.
    Returns:
        _type_: _description_
    """

    print('*** query_uv: ', query_uv.shape, query_uv.min(), query_uv.max())
    print('*** panel_spec: ', panel_spec.keys())
    print('*** geo_img: ', geo_img.shape, geo_img.min(), geo_img.max())
    print('*** mask_img: ', mask_img.shape, mask_img.min(), mask_img.max())


    panel_center = panel_spec["center"]

    converted_uv = coor_sys_conversion(query_uv, panel_center, RESO=256)
    xyz = bilinear_interpolation(converted_uv, geo_img, mask_img, show_every_panel=False)
    return xyz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="<garmagenet-output-dir>/arrangement", type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    all_data_list = sorted(os.listdir(data_dir))

    for idx, data in enumerate(all_data_list):
        data_path = os.path.join(data_dir, data)
        obj_path = glob(os.path.join(data_path, "*.obj"))[0]
        pattern_json_path = os.path.join(data_path, "pattern.json")

        try:
            garmage_path = glob(os.path.join(data_path, "orig_data.pkl"))[0]
        except Exception:
            print("File missing, continue...")
            continue

        try:
            # load garmage
            geos, masks, _ = load_Garmage_pkl(garmage_path)

            panels_json_dict, id2label = load_pattern_json(pattern_json_path)
            obj_dict = load_garment_obj(obj_path, id2label)

            mesh_output_dir = os.path.join(data_path, "mesh")
            os.makedirs(mesh_output_dir, exist_ok=True)
            for label in obj_dict:
                idx = int(label)
                # query position for each vertices of the faltten triangulated panel.
                xyz = query_geo_img_one_panel(
                    obj_dict[label]["uv"],
                    panels_json_dict[label],
                    geos[idx], masks[idx],
                )

                # save output
                visuals = trimesh.visual.texture.TextureVisuals(uv=obj_dict[label]["uv"])
                T = Trimesh(
                    vertices=xyz,
                    visual=visuals,  # UV
                    faces=np.array(obj_dict[label]["faces"]),
                    process=False
                )
                T.export(os.path.join(mesh_output_dir, f"{label}.obj"))
        except Exception as e:
            print(e)
            continue