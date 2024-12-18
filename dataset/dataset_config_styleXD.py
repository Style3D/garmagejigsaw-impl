from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# Breaking Bad geometry assembly dataset

__C.STYLEXD = edict()
__C.STYLEXD.DATA_DIR = "data/obj_format_part_under20"
__C.STYLEXD.DATA_KEYS = ("part_sids",)

__C.STYLEXD.DATA_TYPES = edict()
__C.STYLEXD.DATA_TYPES.TEST = []  # 多选["StyleGen", "StyleGenML", "Brep128", "Brep256", "Brep512", "Brep1024"]

__C.STYLEXD.SHUFFLE = True


# [modified]
__C.STYLEXD.POINT_SPLIT = 1.
__C.STYLEXD.PCS_NOISE_STRENGTH = 1.

__C.STYLEXD.NUM_PC_POINTS = 5000  # points per part
__C.STYLEXD.MIN_PART_POINT = (
    30  # if sampled by area, want to make sure all piece have >30 points
)
__C.STYLEXD.MIN_NUM_PART = 2
__C.STYLEXD.MAX_NUM_PART = 20
__C.STYLEXD.SHUFFLE_PARTS = False

# 在Panel上进行点采样的方式
"""
TYPES：
  stitch
  boundary_pcs 
  boundary_mesh
"""
__C.STYLEXD.PCS_SAMPLE_TYPE = edict()
__C.STYLEXD.PCS_SAMPLE_TYPE.TRAIN = "stitch"
__C.STYLEXD.PCS_SAMPLE_TYPE.VAL = "stitch"
__C.STYLEXD.PCS_SAMPLE_TYPE.TEST = "boundary_pcs"

# 在读取mesh之后，对mesh进行缩水操作（最外面一圈点往内部收缩）
__C.STYLEXD.SHRINK_MESH = edict()
__C.STYLEXD.SHRINK_MESH.TRAIN = False
__C.STYLEXD.SHRINK_MESH.VAL = False
__C.STYLEXD.SHRINK_MESH.TEST = False
__C.STYLEXD.SHRINK_MESH_PARAM = edict()
__C.STYLEXD.SHRINK_MESH_PARAM.TRAIN = 0.
__C.STYLEXD.SHRINK_MESH_PARAM.VAL = 0.
__C.STYLEXD.SHRINK_MESH_PARAM.TEST = 0.

__C.STYLEXD.SHUFFLE_POINTS_AFTER_SAMPLE = False
__C.STYLEXD.PCS_NOISE_TYPE = "default"


# 对采样后的点云按panel作为梯进行加噪
__C.STYLEXD.PANEL_NOISE_TYPE = edict()
__C.STYLEXD.PANEL_NOISE_TYPE.TRAIN = "default"
__C.STYLEXD.PANEL_NOISE_TYPE.VAL = "default"
__C.STYLEXD.PANEL_NOISE_TYPE.TEST = "default"
__C.STYLEXD.SCALE_RANGE = 0.04
__C.STYLEXD.ROT_RANGE = 1.  # random rotation range for each part
__C.STYLEXD.TRANS_RANGE = 1.  # random translation range for each part

# 是否沿着缝合线加噪（仅训练）
__C.STYLEXD.USE_STITCH_NOISE = False
__C.STYLEXD.STITCH_NOISE_STRENGTH = 1.
__C.STYLEXD.STITCH_NOISE_RANDOM_RANGE = [0.8, 2.2]

__C.STYLEXD.LENGTH = -1
__C.STYLEXD.TEST_LENGTH = -1
__C.STYLEXD.OVERFIT = -1

# 点分类预测中的阈值
__C.STYLEXD.PC_CLS_THRESHOLD = 0.8

__C.STYLEXD.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
    [255, 128, 0],
    [128, 0, 255],
    [0, 255, 128],
    [255, 0, 128],
    [128, 255, 0],
    [0, 128, 255],
    [255, 128, 128],
    [128, 128, 255],
    [255, 255, 128],
    [128, 255, 255],
    [255, 128, 255],
    [128, 255, 128]
]