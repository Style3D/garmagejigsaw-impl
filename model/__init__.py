import os

from .modules import *
# This import must be at the top. DO NOT change the order.


def build_model(cfg):
    module_list = cfg.MODULE.lower().split('.')
    if module_list[0] == 'jigsaw':
        from .jigsaw import JointSegmentationAlignmentModel
        return JointSegmentationAlignmentModel(cfg)
    elif module_list[0] == 'jigsaw_stylexd':
        from .jigsaw_stylexd import JointSegmentationAlignmentModel
        return JointSegmentationAlignmentModel(cfg)
    else:
        raise NotImplementedError(f'Model {cfg.MODULE.lower()} not implemented')

def build_point_classifier(cfg):
    from .jigsaw_stylexd.point_classifier import PointClassifier
    return PointClassifier(cfg)

def build_stitch_predictor(cfg):
    from .jigsaw_stylexd.stitch_predictor import StitchPredictor

    pc_cls_ckpt_path = cfg.WEIGHT_FILE_POINTCLASSIFIER
    if not os.path.exists(pc_cls_ckpt_path):
        raise IOError(f"{pc_cls_ckpt_path} is not exist")
    pointclassifier_model = build_point_classifier(cfg).load_from_checkpoint(pc_cls_ckpt_path).cuda()
    pointclassifier_model.pc_cls_threshold = cfg.MODEL.PC_CLS_THRESHOLD
    pointclassifier_model.freeze()

    return StitchPredictor(cfg, pointclassifier_model)
