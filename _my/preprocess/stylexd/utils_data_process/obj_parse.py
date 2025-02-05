# 用于解析 StyleXD中的单个obj文件

import numpy as np

def parse_obj_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    vertices, faces, stitches, UVs= [], [], [], []
    nps = []  # 每个 part 的顶点数
    nfs = []  # 每个 part 的face数

    # 解析obj文件
    previous_start_with = ""
    current_start_with = ""
    for line in lines:
        line = line.strip()
        if line.startswith('v '):
            current_start_with = "v"
            Vertex = [float(s) for s in line[2:].split(" ")]
            vertices.append(np.array(Vertex))
        elif line.startswith('f '):
            current_start_with = "f"
            face = [int(s.split("/")[0]) - 1 for s in line[2:].split(" ")]
            faces.append(np.array(face))
        elif line.startswith('l '):
            current_start_with = "l"
            stitch = [int(s) for s in line[2:].split(" ")]
            stitches.append(np.array(stitch))
        elif line.startswith('vt '):
            current_start_with = "vt"
            uv = [float(s) for s in line[3:].split(" ")][:2]
            UVs.append(np.array(uv,dtype=np.float32))
        else:
            previous_start_with = "#"
            continue

        if current_start_with == "v":
            if previous_start_with != "v":
                nps.append(1)
            else:
                nps[-1] += 1
        elif current_start_with == "f":
            if previous_start_with != "f":
                nfs.append(1)
            else:
                nfs[-1] += 1
        previous_start_with = current_start_with

    obj_dict=dict(
        vertices=vertices,
        faces=faces,
        stitch=stitches,
        uv=UVs,
        nps = nps,
        nfs = nfs
    )
    return obj_dict