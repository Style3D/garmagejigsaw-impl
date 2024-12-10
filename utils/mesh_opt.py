import numpy as np

# calculate mean edge length of a list of mesh.
def cal_mean_edge_len(meshes:list):
    # 计算平均边长
    v_el, v_er = [], []
    for mesh in meshes:
        edges = np.array(mesh.edges)
        # Set e_sample, the sample frequency for a mesh
        if len(edges) < 400:
            e_sample = 2
        elif len(edges) < 1000:
            e_sample = 4
        elif len(edges) < 10000:
            e_sample = 6
        elif len(edges) < 50000:
            e_sample = 20
        else:
            e_sample = 50
        v_el.append(np.array(mesh.vertices)[np.concatenate(edges, axis=0)[0::e_sample]])
        v_er.append(np.array(mesh.vertices)[np.concatenate(edges, axis=0)[1::e_sample]])
    v_el = np.concatenate(v_el, axis=0)
    v_er = np.concatenate(v_er, axis=0)
    mean_edge_len = np.mean(np.sqrt(np.sum((v_el - v_er) ** 2, axis=1)))

    return  mean_edge_len


def compute_adjacency_list(faces, num_vertices):
    adjacency_list = [[] for _ in range(num_vertices)]

    for face in faces:
        i, j, k = face

        if j not in adjacency_list[i]:
            adjacency_list[i].append(j)
        if k not in adjacency_list[i]:
            adjacency_list[i].append(k)

        if i not in adjacency_list[j]:
            adjacency_list[j].append(i)
        if k not in adjacency_list[j]:
            adjacency_list[j].append(k)

        if i not in adjacency_list[k]:
            adjacency_list[k].append(i)
        if j not in adjacency_list[k]:
            adjacency_list[k].append(j)

    return adjacency_list
