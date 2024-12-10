try:
    import os.path
    import meshplot as mp
    import numpy as np

    # 启动 meshplot
    mp.offline()

    def meshes_combine(meshes):
        V, F = [], []
        for mesh in meshes:
            # 合并面，调整第二个网格的面索引
            faces2_adjusted = mesh.faces + len(V)
            if len(F) == 0:
                F = faces2_adjusted
            else:
                F = np.vstack([F, faces2_adjusted])

            # 合并顶点
            if len(V) == 0:
                V = mesh.vertices
            else:
                V = np.vstack([V, mesh.vertices])

        return V, F

    def meshes_visualize(meshes, name="meshes.html"):
        if not name.endswith(".html"):
            name = name + ".html"

        V, F = meshes_combine(meshes)

        mp.plot(V, F).save(os.path.join("_tmp/meshes_visualize", name))
except ImportError:
    def meshes_visualize(meshes, name="meshes.html"):
        pass