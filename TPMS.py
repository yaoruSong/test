# -*- coding: utf-8 -*-
"""
基于遗传算法生成 Gyroid 型 TPMS 实体
输出: ABAQUS .inp 文件 与 PNG 图片至 D:\\demo_plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams

try:
    from skimage.measure import marching_cubes
    HAS_MARCHING_CUBES = True
except ImportError:
    HAS_MARCHING_CUBES = False

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ============== Gyroid TPMS 隐式函数与体素化 ==============

def gyroid_value(x, y, z, L=2 * np.pi):
    """
    Gyroid 隐式函数 (无量纲):
    cos(X)*sin(Y) + cos(Y)*sin(Z) + cos(Z)*sin(X)
    其中 X=2πx/L, Y=2πy/L, Z=2πz/L
    """
    X = 2 * np.pi * x / L
    Y = 2 * np.pi * y / L
    Z = 2 * np.pi * z / L
    return np.cos(X) * np.sin(Y) + np.cos(Y) * np.sin(Z) + np.cos(Z) * np.sin(X)


def voxelize_gyroid(thickness, L, n_grid, n_units=1):
    """
    将 Gyroid 实体体素化: |f| <= thickness 为实体相
    n_units: 每个方向周期数; n_grid: 每个周期网格数
    """
    n_total = n_units * n_grid + 1
    xs = np.linspace(0, n_units * L, n_total)
    ys = np.linspace(0, n_units * L, n_total)
    zs = np.linspace(0, n_units * L, n_total)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    f = gyroid_value(xx, yy, zz, L)
    solid = np.abs(f) <= thickness
    return solid, (xs, ys, zs)


# ============== 遗传算法 ==============

def decode_chromosome(chromosome, t_range=(0.15, 0.55), L_range=(3.0, 8.0)):
    """将 [0,1] 染色体解码为 (thickness, L)"""
    t = t_range[0] + chromosome[0] * (t_range[1] - t_range[0])
    L = L_range[0] + chromosome[1] * (L_range[1] - L_range[0])
    return t, L


def fitness(thickness, L, n_grid=12, n_units=1, target_vol_frac=0.35):
    """
    适应度: 负的与目标体积分数偏差（体积分数越接近 target 越好）
    """
    solid, _ = voxelize_gyroid(thickness, L, n_grid, n_units)
    vol_frac = np.mean(solid)
    return -np.abs(vol_frac - target_vol_frac)


def genetic_algorithm(
    pop_size=30,
    n_gen=15,
    n_grid=12,
    n_units=1,
    target_vol_frac=0.18,
    t_range=(0.15, 0.55),
    L_range=(3.0, 8.0),
):
    """简单遗传算法优化 Gyroid 参数 (thickness, L)"""
    np.random.seed(42)
    # 种群: 每个体 [t_norm, L_norm]
    pop = np.random.rand(pop_size, 2)
    best_chromosome = None
    best_fit = -np.inf

    for gen in range(n_gen):
        fits = np.array(
            [
                fitness(
                    *decode_chromosome(ind, t_range, L_range),
                    n_grid=n_grid,
                    n_units=n_units,
                    target_vol_frac=target_vol_frac,
                )
            for ind in pop
        ])
        # 精英
        idx_best = np.argmax(fits)
        if fits[idx_best] > best_fit:
            best_fit = fits[idx_best]
            best_chromosome = pop[idx_best].copy()
        # 选择 (锦标赛)
        next_pop = [pop[idx_best].copy()]
        for _ in range(pop_size - 1):
            i, j = np.random.randint(0, pop_size, 2)
            winner = pop[i] if fits[i] >= fits[j] else pop[j]
            child = winner.copy()
            # 交叉
            if np.random.rand() < 0.7:
                k = np.random.randint(0, pop_size)
                child = 0.5 * (child + pop[k])
            # 变异
            if np.random.rand() < 0.2:
                child += np.random.randn(2) * 0.1
            child = np.clip(child, 0, 1)
            next_pop.append(child)
        pop = np.array(next_pop)

    return decode_chromosome(best_chromosome, t_range, L_range)


# ============== 导出 ABAQUS .inp (C3D8) ==============

def _voxel_to_points_cells(solid, spacing):
    """体素 -> 节点坐标 (N,3) 与六面体单元 (M,8)，0-based 索引，供 meshio 使用。"""
    if np.isscalar(spacing):
        dx = dy = dz = spacing
    else:
        dx, dy, dz = spacing[0], spacing[1], spacing[2]
    ni, nj, nk = solid.shape
    ox, oy, oz = 0.0, 0.0, 0.0

    node_id = 0
    node_map = {}
    points_list = []

    for i in range(ni + 1):
        for j in range(nj + 1):
            for k in range(nk + 1):
                node_map[(i, j, k)] = node_id
                points_list.append([ox + i * dx, oy + j * dy, oz + k * dz])
                node_id += 1

    cells_list = []
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if not solid[i, j, k]:
                    continue
                n1 = node_map[(i, j, k)]
                n2 = node_map[(i + 1, j, k)]
                n3 = node_map[(i + 1, j + 1, k)]
                n4 = node_map[(i, j + 1, k)]
                n5 = node_map[(i, j, k + 1)]
                n6 = node_map[(i + 1, j, k + 1)]
                n7 = node_map[(i + 1, j + 1, k + 1)]
                n8 = node_map[(i, j + 1, k + 1)]
                cells_list.append([n1, n2, n3, n4, n5, n6, n7, n8])

    points = np.array(points_list, dtype=np.float64)
    cells = np.array(cells_list, dtype=np.int64)
    return points, cells


def voxel_to_abaqus_inp(solid, spacing, out_path):
    """
    将体素实体转为 ABAQUS C3D8 网格并写入 .inp。
    始终手写 .inp，显式写 *Part, name=Part-1，避免 AbaqusNameError（meshio 生成的 .inp 无 Part 名）。
    """
    points, cells = _voxel_to_points_cells(solid, spacing)
    n_nodes, n_elems = len(points), len(cells)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="ascii", newline="\r\n") as f:
        f.write("*Heading\n")
        f.write("Gyroid_TPMS_Model\n")
        f.write("*Part, name=Part1\n")
        f.write("*Node\n")
        for i in range(1, n_nodes + 1):
            p = points[i - 1]
            f.write("%d, %.6e, %.6e, %.6e\n" % (i, p[0], p[1], p[2]))
        f.write("*Element, type=C3D8\n")
        for i in range(1, n_elems + 1):
            c = cells[i - 1]
            f.write("%d, %d, %d, %d, %d, %d, %d, %d, %d\n" % (
                i, c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1,
                c[4] + 1, c[5] + 1, c[6] + 1, c[7] + 1))
        f.write("*End Part\n")
    return n_nodes, n_elems


# ============== 绘制 3D 连续曲面并保存 PNG（依赖 scikit-image marching_cubes）==============

def plot_gyroid_and_save(solid, spacing, png_path, thickness, L, n_units=1):
    """用 scikit-image 的 marching_cubes 提取等值面，绘制连续三角网格曲面并保存 PNG"""
    ni, nj, nk = solid.shape
    if np.isscalar(spacing):
        dx = dy = dz = spacing
    else:
        dx, dy, dz = spacing[0], spacing[1], spacing[2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    verts = None

    if HAS_MARCHING_CUBES:
        # scikit-image: 等值面 |f| - thickness = 0 为实体边界
        xs = np.linspace(0, (ni - 1) * dx, ni)
        ys = np.linspace(0, (nj - 1) * dy, nj)
        zs = np.linspace(0, (nk - 1) * dz, nk)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        f = gyroid_value(xx, yy, zz, L)
        level_set = np.abs(f) - thickness
        try:
            verts, faces, _, _ = marching_cubes(level_set, 0, spacing=(dx, dy, dz))
            # 用三角面片绘制连续曲面（青蓝色 + 深蓝边线）
            mesh = Poly3DCollection(
                verts[faces],
                facecolor="#2E86AB",
                edgecolor="#1a5276",
                linewidths=0.15,
                alpha=0.95,
            )
            ax.add_collection3d(mesh)
        except Exception:
            _fallback_scatter(ax, solid, dx, dy, dz)
    else:
        _fallback_scatter(ax, solid, dx, dy, dz)

    # 根据顶点范围设置坐标轴
    if verts is not None and len(verts) > 0:
        ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    else:
        ext = (ni - 1) * dx
        ax.set_xlim(0, ext)
        ax.set_ylim(0, ext)
        ax.set_zlim(0, ext)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Gyroid TPMS 实体 (遗传算法优化参数)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def _fallback_scatter(ax, solid, dx, dy, dz):
    """无 skimage 时退回表面体素散点"""
    ni, nj, nk = solid.shape
    mask = np.zeros_like(solid, dtype=bool)
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if not solid[i, j, k]:
                    continue
                if (
                    (i == 0 or not solid[i - 1, j, k])
                    or (i == ni - 1 or not solid[i + 1, j, k])
                    or (j == 0 or not solid[i, j - 1, k])
                    or (j == nj - 1 or not solid[i, j + 1, k])
                    or (k == 0 or not solid[i, j, k - 1])
                    or (k == nk - 1 or not solid[i, j, k + 1])
                ):
                    mask[i, j, k] = True
    xc = (np.arange(ni) + 0.5) * dx
    yc = (np.arange(nj) + 0.5) * dy
    zc = (np.arange(nk) + 0.5) * dz
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
    xs = xx[mask].ravel()
    ys = yy[mask].ravel()
    zs = zz[mask].ravel()
    step = max(1, len(xs) // 8000)
    ax.scatter(
        xs[::step], ys[::step], zs[::step],
        c="#2E86AB", s=8, alpha=0.8, edgecolors="none",
    )


# ============== 主流程 ==============

def main():
    out_dir = r"D:\demo_plot"
    os.makedirs(out_dir, exist_ok=True)

    n_grid = 14
    n_grid_export = 45
    n_units = 1
    target_vol_frac = 0.18

    print("遗传算法优化 Gyroid 参数 (thickness, L)...")
    best_t, best_L = genetic_algorithm(
        pop_size=25,
        n_gen=12,
        n_grid=n_grid,
        n_units=n_units,
        target_vol_frac=target_vol_frac,
    )
    print(f"  最优厚度 t = {best_t:.4f}, 周期 L = {best_L:.4f}")

    print("体素化 Gyroid（粗网格，用于校验体积分数）...")
    solid_coarse, (xs_c, ys_c, zs_c) = voxelize_gyroid(best_t, best_L, n_grid, n_units)
    vol_frac = np.mean(solid_coarse)
    print(f"  体积分数 = {vol_frac:.4f}")

    print("精细体素化（用于 .inp 与 PNG）...")
    solid, (xs, ys, zs) = voxelize_gyroid(best_t, best_L, n_grid_export, n_units)
    dx = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else best_L / n_grid_export
    dy = (ys[-1] - ys[0]) / (len(ys) - 1) if len(ys) > 1 else dx
    dz = (zs[-1] - zs[0]) / (len(zs) - 1) if len(zs) > 1 else dx
    spacing = (dx, dy, dz)

    inp_path = os.path.join(out_dir, "Gyroid_TPMS.inp")
    png_path = os.path.join(out_dir, "Gyroid_TPMS.png")

    print("导出 ABAQUS .inp（精细网格，Part 名 Part1）...")
    n_nodes, n_elems = voxel_to_abaqus_inp(solid, spacing, inp_path)
    print(f"  节点数: {n_nodes}, 单元数: {n_elems}")

    if HAS_MARCHING_CUBES:
        print("绘制并保存 PNG（scikit-image 等值面 + 连续曲面）...")
    else:
        print("绘制并保存 PNG（建议安装 scikit-image 以得到连续曲面: pip install scikit-image）...")
    plot_gyroid_and_save(solid, spacing, png_path, thickness=best_t, L=best_L, n_units=n_units)

    print(f"完成。结果已保存至: {out_dir}")
    print(f"  - {inp_path}")
    print(f"  - {png_path}")
    print("")
    print("Abaqus 使用提示:")
    print("  1. 导入: File -> Import -> Model 选择上述 .inp。")
    print("  2. 使用插件 Create geometry from mesh 时：插件对话框中的「名称」必须填写（如 Gyroid），不能留空，否则会报 AbaqusNameError: name is empty。")


if __name__ == "__main__":
    main()
