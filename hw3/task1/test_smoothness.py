import trimesh
import numpy as np
import os

def evaluate_smoothness_by_normals(obj_path):
    """
    基于法线一致性评估光滑度。
    返回值：粗糙度分数（越小表示越光滑），基于 1 - (平均法线点积)
    """
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"警告: {obj_path} 未能解析为三角网格。")
            return None
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            print(f"警告: {obj_path} 为空网格。")
            return None

        # 确保顶点法线已计算
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.compute_vertex_normals()
        
        normals = mesh.vertex_normals
        # 获取每个顶点的邻居顶点索引列表
        neighbors = mesh.vertex_neighbors  # 返回值是 list of lists
        
        avg_dot_list = []
        for i, neighs in enumerate(neighbors):
            if len(neighs) == 0:
                continue
            # 当前顶点法线与所有邻居法线的点积
            dots = np.dot(normals[i], normals[neighs].T)
            avg_dot_list.append(np.mean(dots))
        
        if not avg_dot_list:
            print(f"警告: {obj_path} 没有有效的邻居关系。")
            return None
        
        # 整体平均点积，值域 [-1, 1]，越接近 1 越光滑
        overall_avg_dot = np.mean(avg_dot_list)
        # 转化为粗糙度分数：越小越光滑，理想光滑平面为 0
        roughness_score = 1.0 - overall_avg_dot
        # 也可以输出标准差等附加信息
        return {
            'file': obj_path,
            'file_name': os.path.basename(obj_path),
            'roughness_score': roughness_score,   # 核心指标，越小越光滑
            'avg_dot': overall_avg_dot,
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces)
        }
    except Exception as e:
        print(f"处理 {obj_path} 时出错: {e}")
        return None

def batch_compare(obj_paths):
    results = []
    for p in obj_paths:
        print(f"正在处理: {p}")
        res = evaluate_smoothness_by_normals(p)
        if res:
            results.append(res)
            print(f"  成功，粗糙度分数 = {res['roughness_score']:.6f}")
        else:
            print(f"  跳过 {p}")
    
    if not results:
        print("没有成功分析的模型。")
        return
    
    results.sort(key=lambda x: x['roughness_score'])
    print("\n" + "="*60)
    print("光滑度评估报告 (按粗糙度分数升序排列，值越小表示越光滑)")
    print("="*60)
    print(f"{'排名':<4} | {'文件名':<30} | {'粗糙度分数':<15} | {'法线一致性'}")
    print("-"*70)
    for i, res in enumerate(results, 1):
        print(f"{i:<4} | {res['file_name']:<30} | {res['roughness_score']:<15.6f} | {res['avg_dot']:.6f}")
    print("="*60)

# --- 主程序 ---
if __name__ == "__main__":
    # 请替换为您的三个OBJ文件的实际路径
    my_files = [
        "multi/multi.obj",
        "pic23d/pic23d.obj",
        "text23d/text23d.obj"
    ]
    # 检查文件是否存在
    valid_files = [f for f in my_files if os.path.exists(f)]
    if len(valid_files) < 2:
        print("至少需要两个有效的OBJ文件。")
    else:
        batch_compare(valid_files)