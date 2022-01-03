import sys
import os
sys.path.append(os.getcwd())
import open3d as o3d
import numpy as np
from glob import glob
import os
from sklearn.neighbors import KDTree
from models import gen_utils


SAMPLING_NUM = 32000
SAVE_PATH = "sampled_cls_full"
os.makedirs(os.path.join('data', SAVE_PATH), exist_ok=True)
result_name="sugo"

for case_idx in range(3,11):
    stl_path_list = glob(os.path.join("data",f"case{case_idx}","Crown model","*.stl"))
    full_stl_path_list = glob(os.path.join("data",f"case{case_idx}","Aligned fullArch Model and Transform value","*.stl"))

    global_mesh = o3d.io.read_triangle_mesh(full_stl_path_list[0])
    global_mesh = global_mesh.remove_duplicated_vertices()
    is_up = gen_utils.get_up_from_name(full_stl_path_list[0])
    global_mesh_arr = np.asarray(global_mesh.vertices)
    global_mesh_arr = np.concatenate([global_mesh_arr, np.zeros((global_mesh_arr.shape[0], 1))], axis=1)

    tree = KDTree(global_mesh_arr[:,:3], leaf_size=2)

    for stl_path in stl_path_list:
        if(gen_utils.get_number_from_name(stl_path)>=30):
            if is_up:
                continue
        else:
            if not is_up:
                continue
        tooth_number = gen_utils.get_number_from_name(stl_path)%10
        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh = mesh.remove_duplicated_vertices()
        mesh_arr = np.asarray(mesh.vertices)
        dist, indexs = tree.query(mesh_arr, k=1)
        for point_num, corr_idx in enumerate(indexs):
            global_mesh_arr[corr_idx[0], 3] = tooth_number

    sample_global_mesh = global_mesh.sample_points_poisson_disk(SAMPLING_NUM)

    sample_global_mesh_arr = np.asarray(sample_global_mesh.points)
    sample_global_mesh_n_arr = np.asarray(sample_global_mesh.normals)
    sample_global_mesh_arr = np.concatenate([sample_global_mesh_arr, sample_global_mesh_n_arr], axis=1)
    sample_global_mesh_arr = np.concatenate([sample_global_mesh_arr, np.zeros((SAMPLING_NUM,1))], axis=1)

    indexs = tree.query_radius(sample_global_mesh_arr[:,:3], 0.2)
    count_only_ls = tree.query_radius(sample_global_mesh_arr[:,:3], 0.1, count_only=True)

    for point_num, corr_idx in enumerate(indexs):
        if(len(corr_idx)!=0):
            sample_global_mesh_arr[point_num,6] = global_mesh_arr[corr_idx[0],3]
            
        
    sample_min = np.min(sample_global_mesh_arr[:, :3])
    sample_global_mesh_arr[:, :3] -= sample_min
    sample_max = np.max(sample_global_mesh_arr[:, :3])
    sample_global_mesh_arr[:, :3] /= sample_max

    name_id = result_name+"_"+str(case_idx)
    ""

    cp_dict = {}
    for stl_path in stl_path_list:
        if(gen_utils.get_number_from_name(stl_path)>=30):
            if is_up:
                continue
        else:
            if not is_up:
                continue

        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh = mesh.remove_duplicated_vertices()
        mesh_arr = np.asarray(mesh.vertices)
        mesh_arr -= sample_min
        mesh_arr /= sample_max
        cp_dict[gen_utils.get_number_from_name(stl_path)] = np.mean(mesh_arr, axis=0)
    cp_arr = []
    cp_exists = []

    if is_up:
        base = 11
    else:
        base = 31

    for class_idx in range(8):
        tooth_num = base+class_idx
        if tooth_num in cp_dict.keys():
            cp_arr += [cp_dict[tooth_num]]
            cp_exists.append(1)
        else:
            cp_arr += [np.array([-10,-10,-10])]
            cp_exists.append(0)

    for class_idx in range(8):
        tooth_num = base+class_idx+10
        if tooth_num in cp_dict.keys():
            cp_arr += [cp_dict[tooth_num]]
            cp_exists.append(1)
        else:
            cp_arr += [np.array([-10,-10,-10])]
            cp_exists.append(0)

    np.save(os.path.join("data",SAVE_PATH,f"{name_id}_mesh"), sample_global_mesh_arr)
    np.save(os.path.join("data",SAVE_PATH,f"{name_id}_centroid"), np.array(cp_arr))
    np.save(os.path.join("data",SAVE_PATH,f"{name_id}_centroid_exist"), np.array(cp_exists))