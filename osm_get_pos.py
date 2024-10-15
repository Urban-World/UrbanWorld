import bpy
import os
import sys
import argparse
import math
import numpy as np
from mathutils import Vector
from math import atan2, degrees, radians
import bpy

def get_object_dimensions(obj):
    return obj.dimensions

def adjust_path_road_scale(obj_a_path, target_obj):
    imported_objects = import_obj(obj_a_path, "ReferenceObject")
    reference_obj = imported_objects[0]
    reference_obj.dimensions[1] = 0

    target_dimensions = get_object_dimensions(target_obj)
    source_dimensions = get_object_dimensions(reference_obj)
    scale_factors = [source_dimensions[i] / target_dimensions[i] for i in [0, 2]]  
    
    bpy.data.objects.remove(reference_obj, do_unlink=True)

    return scale_factors[0], 1.0, scale_factors[1]

def adjust_object_scale(obj_a_path, target_obj):
    imported_objects = import_obj(obj_a_path, "ReferenceObject")

    reference_obj = imported_objects[0]

    scale_factors = calculate_scale_factors(target_obj, reference_obj)

    reference_obj_name = reference_obj.name  
    if reference_obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[reference_obj_name], do_unlink=True)

    return scale_factors




def calculate_scale_factors(source_obj, target_obj):
    target_dimensions = get_object_dimensions(target_obj)
    # print(target_dimensions)
    source_dimensions = get_object_dimensions(source_obj)
    # print(source_dimensions)
    
    scale_factors = [target_dimensions[i] / source_dimensions[i] for i in range(3)]
    
    return scale_factors

def import_obj(filepath, name):
    objects_before = set(bpy.data.objects)
    bpy.ops.import_scene.obj(filepath=filepath)
    objects_after = set(bpy.data.objects)
    new_objects = objects_after - objects_before

    for i, obj in enumerate(new_objects):
        obj.name = f"{name}_{i}" if len(new_objects) > 1 else name

    return list(new_objects)  

def get_object_height(obj):
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min([v.z for v in bbox])
    max_z = max([v.z for v in bbox])
    height = max_z - min_z
    return height

def import_obj_and_get_centroids_building(obj_file_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    
    object_centroids = {}
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and 'building' in obj.name:
            mesh = obj.data
            total_vert = len(mesh.vertices)
            local_center = sum((vert.co for vert in mesh.vertices), Vector()) / total_vert
            global_center = obj.matrix_world @ local_center
            object_centroids[obj.name] = tuple(global_center)            

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    return object_centroids

def import_obj_and_get_centroids_others(obj_file_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    
    object_centroids = {}
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and 'building' not in obj.name:
            mesh = obj.data
            total_vert = len(mesh.vertices)
            local_center = sum((vert.co for vert in mesh.vertices), Vector()) / total_vert
            global_center = obj.matrix_world @ local_center
            object_centroids[obj.name] = tuple(global_center)            

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    return object_centroids

def calculate_ground_plane_size(buildings_centroids):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for centroid in buildings_centroids.values():
        min_x = min(min_x, centroid[0])
        max_x = max(max_x, centroid[0])
        min_y = min(min_y, centroid[1])
        max_y = max(max_y, centroid[1])
    
    width = max_x - min_x
    depth = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    return width, depth, center_x, center_y

def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def get_object_bottom_z(obj):
    min_z = float('inf')
    for vert in obj.data.vertices:
        world_coord = obj.matrix_world @ vert.co
        if world_coord.z < min_z:
            min_z = world_coord.z
    return min_z

def set_object_bottom_to_z_zero(obj):
    apply_transforms(obj)
    
    min_z = get_object_bottom_z(obj)
    
    offset = -min_z
    
    obj.location.z += offset
    
def load_and_combine_objects(obj_save_path, osm_info, export_combined_path):
    scene_obj = bpy.data.objects.new("SceneObject", None)
    bpy.context.collection.objects.link(scene_obj)
    
    for obj_name in osm_info.keys():
        if os.path.exists(os.path.join(obj_save_path, obj_name, "output/refine/UV_inpaint_res_0.png")):
            obj_file = os.path.join(obj_save_path, obj_name, "output/res-0/mesh.obj")
            bpy.ops.import_scene.obj(filepath=obj_file)
            del_obj_name = 'Cube'
            del_obj = bpy.data.objects.get(del_obj_name)
            if del_obj:
                bpy.data.objects.remove(del_obj, do_unlink=True)
            obj = bpy.context.selected_objects[-1]
            texture_path = os.path.join(obj_save_path, obj_name, "output/refine/UV_inpaint_res_0.png")
            material = bpy.data.materials.new(name="Material_" + obj_name)
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            
            tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')
            tex_image.image = bpy.data.images.load(texture_path)
            
            hue_saturation = material.node_tree.nodes.new('ShaderNodeHueSaturation')
            hue_saturation.inputs['Saturation'].default_value = 1.3
            
            bright_contrast = material.node_tree.nodes.new('ShaderNodeBrightContrast')
            bright_contrast.inputs['Bright'].default_value = 0.1
            bright_contrast.inputs['Contrast'].default_value = 0.3
            
            material.node_tree.links.new(tex_image.outputs['Color'], hue_saturation.inputs['Color'])
            material.node_tree.links.new(hue_saturation.outputs['Color'], bright_contrast.inputs['Color'])
            material.node_tree.links.new(bright_contrast.outputs['Color'], bsdf.inputs['Base Color'])
            
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)

            obj.location = osm_info[obj_name]

            if 'path_road' in obj_name:
                print(os.path.join(obj_save_path, obj_name,'mesh.obj'))
                scale = adjust_path_road_scale(os.path.join(obj_save_path, obj_name,'mesh.obj'), obj)
                obj.scale = scale
                set_object_bottom_to_z_zero(obj)
            elif 'building' in obj_name:
                scale = adjust_object_scale(os.path.join(obj_save_path, obj_name,'mesh.obj'), obj)
                obj.scale = scale
                set_object_bottom_to_z_zero(obj)
            elif 'forest' in obj_name:
                scale = adjust_object_scale(os.path.join(obj_save_path, obj_name,'mesh.obj'), obj)
                obj.scale = scale
                set_object_bottom_to_z_zero(obj)
            elif 'vegetation' in obj_name:
                scale = adjust_path_road_scale(os.path.join(obj_save_path, obj_name,'mesh.obj'), obj)
                obj.scale = scale
                set_object_bottom_to_z_zero(obj)
            elif 'water' in obj_name:
                scale = adjust_object_scale(os.path.join(obj_save_path, obj_name,'mesh.obj'), obj)
                obj.scale = scale
                set_object_bottom_to_z_zero(obj)
            elif 'area' in obj_name:
                scale = adjust_object_scale(os.path.join(obj_save_path, obj_name,'mesh.obj'), obj)
                obj.scale = scale
                set_object_bottom_to_z_zero(obj)


            obj.parent = scene_obj

    bpy.ops.export_scene.gltf(filepath=export_combined_path, export_format='GLTF_SEPARATE')
    # bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.cycles.use_denoising = True
    # bpy.context.scene.render.resolution_x = 1920
    # bpy.context.scene.render.resolution_y = 1080
    # bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    # bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    # bpy.context.scene.render.filepath = os.path.join(obj_save_path, 'video.mp4') 


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument(
        "--obj_save_path",
        type=str,
        default='',
        help="Obj save path",
    )
    opt.add_argument(
        "--osm_save_path",
        type=str,
        default=None,
        help="OSM save path",
    )
    opt.add_argument(
    "--max_lat",
    type=float,
    default=40.001,
    help="Max lat of OSM area",
    )
    opt.add_argument(
        "--min_lat",
        type=float,
        default= 39.998,
        help="Min lat of OSM area",
    )
    opt.add_argument(
        "--max_lon",
        type=float,
        default=116.327,
        help="Max lon of OSM area",
    )
    opt.add_argument(
        "--min_lon",
        type=float,
        default=116.326,
        help="Min lon of OSM area",
    )

    argv = sys.argv[sys.argv.index("--") + 1 :]
    opt = opt.parse_args(argv)

    osm_info_building = import_obj_and_get_centroids_building(os.path.join(opt.obj_save_path,'map_all/mesh.obj'))  
    osm_info_others = import_obj_and_get_centroids_others(os.path.join(opt.obj_save_path,'map_all/mesh.obj'))    
    osm_info = {**osm_info_building, **osm_info_others}
    if not os.path.exists(os.path.join(opt.obj_save_path, 'gltf_model')):
        os.mkdir(os.path.join(opt.obj_save_path, 'gltf_model'))
    export_combined_path = os.path.join(opt.obj_save_path, 'gltf_model/combined.gltf')
    load_and_combine_objects(opt.obj_save_path, osm_info, export_combined_path)
    print(osm_info)
