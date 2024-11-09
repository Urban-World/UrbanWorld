import bpy
import os
import shutil
import sys
import argparse
import numpy as np

import xml.etree.ElementTree as ET

def filter_and_merge_path_road(object_filepath):
    bpy.ops.object.select_all(action='DESELECT')
    count = 0
    for obj in bpy.context.scene.objects:
        if ('path' in obj.name or 'road' in obj.name) and 'building' not in obj.name:
            obj.select_set(True)
            count += 1
    
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        merged_obj = bpy.context.object  
        merged_obj.name = 'map.osm_path_road'  
        
        merged_obj.data.name = 'map.osm_path_road'  
    
    if count>0:
        bpy.ops.export_scene.obj(filepath=object_filepath, use_selection=True)

def filter_and_merge_vegetation(object_filepath):
    bpy.ops.object.select_all(action='DESELECT')
    count = 0
    for obj in bpy.context.scene.objects:
        if 'vegetation' in obj.name and 'building' not in obj.name:
            obj.select_set(True)
            count += 1
            print('veg+1')
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        merged_obj = bpy.context.object  
        merged_obj.name = 'map.osm_vegetation'  
        
        merged_obj.data.name = 'map.osm_vegetation'  
    print('veg num:',count)
    if count>0:
        bpy.ops.export_scene.obj(filepath=object_filepath, use_selection=True)


def filter_and_merge_forest(object_filepath):
    bpy.ops.object.select_all(action='DESELECT')
    count = 0
    
    for obj in bpy.context.scene.objects:
        if 'forest' in obj.name and 'building' not in obj.name:
            obj.select_set(True)
            count += 1
            print('forest+1')
    
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        merged_obj = bpy.context.object  
        merged_obj.name = 'map.osm_forest'  
        
        merged_obj.data.name = 'map.osm_forest'  
    print('forest num:',count)

    if count>0:
        bpy.ops.export_scene.obj(filepath=object_filepath, use_selection=True)

def filter_and_merge_water(object_filepath):
    bpy.ops.object.select_all(action='DESELECT')
    count = 0
    
    for obj in bpy.context.scene.objects:
        if 'water' in obj.name and 'building' not in obj.name:
            obj.select_set(True)
            count += 1
            print('water+1')
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        merged_obj = bpy.context.object  
        merged_obj.name = 'map.osm_water'  

        merged_obj.data.name = 'map.osm_water'  
    print('water num:',count)

    if count>0:
        bpy.ops.export_scene.obj(filepath=object_filepath, use_selection=True)

def filter_and_merge_area(object_filepath):
    bpy.ops.object.select_all(action='DESELECT')
    count = 0
    for obj in bpy.context.scene.objects:
        if 'area' in obj.name and 'building' not in obj.name:
            obj.select_set(True)
            count += 1
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        merged_obj = bpy.context.object  
        merged_obj.name = 'map.osm_area'  
        
        merged_obj.data.name = 'map.osm_area'  
    if count>0:
        bpy.ops.export_scene.obj(filepath=object_filepath, use_selection=True)

def merge_objects():
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()

def extract_building_heights(osm_file, default_building_height, default_building_level):
    tree = ET.parse(osm_file)
    root = tree.getroot()

    building_dict = {}
    building_counter = 1

    for way in root.findall('way'):
        building = False
        height = None
        levels = None

        # Check for "building" tags
        for tag in way.findall('tag'):
            if tag.attrib['k'] == 'building':
                building = True
            if tag.attrib['k'] == 'height':
                height = float(tag.attrib['v'])
            if tag.attrib['k'] == 'building:levels':
                if tag.attrib['v'].isdigit():
                    levels = int(tag.attrib['v'])


        if building:
            building_name = f"buildings.{building_counter:03d}"

            if height is not None:
                # If height is explicitly provided
                building_dict[building_name] = height
            elif levels is not None:
                # If building:levels exists, calculate height
                building_dict[building_name] = levels * default_building_height
            else:
                # Use default levels and default height
                building_dict[building_name] = default_building_level * default_building_height

            building_counter += 1

    return building_dict

def download_area(min_lat, max_lat, min_long, max_long, fp):
    bpy.data.scenes["Scene"].blosm.maxLat = max_lat
    bpy.data.scenes["Scene"].blosm.minLat = min_lat
    bpy.data.scenes["Scene"].blosm.maxLon = max_long
    bpy.data.scenes["Scene"].blosm.minLon = min_long
    bpy.ops.blosm.import_data()

    bpy.ops.export_scene.obj(filepath=fp)

def calculate_scene_center():
    obj_locations = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj_locations.append(np.array(obj.location))
    if obj_locations:
        center = np.mean(obj_locations, axis=0)
        return center
    return np.array([0.0, 0.0, 0.0])

def download_divide(max_lat, min_lat, max_long, min_long, obj_save_path, osm_save_path, mtl_file, png_file, addonpath, asset_save_path,export_path, default_level_height, default_building_level):
    fp = obj_save_path + 'map.obj'

    if not os.path.exists(obj_save_path):
        os.makedirs(obj_save_path)

    if not os.path.exists(osm_save_path):
        os.makedirs(osm_save_path)

    bpy.ops.preferences.addon_install(filepath=addonpath)
    bpy.ops.preferences.addon_enable(module="blosm")

    bpy.context.preferences.addons['blosm'].preferences.dataDir = osm_save_path
    bpy.context.preferences.addons['blosm'].preferences.assetsDir = asset_save_path
    bpy.context.preferences.addons['blosm'].preferences.googleMapsApiKey = ""

    bpy.data.scenes["Scene"].blosm.dataType = 'osm'
    bpy.data.scenes["Scene"].blosm.relativeToInitialImport = True

    download_area(min_lat, max_lat, min_long, max_long, fp)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    bpy.ops.import_scene.obj(filepath=fp)

    imported_objects = bpy.context.selected_objects
    print(imported_objects)

    if not imported_objects:
        print("Error: No objects were imported.")
    else:
        transform_data = {}

        for obj in imported_objects:
            print(obj.name)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.separate(type='LOOSE')
                # print('SEP')

            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris()

            bpy.ops.object.mode_set(mode='OBJECT')


        bpy.context.view_layer.update()
        obj_loc = {}
        osm_file = osm_save_path+'osm/map.osm'
        building_heights = extract_building_heights(osm_file, default_level_height, default_building_level)

        for obj in bpy.context.selected_objects:
            if 'profile' not in obj.name:
                obj_loc[obj.name] = obj.location
                if 'buildings' in obj.name:
                    for b in building_heights.keys():
                        if b in obj.name:
                            break
                    obj.dimensions.y = building_heights[b]
                
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='OBJECT')
                obj.select_set(True)

                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')

                override = bpy.context.copy()
                for area in bpy.context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        override['area'] = area
                        override['region'] = area.regions[-1]
                        override['space_data'] = area.spaces.active
                        break
                if 'building' in obj.name:
                    bpy.ops.uv.smart_project(override)

                bpy.ops.object.mode_set(mode='OBJECT')
                if 'building' in obj.name:
                    object_dir = os.path.join(obj_save_path, obj.name)
                    if not os.path.exists(object_dir):
                        os.makedirs(object_dir)

                    object_filepath = os.path.join(object_dir, "mesh.obj")
                    
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    bpy.ops.export_scene.obj(filepath=object_filepath, use_selection=True)

                    shutil.copy(mtl_file, object_dir)
                    shutil.copy(png_file, object_dir)
    print(obj_loc)

    print(obj_loc)
    bpy.context.view_layer.update()

    #merge path and road
    object_dir = os.path.join(obj_save_path, 'map.osm_path_road')
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    object_filepath = os.path.join(object_dir, "mesh.obj")
    filter_and_merge_path_road(object_filepath)
    
    #merge vegetation
    object_dir = os.path.join(obj_save_path, 'map.osm_vegetation')
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    object_filepath = os.path.join(object_dir, "mesh.obj")
    filter_and_merge_vegetation(object_filepath)
    
    #merge forest 
    object_dir = os.path.join(obj_save_path, 'map.osm_forest')
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    object_filepath = os.path.join(object_dir, "mesh.obj")
    filter_and_merge_forest(object_filepath)
    
    #merge water
    object_dir = os.path.join(obj_save_path, 'map.osm_water')
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    object_filepath = os.path.join(object_dir, "mesh.obj")
    filter_and_merge_water(object_filepath)

    #merge area
    object_dir = os.path.join(obj_save_path, 'map.osm_area')
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    object_filepath = os.path.join(object_dir, "mesh.obj")
    filter_and_merge_area(object_filepath)

    bpy.context.view_layer.update()
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    bpy.ops.export_scene.obj(filepath=export_path+'mesh.obj')


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    #Blender params
    opt.add_argument(
        "--max_lat",
        type=float,
        default=40.001,
        help="Max lat of OSM area",
    )
    opt.add_argument(
        "--min_lat",
        type=float,
        default=39.998,
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
    opt.add_argument(
        "--obj_save_path",
        type=str,
        default='',
        help="Saving path of the obj file",
    )
    opt.add_argument(
        "--osm_save_path",
        type=str,
        default='',
        help="Saving path of the osm file",
    )
    opt.add_argument(
        "--mtl_file",
        type=str,
        default='',
        help="Mtl file path",
    )
    opt.add_argument(
        "--png_file",
        type=str,
        default='',
        help="Initial material file path",
    )
    opt.add_argument(
        "--addonpath",
        type=str,
        default='',
        help="Addon Blosm file path",
    )
    opt.add_argument(
        "--asset_save_path",
        type=str,
        default="",
        help="Asset save path",
    )
    opt.add_argument(
        "--export_path",
        type=str,
        default="",
        help="Export path",
    )
    opt.add_argument(
        "--default_level_height",
        type=float,
        default=3.0
    )
    opt.add_argument(
        "--default_building_level",
        type=int,
        default=3
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    opt = opt.parse_args(argv)
    #=======Start downloading OSM files=======
    download_divide(opt.max_lat, opt.min_lat, opt.max_lon, opt.min_lon, opt.obj_save_path, opt.osm_save_path, opt.mtl_file, opt.png_file, opt.addonpath, opt.asset_save_path,opt.export_path, opt.default_level_height, opt.default_building_level)
