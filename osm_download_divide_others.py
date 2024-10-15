import bpy
import os
import shutil
import sys
import argparse
import numpy as np
import bmesh

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

def download_divide(max_lat, min_lat, max_long, min_long, obj_save_path, osm_save_path, mtl_file, png_file, addonpath, asset_save_path):
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

            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris()

            bpy.ops.object.mode_set(mode='OBJECT')


        bpy.context.view_layer.update()

        for obj in bpy.context.selected_objects:
            if 'profile' not in obj.name and 'building' not in obj.name:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='OBJECT')
                obj.select_set(True)

                solidify_modifier = obj.modifiers.new(name='Solidify', type='SOLIDIFY')
                solidify_modifier.thickness = 3  

                bpy.ops.object.modifier_apply(modifier='Solidify')

                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
            
                override = bpy.context.copy()
                for area in bpy.context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        override['area'] = area
                        override['region'] = area.regions[-1]
                        override['space_data'] = area.spaces.active
                        break

                bpy.ops.uv.smart_project(override)
                
                bpy.ops.object.mode_set(mode='OBJECT')

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

    bpy.context.view_layer.update()

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
    argv = sys.argv[sys.argv.index("--") + 1 :]
    opt = opt.parse_args(argv)
    #=======Start downloading OSM files=======
    download_divide(opt.max_lat, opt.min_lat, opt.max_lon, opt.min_lon, opt.obj_save_path, opt.osm_save_path, opt.mtl_file, opt.png_file, opt.addonpath, opt.asset_save_path)
