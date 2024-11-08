#OSM params
max_lat=40.7523
min_lat=40.7463
max_lon=-73.9810
min_lon=-73.9892

obj_save_path='./test_obj/'
osm_save_path='./test_osm/'
export_path='./test_obj/map_all/'
default_level_height=3
default_building_level=15

mtl_file='./initial.mtl'
png_file='./initial_material.png'

#Blosm addon settings
addonpath='./blosm.zip'
asset_save_path='./assets/'
blender_path='<your blender path>'

#Rendering params
sd_config_1='./controlnet/config/depth_based_osm_template.yaml'
sd_config_2='./controlnet/config/UV_based_osm_template.yaml'
render_config='./rendering/config/train_config_render.py'
# ip_img='<your image path>'
seed=1
   
    #Download, import and separate 3D scene from OSM
    source activate blend
    $blender_path -b -P osm_download_divide_terrain.py -- \
                --max_lat $max_lat \
                --min_lat $min_lat \
                --max_lon $max_lon \
                --min_lon $min_lon \
                --obj_save_path $obj_save_path \
                --osm_save_path $osm_save_path \
                --mtl_file $mtl_file \
                --png_file $png_file \
                --addonpath $addonpath \
                --asset_save_path $asset_save_path 
    

    $blender_path -b -P osm_download_divide_building.py -- \
                --max_lat $max_lat \
                --min_lat $min_lat \
                --max_lon $max_lon \
                --min_lon $min_lon \
                --obj_save_path $obj_save_path \
                --osm_save_path $osm_save_path \
                --mtl_file $mtl_file \
                --png_file $png_file \
                --addonpath $addonpath \
                --asset_save_path $asset_save_path \
                --export_path $export_path \
                --default_level_height $default_level_height \
                --default_building_level $default_building_level   

    $blender_path -b -P osm_download_divide_others.py -- \
                --max_lat $max_lat \
                --min_lat $min_lat \
                --max_lon $max_lon \
                --min_lon $min_lon \
                --obj_save_path $obj_save_path \
                --osm_save_path $osm_save_path \
                --mtl_file $mtl_file \
                --png_file $png_file \
                --addonpath $addonpath \
                --asset_save_path $asset_save_path 

    # LLM-empowered 3D asset rendering
    source activate urbanworld
    CUDA_VISIBLE_DEVICES=0 python pipeline_osm.py  \
                            --sd_config $sd_config_1  \
                            --render_config $render_config \
                            --obj_save_path $obj_save_path \
                            --seed $seed \
                            --sd_config2 $sd_config_2 \
                            # --ip_img $ip_img

    #3D asset organization with OSM constraints
    source activate blend
    $blender_path -b -P osm_get_pos.py -- \
                --obj_save_path $obj_save_path \
                --osm_save_path $osm_save_path \
                --max_lat $max_lat \
                --min_lat $min_lat \
                --max_lon $max_lon \
                --min_lon $min_lon 

    
