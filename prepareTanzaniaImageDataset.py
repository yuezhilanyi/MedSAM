# %%
import os 
import glob 
import rasterio 

import geopandas as gpd 

from rasterio import features
from rasterio.windows import Window

# %% shapeFile to tiff 
def rasterize_shapefile(
        rst_fn, 
        shapefile_path, 
        out_fn, 
    ):
    # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    # @Michael Lindgren, @wfgeo
    # open reference raster file 
    # 打开矢量数据
    df = gpd.read_file(shapefile_path)
    df['value'] = 0
    dct = {'Complete':1, 'Incomplete':2, 'Foundation': 3}
    for k, v in dct.items():
        df.loc[df['condition']==k, 'value'] = v
    
    print(df.value.max())

    rst = rasterio.open(rst_fn)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    meta.update(count=1)
    # meta.update(crs='EPSG:4326')
    # meta.update(transform=rasterio.transform.from_bounds(*df.total_bounds, meta['width'], meta['height']))
    df = df.to_crs(rst.crs.to_dict()) 

    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(df.geometry, df.value))

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        print(burned.max())
        out.write_band(1, burned)

# %%
# 示例使用
data_root = r"E:\datasets\2018_Open_AI_Tanzania_Building_Footprint_Segmentation_Challenge"
shapefile_paths = sorted(glob.glob(os.path.join(data_root, "**/*.shp"), recursive=True))
rst_fns = [shapefile_path.replace('shp', 'tif') for shapefile_path in shapefile_paths] 
out_fns = []
for rst_fn in rst_fns:
    head, tail = os.path.split(rst_fn)
    out_fns.append(os.path.join(head, 'label/' + tail))

for rst_fn, shapefile_path, out_fn in zip(rst_fns, shapefile_paths, out_fns):
    rst_fn = os.path.join(data_root, rst_fn)
    shapefile_path = os.path.join(data_root, shapefile_path)
    out_fn = os.path.join(data_root, out_fn)
    print(rst_fn, shapefile_path, out_fn)
    rasterize_shapefile(rst_fn, shapefile_path, out_fn)
    # break


# %% prepare Tanzania dataset
def prepareTanzaniaImageDataset(y_size=1024, x_size=1024, total_numbers=10000):
    # find all train data
    data_root = r"E:\datasets\2018_Open_AI_Tanzania_Building_Footprint_Segmentation_Challenge"
    gt_path_files = sorted(glob.glob(os.path.join(data_root, "*label\\*.tif"), recursive=True))
    img_path_files = [gt_path.replace('label\\', '') for gt_path in gt_path_files]
    assert len(gt_path_files) == len(img_path_files)
    print(img_path_files)
    # generate rois (1024x1024), (row_index, image_index, )
    with open("dataset.txt", 'w') as fp:
        for img_path, gt_path in zip(img_path_files, gt_path_files):
            rst = rasterio.open(img_path)
            height = rst.meta['height']
            width = rst.meta['width']
            for y in range(0, height, y_size):
                if y + y_size >= height:
                    continue
                for x in range(0, width, x_size):
                    if x + x_size >= width:
                        continue 
                    # image = rst.read(window=Window(x, y, x_size, y_size))
                    # c, h, w = image.shape
                    # assert(h!=0 and w!=0), (img_path, c, h, w, y, x, height, width)
                    fp.write(img_path + ',' + gt_path + ',' + str(y) + ',' + str(x) + '\n')

prepareTanzaniaImageDataset()
