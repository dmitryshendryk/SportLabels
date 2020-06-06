import os 
import shutil


# ROOT = os.path.abspath('./')

# list_dirs = os.listdir('./data/SAMPLEDATA/ADIDAS')

# img_iter = {'size':0,'box':0}
# for folder in list_dirs:
#     if not folder.startswith('.'):
#         for img_name in os.listdir(os.path.join(ROOT,'data/SAMPLEDATA/ADIDAS', folder)):
#             if not img_name.startswith('.'):
#                 src_path = os.path.join(ROOT,'data/SAMPLEDATA/ADIDAS', folder, img_name)
#                 img_cls = img_name.split('.')[0]
#                 if img_cls in ['size','box']:
#                     dist_path = os.path.join(ROOT,'data/train',img_cls, str(img_iter[img_cls]) + '.jpg' )
#                     img_iter[img_cls] += 1
#                     shutil.copy(src_path,dist_path)



def prepare_dataset():
    ROOT = os.path.abspath('./')
    iter_number = 0
    list_dirs = os.listdir(os.path.join(ROOT, 'data/SAMPLEDATA/ADIDAS'))

    for folder in list_dirs:
        if not folder.startswith('.'):
            for img_name in os.listdir(os.path.join(ROOT,'data/SAMPLEDATA/ADIDAS', folder)):
                if not img_name.startswith('.'):
                    src_path = os.path.join(ROOT,'data/SAMPLEDATA/ADIDAS', folder, img_name)
                    dist_path = os.path.join(ROOT,'data/ssd', str(iter_number) + '.jpg' )
                    shutil.copy(src_path,dist_path)
                    iter_number += 1 

prepare_dataset()