# --------------------------------------------------------
# Live Dataset
# Written by Jerome Mutgeert
# --------------------------------------------------------

"""

"""
import numpy as np
import time
import os
import threading

from scipy.sparse import csr_matrix

TARGET_DATA_PATH = "./TargetDataLoaderProcess/{}"
PYTHON3 = os.environ['PYTHON3'] #= "/home/jerome/anaconda3/envs/rl/bin/python"

# counter txt's interface:
def update_read(read):
    tempfile = TARGET_DATA_PATH.format("read_temp.txt")
    file = TARGET_DATA_PATH.format("read.txt")
    with open(tempfile,'w') as f:
        f.write(str(read))
        f.flush()
        os.fsync(f.fileno())
    # atomic:
    os.rename(tempfile,file)

def get_fetched():
    with open(TARGET_DATA_PATH.format("fetched.txt"),'r') as f:
        numstr = f.read()
    return int(numstr)

# This is the interface that is used by the code:
def target_file_streamer():
    
    num = 0
    fetched = 0
    read = 0
    
    # Trigger crash of possible previous running data loader process:
    update_read(read)
    time.sleep(.5)
    
    # Make sure we do not read from an old version of fetched.txt: write 0
    with open(TARGET_DATA_PATH.format("fetched.txt"),'w') as f:
        f.write(str(0))
        f.flush()
        os.fsync(f.fileno())
    
    # start data loader process:
    os.chdir(TARGET_DATA_PATH.format(""))
    os.system(" ".join([PYTHON3,"data_loader.py",'&']))
    os.chdir("..")
    
    while True:
        
        # Ensure the file 'target_<num>.jpg" is loaded:
        if not (fetched > num):
            fetched = get_fetched()
        while not (fetched > num):
            time.sleep(.05) #query with 20 Hz untill the file(s) is (are) loaded.
            fetched = get_fetched()
        
        filepath = TARGET_DATA_PATH.format("target_{}.jpg".format(num))
        yield filepath
        
        read += 1
        update_read(read)
        
        num += 1

dummy_gt_overlaps = np.zeros((1,81),dtype=np.float32)
dummy_gt_overlaps[0,1] = 1.
dummy_gt_overlaps = csr_matrix(dummy_gt_overlaps)

def target_roi_gen():
    for filepath in target_file_streamer():
        flipped = np.random.randint(2) == 1
        
        # roi = {'dataset':'live_targets',
        #        'gt_classes': np.array([0], dtype=np.int32),
        #        'flipped': True,
        #        'is_source': np.array([ True]),
        #        'max_classes': np.array([19, 19, 19, 19, 19, 19, 19]),
        #        'image': '/home/jerome/Study/Thesis/Detectron-DA-Faster-RCNN/detectron/datasets/data/coco/val2017/000000295231.jpg',
        #        'max_overlaps': np.array([1., 1., 1., 1., 1., 1., 1.], dtype=np.float32),
        #        'boxes': np.array([[109.98001 , 215.29    , 477.63    , 617.21    ],
        #            [130.12    , 107.26    , 322.1     , 232.13    ],
        #            [ 88.17001 ,  11.92    , 477.44    , 125.4     ],
        #            [  0.      ,  65.92    , 135.26999 , 210.18    ],
        #            [ 21.140015, 160.92    , 167.93    , 320.4     ],
        #            [292.16    ,  83.12    , 479.      , 241.7     ],
        #            [200.32    , 164.03    , 479.      , 546.62    ]], dtype=np.float32),
        #        'coco_url': 'http://images.cocodataset.org/val2017/000000295231.jpg',
        #        'height': 640,
        #        'width': 480,
        #        'is_crowd': np.array([False, False, False, False, False, False, False]),
        #        'bbox_targets': np.array([[19.,  0.,  0.,  0.,  0.],
        #            [19.,  0.,  0.,  0.,  0.],
        #            [19.,  0.,  0.,  0.,  0.],
        #            [19.,  0.,  0.,  0.,  0.],
        #            [19.,  0.,  0.,  0.,  0.],
        #            [19.,  0.,  0.,  0.,  0.],
        #            [19.,  0.,  0.,  0.,  0.]], dtype=np.float32),
        #        'box_to_gt_ind_map': np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32),
        #        'flickr_url': 'http://farm4.staticflickr.com/3056/5868051368_6ace5b29dc_z.jpg',
        #        'seg_areas': np.array([81757.03 , 12373.193, 25199.621, 12078.425, 11397.503, 21294.613, 16879.371], dtype=np.float32),
        #        'has_visible_keypoints': False,
        #        'segms': [[
        #            [310.43, 608.61, 307.69, 579.83, 295.36, 555.16, 300.84000000000003, 515.42, 303.58000000000004, 492.12, 320.03, 485.27, 322.77, 460.6, 315.91999999999996, 451.01, 343.33000000000004, 444.16, 359.77, 426.34, 362.51, 463.34, 359.77, 486.64, 363.88, 497.61, 362.51, 552.42, 357.03, 567.5, 388.55, 574.35, 388.55, 563.39, 383.07, 546.94, 381.7, 488.01, 388.55, 471.57, 380.33, 459.23, 389.92, 409.9, 395.4, 444.16, 418.7, 442.79, 431.03, 435.94, 450.22, 424.97, 447.48, 412.64, 432.4, 415.38, 421.44, 407.16, 429.65999999999997, 398.93, 440.63, 398.93, 451.59, 401.67, 451.59, 394.82, 461.18, 390.71, 473.52, 383.86, 477.63, 361.93, 477.63, 353.71, 465.3, 360.56, 455.7, 372.89, 451.59, 378.38, 448.85, 371.52, 457.07, 353.71, 472.15, 319.45, 440.63, 360.56, 437.89, 349.6, 440.63, 341.37, 429.65999999999997, 334.52, 406.37, 334.52, 406.37, 322.19, 407.74, 311.22, 407.74, 297.52, 400.88, 289.3, 396.77, 278.33, 396.77, 268.74, 388.55, 264.63, 394.03, 257.78, 384.44, 246.81, 369.36, 237.22, 352.92, 230.37, 337.84000000000003, 218.03, 309.06, 215.29, 291.25, 222.15, 265.21000000000004, 220.77, 255.62, 220.77, 237.8, 227.63, 225.47, 241.33, 214.5, 249.55, 210.39, 237.22, 207.64999999999998, 229.0, 192.57999999999998, 224.89, 189.83999999999997, 226.26, 187.08999999999997, 229.0, 172.01999999999998, 233.11, 151.45999999999998, 242.7, 140.5, 252.3, 135.01999999999998, 263.26, 129.54000000000002, 271.48, 115.82999999999998, 290.67, 108.98000000000002, 305.74, 108.98000000000002, 319.45, 110.35000000000002, 341.37, 114.45999999999998, 352.34, 111.72000000000003, 367.41, 118.57, 387.97, 124.05000000000001, 392.08, 119.94, 401.67, 128.16000000000003, 411.27, 144.61, 419.49, 156.94, 434.56, 161.06, 437.31, 177.5, 442.79, 177.5, 457.86, 173.39, 475.68, 172.01999999999998, 479.79, 172.01999999999998, 485.27, 174.76, 489.38, 189.83999999999997, 497.61, 191.20999999999998, 509.94, 204.91000000000003, 519.53, 206.27999999999997, 527.76, 213.13, 542.83, 214.5, 574.35, 206.27999999999997, 596.28, 209.01999999999998, 609.98, 225.47, 615.46, 235.06, 618.21, 233.69, 604.5, 243.28, 597.65, 244.65, 593.54, 230.95, 570.24, 225.47, 537.35, 228.21, 518.16, 246.02, 518.16, 255.62, 515.42, 239.17, 541.46, 236.43, 562.02, 235.06, 574.35, 235.06, 575.72, 236.43, 575.72, 247.39, 581.2, 259.73, 575.72, 263.84000000000003, 567.5, 262.47, 546.94, 261.1, 525.01, 262.47, 518.16, 270.69, 523.64, 281.65999999999997, 525.01, 278.90999999999997, 537.35, 281.65999999999997, 552.42, 283.03, 575.72, 283.03, 592.17, 283.03, 607.24, 289.88, 607.24, 307.69, 601.76]], [[319.35, 108.72, 296.90999999999997, 107.26, 273.5, 109.7, 261.78999999999996, 114.09, 253.01, 117.99, 249.11, 118.97, 240.82, 121.41, 234.96, 120.92, 231.06, 116.04, 215.45, 111.65, 207.16000000000003, 109.7, 192.52999999999997, 111.65, 181.31, 115.06, 173.51, 120.92, 163.75, 123.36, 155.95, 120.92, 144.24, 121.41, 129.12, 127.26, 130.57999999999998, 129.7, 137.41000000000003, 137.01, 150.08999999999997, 143.84, 161.8, 144.82, 166.68, 144.82, 169.12, 144.33, 172.52999999999997, 145.79, 170.08999999999997, 151.65, 168.63, 154.09, 167.17000000000002, 156.04, 166.68, 163.35, 167.64999999999998, 173.11, 169.12, 184.33, 163.26, 208.74, 163.26, 215.09, 165.7, 224.35, 168.14, 233.13, 200.32999999999998, 226.3, 211.55, 218.5, 219.36, 212.16, 223.26, 201.92, 230.57, 192.16, 235.94, 188.75, 238.87, 186.31, 239.84, 171.67, 243.26, 166.8, 249.11, 164.85, 252.52, 173.14, 256.90999999999997, 178.99, 262.28, 181.43, 272.03999999999996, 182.89, 281.3, 183.38, 288.62, 179.97, 316.91999999999996, 154.27, 321.36, 145.39, 320.62, 144.65, 302.14, 137.26, 293.27, 131.35, 291.78999999999996, 129.87, 291.78999999999996, 121.0, 291.78999999999996, 117.3, 299.91999999999996, 113.6, 309.53, 111.39, 316.91999999999996, 110.65, 322.1, 110.65]], [[348.75, 41.27, 314.38, 41.55, 243.11, 33.6, 207.04000000000002, 35.31, 195.39999999999998, 37.01, 194.89999999999998, 26.35, 192.08999999999997, 24.66, 176.16000000000003, 25.6, 157.62, 22.98, 146.56, 26.16, 139.82, 28.97, 131.95, 31.78, 115.64999999999998, 27.1, 106.66000000000003, 22.79, 95.60000000000002, 17.54, 88.11000000000001, 11.92, 87.17000000000002, 14.36, 91.67000000000002, 21.85, 101.41000000000003, 31.41, 107.41000000000003, 36.09, 114.52999999999997, 41.71, 114.14999999999998, 61.2, 117.14999999999998, 77.69, 116.01999999999998, 100.92, 122.76999999999998, 108.6, 131.01, 112.53, 138.69, 113.28, 146.94, 111.78, 154.24, 121.9, 157.62, 123.4, 165.86, 123.96, 185.91000000000003, 123.77, 183.66000000000003, 120.96, 183.10000000000002, 114.97, 190.02999999999997, 112.35, 198.45999999999998, 112.53, 205.95, 116.28, 210.07999999999998, 111.6, 225.25, 114.41, 234.81, 122.84, 238.74, 126.4, 267.40999999999997, 114.03, 295.88, 108.04, 341.03999999999996, 109.35, 383.44, 107.1, 420.93, 89.78, 451.03, 86.94, 477.44, 80.69, 475.17, 72.74, 453.3, 53.15, 430.58, 44.35, 410.14, 42.07, 387.42, 44.06, 370.65999999999997, 42.93]], [[119.44999999999999, 73.11, 90.69, 74.55, 69.11000000000001, 77.42, 51.85000000000002, 65.92, 48.98000000000002, 75.98, 59.04000000000002, 84.61, 71.99000000000001, 90.37, 64.80000000000001, 99.0, 51.85000000000002, 97.56, 11.579999999999984, 91.81, -1.0, 90.37, 0.07999999999998408, 130.64, -1.0, 195.36, -1.0, 206.86, 1.5199999999999818, 211.18, 13.019999999999982, 209.74, 44.660000000000025, 170.91, 67.67000000000002, 170.91, 87.81, 166.59, 135.26999999999998, 166.59]], [[20.139999999999986, 232.63, 26.970000000000027, 244.33, 40.139999999999986, 245.8, 75.25999999999999, 232.63, 68.43, 245.31, 69.39999999999998, 256.53, 72.32999999999998, 258.97, 86.95999999999998, 303.84, 109.39999999999998, 321.4, 112.81, 299.45, 140.13, 252.14, 167.93, 233.6, 162.07999999999998, 213.6, 156.22000000000003, 218.48, 142.07999999999998, 200.43, 104.02999999999997, 167.27, 102.07999999999998, 167.27, 95.25, 160.92, 71.35000000000002, 168.24, 56.23000000000002, 174.09, 45.99000000000001, 191.65, 25.00999999999999, 213.12]], [[479.0, 229.4, 390.9, 242.7, 370.95, 234.39, 352.65999999999997, 219.43, 321.08000000000004, 214.44, 326.06, 189.51, 327.73, 134.65, 309.44, 134.65, 291.15999999999997, 116.36, 321.08000000000004, 111.38, 347.68, 114.7, 379.26, 101.4, 417.49, 86.44, 472.35, 83.12, 477.34, 83.12]], [[478.16, 238.29, 420.63, 239.76, 391.12, 242.71, 345.38, 222.06, 388.17, 254.52, 391.12, 258.94, 389.64, 267.79, 397.02, 267.79, 401.45, 289.93, 408.82, 298.78, 407.35, 332.71, 422.1, 335.66, 429.48, 337.14, 436.85, 350.41, 441.28, 362.22, 472.26, 319.43, 448.66, 368.12, 448.66, 376.97, 453.08, 376.97, 458.98, 365.17, 479.0, 354.35, 476.69, 239.27], [348.33000000000004, 538.77, 333.58000000000004, 538.77, 324.73, 531.39, 324.73, 524.02, 338.01, 503.36, 345.38, 488.61, 345.38, 478.28, 346.86, 470.9, 349.81, 459.1, 342.43, 442.87, 360.14, 425.17, 363.09000000000003, 470.9, 357.19, 482.71, 364.56, 494.51, 363.09000000000003, 509.26, 363.09000000000003, 529.92, 357.19, 529.92, 352.76, 532.87], [389.64, 423.69, 391.12, 469.43, 397.02, 516.64, 398.5, 525.49, 405.87, 540.24, 401.45, 544.67, 394.07, 547.62, 382.27, 546.15, 379.32, 488.61, 386.69, 470.9, 382.27, 460.58, 388.17, 428.12], [323.25, 164.03, 298.16999999999996, 195.01, 314.4, 215.67, 293.75, 223.05, 252.44, 221.57, 228.83, 233.37, 215.55, 249.6, 211.13, 246.65, 208.18, 231.9, 199.32, 226.0, 218.5, 209.77, 221.45, 199.44, 231.78, 184.69, 237.68, 183.21, 237.68, 169.93, 243.58, 164.03, 248.01, 164.03, 258.34000000000003, 181.74, 284.89, 181.74, 295.22, 172.88, 320.3, 166.98]]],
        #        'id': -1,
        #        'gt_overlaps': np.array([1.],dtype=np.float32) }
        
        
        
        roi = {'image': filepath,
               'flipped': flipped,
               'is_source': np.array([ False ]),
               'dataset':'live_targets',
               'gt_classes': np.array([1], dtype=np.int32),
               'max_classes': np.array([1]),
               'max_overlaps': np.array([1.], dtype=np.float32),
               'boxes': np.array([[0.,0.,0.,0.]], dtype=np.float32),
               # 'coco_url': 'http://images.cocodataset.org/val2017/000000295231.jpg',
               'height': 1,
               'width': 1,
               'is_crowd': np.array([False]),
               'bbox_targets': np.array([[1.,0.,0.,0.,0.]], dtype=np.float32),
               'box_to_gt_ind_map': np.array([0], dtype=np.int32),
               # 'flickr_url': 'http://farm4.staticflickr.com/3056/5868051368_6ace5b29dc_z.jpg',
               'seg_areas': np.array([1.], dtype=np.float32),
               'has_visible_keypoints': False,
               'segms': [[[1.]]],
               'id': -1,
               'gt_overlaps': dummy_gt_overlaps }
        
        # roi = {
        #     'gt_classes': np.array([1], dtype=np.int32),
        #     'max_classes': np.array([1]),
        #     'bbox_targets': np.array([[3.,0.,0.,0.,0.]], dtype=np.float32),
        #     'boxes': np.array([[0,0,0,0]], dtype=np.uint16),
        #     'max_overlaps': np.array([1.], dtype=np.float32),
        #     'gt_overlaps': np.array([1.],dtype=np.float32),
        #     'image' : filepath,
        #     'width': 0,
        #     'height': 0,
        #     'flipped': flipped
        # }
        yield roi


# we will create a list wrapper
class LiveRoidb(list):
    def __init__(self):
        super(LiveRoidb,self).__init__()
        self.generator = target_roi_gen()
        self.lock = threading.Lock()
        
    def __getitem__(self, i):
        """ignores i"""
        with self.lock:
            roi = next(self.generator)
        return roi
    
    def __len__(self):
        return 10  # it is lying, the truth is 0.
    
    def __iter__(self):
        raise NotImplementedError()


def dl_by_id(im_id):
    import urllib2
    import cv2
    import numpy as np
    
    url_base = "https://test.yisual.com/images/media/download/picturethis/"
    headers = {"api-key": "ccea03e0e3a08c428870393376e5cf7b7be7a55c", "api-secret": os.environ["SECRET"]}
    url = url_base + im_id
    req = urllib2.Request(url, headers=headers)
    connection = urllib2.urlopen(req)
    jpeg_str = connection.read()
    connection.close()
    img = np.frombuffer(jpeg_str, np.uint8)
    im = cv2.imdecode(img,1)
    return im

if __name__ == '__main__':
    roidb = LiveRoidb()
    for i in range(10):
        print(roidb[100])
    print(len(roidb))
    for roi in roidb:
        print("roi:",roi)
    roidb = [x for x in roidb]
    
    print(len(roidb))
    print(roidb[100])