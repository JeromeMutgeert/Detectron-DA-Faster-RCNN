
import numpy as np
from detectron.core.config import cfg

# import matplotlib
# import matplotlib.pyplot as plt
from tools.analyse_detects import coco_classes, KL_div

def print_dist(dist,name):
    order = np.argsort(dist)[::-1]
    dist = dist[order]
    classes = coco_classes[order]
    print(name)
    for w,c in list(zip(dist,classes))[:]:
        print("p = ({:.3f}):{}".format(w,c))


class ClassWeightDB(object):
    
    def __init__(self,weight_db=None):
        self.weight_db = weight_db
        self.total_sum_softmax = None
        self.class_weights = None
        self.gt_ins_dist = None
        self.starting_dist = None
        self.avg_pada_weight = 0
    
    def setup(self,roi_data_loader):
        source_roidb = roi_data_loader._roidb
        target_roidb = roi_data_loader._target_roidb
        continuing = self.weight_db is not None
        if continuing:
            print("weight_db taken from checkpoint")
        gt_ins_counts = np.bincount([cls_idx
                                     for rois in source_roidb
                                     for cls_idx, crowd in zip(rois['gt_classes'],rois['is_crowd'])
                                     if not crowd],
                                    minlength=cfg.MODEL.NUM_CLASSES)
        n_instances = gt_ins_counts.sum()
        self.gt_ins_dist = gt_ins_counts/float(n_instances) + np.finfo(float).eps
        print_dist(self.gt_ins_dist,'gt_ins_dist')
        
        if not continuing:
            self.weight_db = np.concatenate([rois['sum_softmax'][None,:] for rois in target_roidb],axis=0)
        else:
            self.starting_dist = np.sum([rois['sum_softmax'] for rois in target_roidb],axis=0)
            self.starting_dist /= self.starting_dist.sum()
        print("ClassWeightDB initiated with weight db of shape {}".format(self.weight_db.shape))
        self.total_sum_softmax = self.weight_db.sum(axis=0)
        self.class_weights = self.total_sum_softmax / self.total_sum_softmax.max()
        # w = self.class_weights
        # print('absolute weights: mean,min,max,median:',w.mean(),w.min(),w.max(),np.median(w))
        if not continuing:
            self.starting_dist = self.class_weights/self.class_weights.sum()
        # print_dist(self.starting_dist,name='starting_dist')
        
        self.class_weights = self.class_weights / self.gt_ins_dist
        self.class_weights /= self.class_weights.max()
        w = self.class_weights
        print('pada weights: mean,min,max,median:',w.mean(),w.min(),w.max(),np.median(w))
        print_dist(self.class_weights,'Corrected pada weights')
        
        avg_pada_weight = (self.class_weights * self.gt_ins_dist).sum()
        print("Weighted avg pada weight (by gt dist):", avg_pada_weight)
        # self.avg_pada_stats = RollingAvg(n_instances,avg_resume=avg_pada_weight)
        self.avg_pada_weight = avg_pada_weight
    
    def update_class_weights(self,im_idx,sum_softmax):
        prev_sum_softmax = self.weight_db[im_idx].copy()
        self.weight_db[im_idx] = sum_softmax
        print('NormalizedMeanSquaredUpdate:',((prev_sum_softmax - sum_softmax)**2).mean()/prev_sum_softmax.sum(),prev_sum_softmax.sum(),sum_softmax.sum(),im_idx)
        self.total_sum_softmax += sum_softmax - prev_sum_softmax
        self.class_weights = self.total_sum_softmax / self.gt_ins_dist
        self.class_weights /= self.class_weights.max()
        self.avg_pada_weight = (self.class_weights * self.gt_ins_dist).sum()
        
    def update_get_avg_pada_weight(self,observed_fg_weight,count=1):
        return self.avg_pada_weight
        # return self.avg_pada_stats.update_and_get(observed_fg_weight,weight=count)
    
    def get_avg_pada_weight(self):
        return self.avg_pada_weight
        # return self.avg_pada_stats.get()
    
    def get_dist(self):
        current_dist = self.class_weights * self.gt_ins_dist
        current_dist /= current_dist.sum()
        return current_dist
    
    def get_KL_to_init(self):
        return KL_div(self.get_dist(),self.starting_dist)
    
    def get_state(self):
        return self.weight_db
        
        
class RollingAvg(object):
    def __init__(self,max_sample_size,avg_resume=None,n_resume=None):
        self.n = 0
        self.max_n = max_sample_size
        self.sum = 0.0
        self.avg = 0.0
        if avg_resume is not None:
            if n_resume is not None:
                self.n = n_resume
            else:
                self.n = self.max_n
            self.sum = avg_resume * self.n
        if self.n != 0:
            self.avg = self.sum / self.n
    
    def update_and_get(self,sample,weight = 1):
        if (self.n + weight) < self.max_n:
            self.n += weight
            self.sum += sample * weight
        elif self.n < self.max_n:
            diff = (self.max_n - self.n)
            self.sum += sample * diff
            self.n = self.max_n
            weight = weight - diff
        if self.n == self.max_n:
            self.sum = self.sum * (self.n - weight) / self.n + sample * weight
        self.avg = self.sum / self.n
        return self.avg
    
    def get(self):
        return self.avg
    
class DAScaleFading(object):
    """Fading-in the adversarial objective according the way of DANN:
    http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf
    The formula for the weight given the progression p from 0 to 1 is:
    2 / (1 + exp(- gamma * p) -1
    where gamma is chosen by the autors as 10 and kept fixed across experiments."""
    
    def __init__(self,max_iter,gamma=10.0):
        self.max_iter = float(max_iter)
        self.gamma = float(gamma)
        self.it = 0
        
    def set_iter(self,it):
        self.it = float(it)
        
    def get_weight(self):
        return 2 / (1 + np.exp(-self.gamma * self.it / self.max_iter)) - 1