# INITIALIZE BASE PATH (= MODEL FOLDER)
base_path = '/content/drive/MyDrive/yolo/keras-yolo3-master/'

# INITIALIZE CLASSES & ANCHORS
classes_path = 'Model/synthetic_classes.txt'
#classes_path = 'Model/coco_classes.txt'
anchors_path = 'Model/yolo_anchors.txt'

# TRAINING MODEL   << TRAINING PATHS ONLY
annot_path = '/content/mpii_finetune_500.txt'
logs_path = '/content/drive/MyDrive/yolo/model_surreal/Train/logs/bsl_9k_single_fin500/'
input_weights = 'Train/logs/bsl_9k/ep033-loss6.017-val_loss6.041.h5'




# PERFORM YOLOV3 MODEL  << TESTING PATHS
model_name = 'model_surreal'  ## Not that important, just naming your model in the excel file
model_path = 'Train/logs/bsl_25k_single_fin500/ep018-loss16.369-val_loss14.819.h5'
true_path = '/content/mpii_test_500/'
output_path = '/content/output_single_25k_500_ep18/'


param_lr = 1e-4