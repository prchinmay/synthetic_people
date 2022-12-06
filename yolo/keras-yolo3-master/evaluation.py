import pandas as pd
import glob

from cfg import *

test_folder = base_path + 'test1/'
true_folder = true_path
overlap_threshold = 0.5
beta = 2


def intersection_area(A, B):
    bbox_A = list(map(int, A.split(',')[:-1]))
    bbox_B = list(map(int, B.split(',')[:-1]))
    x1, y1 = max(bbox_A[0], bbox_B[0]), max(bbox_A[1], bbox_B[1])
    x2, y2 = min(bbox_A[2], bbox_B[2]), min(bbox_A[3], bbox_B[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    AoI = (x2 - x1) * (y2 - y1)
    return AoI
    
def union_area(A,B):
    bbox_A = list(map(int, A.split(',')[:-1]))
    bbox_B = list(map(int, B.split(',')[:-1]))
    area_A = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
    area_B = (bbox_B[2] - bbox_B[0]) * (bbox_B[3] - bbox_B[1])
    AoI = intersection_area(A, B)
    AoU = area_A + area_B - AoI
    return AoU
    
def calculate_scores(list_true, list_test):
    list_true = list_true.copy()
    list_test = list_test.copy()

    list_scores = {'AoI': [], 'AoU': [], 'IoU': []}
    if len(list_test)>0 and len(list_true)==0:      # FP ONLY
        for bbox_test in list_test:
            list_scores['AoI'].append(-1)
            list_scores['AoU'].append(-1)
            list_scores['IoU'].append(-1)
            return list_scores
    if len(list_test)==0 and len(list_true)>0:      # FN ONLY
        for bbox_true in list_true:
            list_scores['AoI'].append(0)
            list_scores['AoU'].append(0)
            list_scores['IoU'].append(0)
            return list_scores
    else:  
        for bbox_true in list_true:
            df_temp = pd.DataFrame()                # TP + FN
            if len(list_test)==0:
                list_scores['AoI'].append(0)
                list_scores['AoU'].append(0)
                list_scores['IoU'].append(0)
                continue
            for bbox_test in list_test:
                AoI = intersection_area(bbox_true, bbox_test)
                AoU = union_area(bbox_true, bbox_test)
                IoU = round(AoI / AoU, 4)
                temp = {'bbox_true': bbox_true, 'bbox_test': bbox_test, 'AoI': AoI, 'AoU': AoU, 'IoU': IoU}
                df_temp = df_temp.append(temp, ignore_index=True)
            temp_best = df_temp.loc[df_temp['IoU'] == df_temp['IoU'].max(), :]
            list_test.remove(temp_best['bbox_test'].values)
            list_scores['AoI'].append(temp_best['AoI'].values[0])
            list_scores['AoU'].append(temp_best['AoU'].values[0])
            list_scores['IoU'].append(temp_best['IoU'].values[0])
        if len(list_test) > 0:                      # TP + FP
            for bbox_test in list_test:
                list_scores['AoI'].append(-1)
                list_scores['AoU'].append(-1)
                list_scores['IoU'].append(-1)
            
    return list_scores
    
if __name__ == '__main__':
    # OBTAIN LIST OF BOUNDING BOXES
    list_test = glob.glob(test_folder + '*.txt')
    list_true = glob.glob(true_folder + '*.txt')
    
    # SELECT ONLY THE RELEVANT TRUE BOUNDING BOX >> LEN(LIST_TEST) == LEN(LIST_TRUE)
    true_test = [x.split('/')[-1] for x in list_test]
    list_true = [x for x in list_true if x.split('/')[-1] in true_test]
    
    # EXTRACT TEST BOUNDING BOX
    df_test = pd.DataFrame()
    for path in list_test: 
        with open(path) as file:
            line = file.readline()
        if line.split(' ')[-1]:
            bbox = [box.rstrip('\n') for box in line.split(' ')[1:] if 'person' in box]
        else:
            bbox = [['0,0,0,0,0']]
        temp = {'IMG': line.split(' ')[0].split('/')[-1].strip(), 'bbox_test': bbox}
        df_test = df_test.append(temp, ignore_index=True)
    
    # EXTRACT TRUE BOUNDING BOX
    df_true = pd.DataFrame()
    for path in list_true:
        with open(path) as file:
            line = file.readline()
        if line.split(' ')[1:][-1].rstrip('\n'):
            bbox = [box.rstrip('\n') for box in line.split(' ')[1:] if box.rstrip('\n')[-1] == '0']
        else:
            bbox = [['0,0,0,0,0']]
        temp = {'IMG': line.split(' ')[0].split('/')[-1].strip(), 'bbox_true': bbox}
        df_true = df_true.append(temp, ignore_index=True)
    
    df_output = df_true.merge(df_test, on='IMG', how='left')
    
    # EVALUATE TEST BOUNDING BOX VS TRUE BOUNDING BOX
    df_scores = pd.DataFrame()
    for i in range(len(df_output)):
        scores = calculate_scores(df_output.loc[i, 'bbox_true'], df_output.loc[i, 'bbox_test'])
        temp = pd.DataFrame({'AoI': [scores['AoI']], 'AoU': [scores['AoU']], 'IoU': [scores['IoU']]})
        df_scores = df_scores.append(temp, ignore_index=True)
    
    df_output = df_output.merge(df_scores, how='left', left_index=True, right_index=True)
    
    # CALCULATE TRUE POSITIVE, FALSE POSITIVE, AND FALSE NEGATIVE
    df_output['TP'] = df_output['IoU'].apply(lambda x: len(list(filter(lambda y: (y >= overlap_threshold), x))))
    df_output['FP'] = df_output['IoU'].apply(lambda x: len(list(filter(lambda y: (y == -1), x))))
    df_output['FN'] = df_output['IoU'].apply(lambda x: len(list(filter(lambda y: (abs(y) < overlap_threshold), x))))
    df_output.to_csv('evaluation1.csv', index=False)
    
    # SUMMARY AND SCORE
    df_metrics = pd.DataFrame()
    df_metrics['Precision'] = [df_output['TP'].sum() / (df_output['TP'].sum() + df_output['FP'].sum())]
    df_metrics['Recall'] = [df_output['TP'].sum() / (df_output['TP'].sum() + df_output['FN'].sum())]
    df_metrics['Accuracy'] = [df_output['TP'].sum() / (df_output['TP'].sum() + df_output['FP'].sum() + df_output['FN'].sum())]
    df_metrics['F1_score'] = [2*df_metrics['Precision'][0]*df_metrics['Recall'][0] / (df_metrics['Precision'][0] + df_metrics['Recall'][0])]
    df_metrics['F{}_score'.format(beta)] = [(1 + beta**2)*df_metrics['Precision'][0]*df_metrics['Recall'][0] / ((beta**2)*df_metrics['Precision'][0] + df_metrics['Recall'][0])]
    df_metrics.to_csv('evaluation1_summary.csv', index=False)
    
    print(df_metrics)
    