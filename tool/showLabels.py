import os 
import numpy as np

cls_labels_dict = np.load('../voc12/cls_labels.npy', allow_pickle=True)
print(cls_labels_dict)

cls_labels_dict = np.load('../voc12/cls_labels.npy', allow_pickle=True).item()

# print('2008_004482',cls_labels_dict['2008_004482'])
# print('2011_000053',cls_labels_dict['2011_000053'])
# '2010_002152'
print('2008_005035',cls_labels_dict['2008_005035'])
print('2008_000899',cls_labels_dict['2008_000899'])

# '2010_002552', '2008_002115' '2008_008500', '2009_002883'

# list1 = [i for i in range(20) if label1[i] == 0 and label2[i] == 0]

# list1 = []
# for i in range(20):
#     if label1[i] == 0 and label2[i] == 0:
#         list1.append(i)

# name_pair[0] 2008_005035 label1:  
# tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
#         0., 0., 0., 0., 1., 0., 0., 1., 0., 0.])
# name_pair[1] 2008_000899 label2:  
# tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
#         0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

# label_idx**:  
# [14, 17, 13, 16, 6, 8, 4, 10, 5, 1, 
#  11, 15, 0, 12, 18, 19, 3, 7, 2, 9]

# label x:  tensor(
# [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
#  1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])