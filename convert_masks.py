import argparse
import numpy as np
import os
import csv
import cv2
# -

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cell', help='convert mode:cell/tissue')
    parser.add_argument('--cell_radius', type=int, default=5, help='radius of cells')

    parser.add_argument('--save_path', type=str, default='pred_mask', help='path to save mask')
    parser.add_argument('--data_path', nargs='+', type=str, default='../DFUC2022_val' , help='path to testing data')
    
    return parser.parse_args()

def get_ground_truth_heatmap(annotations, num_classes, output_shape, image_shape):
  heatmap = np.zeros((num_classes, output_shape[0], output_shape[1]), dtype=np.float32)

  for ann in annotations:
    x_center, y_center, cls_id = ann[0], ann[1], ann[2]

    x_center = int(x_center * output_shape[1] / image_shape[0])  # 將座標映射到輸出 heatmap 的尺寸上
    y_center = int(y_center * output_shape[0] / image_shape[1])

    # heatmap[cls_id, y_center, x_center] = 255.0  # 在對應的位置上將該類別的值設為 x

    radius = 5
    for i in range(max(0, y_center - radius), min(output_shape[0], y_center + radius + 1)):
      for j in range(max(0, x_center - radius), min(output_shape[1], x_center + radius + 1)):
        if ((j - x_center) ** 2 + (i - y_center) ** 2) <= radius ** 2:
          heatmap[cls_id, i, j] = 255.0  # 在半徑範圍內將該類別的值設為 x

  return heatmap

if __name__ == '__main__':
    opt = arg_parser()
    
    if opt.mode == 'cell':
        num_class = 3
        trainsize = (1024,1024)
        image_shape = (1024,1024)
        
        csv_folder = opt.data_path  # 包含CSV檔案的資料夾路徑
        csv_names = sorted(os.listdir(csv_folder))
        save_dir = opt.save_path
        os.makedirs(save_dir, exist_ok=True)
        for csv_name in csv_names:
            if csv_name.endswith('.csv'):
                csv_file = os.path.join(csv_folder, csv_name)
                gt_ann = []

                with open(csv_file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for row in csvreader:
                        x, y, class_label = float(row[0]), float(row[1]), int(row[2])
                        gt_ann.append([x, y, class_label])

                heatmap = get_ground_truth_heatmap(gt_ann, num_class, trainsize, image_shape)

                file_name = csv_name.split('.')[0]

                # 分別儲存三個類別的熱圖
                for cls in range(3):
                    class_heatmap = heatmap[cls]
                    save_path = save_dir + f"{file_name}_{cls}.jpg"
                    print(save_path)
                    cv2.imwrite(save_path, class_heatmap)  
    else:# tissue
        mask_directory = opt.data_path
        save_dir = opt.save_path
        os.makedirs(save_dir, exist_ok=True)

        # 讀取每個 mask 檔案並保存對應的熱圖
        for filename in sorted(os.listdir(mask_directory)):
            if filename.endswith('.png'):
                print(filename)
                mask_path = os.path.join(mask_directory, filename)
                # 读取分割遮罩图像
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # 创建三个空白的类别遮罩图像
                class0_mask = np.zeros_like(mask)
                class1_mask = np.zeros_like(mask)
                class2_mask = np.zeros_like(mask)

                # 遍历每个像素，并根据像素值存储到相应的类别遮罩图像中
                for row in range(mask.shape[0]):
                    for col in range(mask.shape[1]):
                        pixel_value = mask[row, col]
                        if pixel_value == 255:
                            class0_mask[row, col] = 255
                        elif pixel_value == 1:
                            class1_mask[row, col] = 255
                        elif pixel_value == 2:
                            class2_mask[row, col] = 255

                # 提取檔案名稱，用於保存圖像
                file_name = os.path.splitext(filename)[0]

                # 保存类别遮罩图像
                save_path = save_dir + file_name
                cv2.imwrite(save_path + '_0.jpg', class0_mask)
                cv2.imwrite(save_path + '_1.jpg', class1_mask)
                cv2.imwrite(save_path + '_2.jpg', class2_mask)

        
    
    