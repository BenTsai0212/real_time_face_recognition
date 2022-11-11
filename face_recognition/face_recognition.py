'''
    goal: 使用 FaceNet pretrained model 完成人臉辨識
    process:
        1. 使用 FaceNet model 將 ./image 路徑下的 image 取得其 embedding_vector
        2. 將各 label 對應各 emb_vector 的 dictionary 存入 emb_vector.db (sqlite3)
        3. 新的圖片 input 後使用 model 取得 emb_vector, 用歐氏距離計算與新的圖片最近的 label,
           作為 predict 值 output
    TODO:
        1. 現在還沒研究出如何在 pycharm 中顯示 image, 這樣 _test() 驗證不太直觀

    update: 2022/11/8
'''

import glob
import os
from PIL import Image
import numpy as np
# from IPython.display import display
import configparser
import warnings
from facenet_pytorch import MTCNN, InceptionResnetV1
from sqlitedict import SqliteDict
import torch
import torchvision.transforms as transforms

db_name = './face_recognition/emb_vector.db'
config_pth = './face_recognition/config.ini'
config = configparser.ConfigParser(allow_no_value=True)
config.read(config_pth)

warnings.simplefilter(action='ignore', category=FutureWarning)

class Face_recognition:
    def __init__(self):
        self.config = config
        self.image_pth = self.config.get('directory', 'image_pth')
        self.test_pth = self.config.get('directory', 'test_pth')
        self.test_mode = True
        self.is_cropped = False    # 如果 input image 已經經過 mtcnn 處理過, 在 facenet 這邊就不用再 crop 一次

        self.mtcnn = MTCNN(image_size=224, margin=70)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.distance_norm = 2.2    # 計算 score 的標準值, 視情形可調整 (1.0~2.5)
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])
        self.db = SqliteDict(db_name)

        key_list, item_list = [], []
        for key, item in self.db.items():
            key_list.append(key)
            item_list.append(item)

        self.emb_vector = dict(zip(key_list, item_list))    # 所有人員臉的 embedding vector

    def emb_init(self):
        '''
            取得 image_path 下所有人員臉部的 embedding vector\n
            把結果存進 sqlite3 (./emb_vector.db)
        '''

        emb_dict = {}
        for file_pth in glob.glob(os.path.join(self.image_pth, '*')):
            label = file_pth.split('/')[-1]
            emb_ls = []
            for target_pth in glob.glob(os.path.join(file_pth, '*[.jpg-.jpeg]')):
                img = Image.open(target_pth)
                img_cropped = self.mtcnn(img)
                if img_cropped != None:
                    img_emb = self.model(img_cropped.unsqueeze(0))
                    emb_ls.append(img_emb.detach().numpy()[0])
                else:
                    emb_ls.append('_')
            emb_dict[label] = emb_ls

        self.emb_vector = emb_dict

        #### 將 emb_dict 存入 emb_vector.db (sqlite3)
        if self.test_mode == False:
            for key in emb_dict.keys():
                self.db[key] = emb_dict[key]
            self.db.commit()

    def predict(self, img, is_cropped = False):
        '''
        辨識圖片 label\n
        :param img: 圖片, PIL Image 格式
        :return: pred_label, min_distance, score
        '''
        self.is_cropped = is_cropped

        result = self._img_simiarity_evaluate(img)
        if result == None: return None
        pred_label, min_distance, score = result
        return pred_label, min_distance, score

    def predict_by_path(self, path, is_cropped = False):
        '''
        辨識圖片 label\n
        :param path: 圖片路徑
        :return: pred_label, min_distance, score
        '''
        self.is_cropped = is_cropped

        result = self._img_simiarity_evaluate_by_path(path)
        if result == None: return None
        pred_label, min_distance, score = result
        return pred_label, min_distance, score

    def _img_simiarity_evaluate(self, img):
        '''
        透過圖片與各 label 之距離, 回傳最近似之 label\n
        圖片為 PIL.Image 格式\n
        :param path: 圖片的路徑
        :return: pred_label, min_distance, score
        '''
        distance_dict = self._emb_distance_evaluate(img)

        if distance_dict == None: return None

        #### 取得距離最小之 label
        label_keys = list(distance_dict.keys())
        min_distance = min(list(distance_dict.values()))
        min_idx = list(distance_dict.values()).index(min_distance)
        pred_label = label_keys[min_idx]

        score = ((self.distance_norm - min_distance)/self.distance_norm)*100

        return pred_label, min_distance, score

    def _img_simiarity_evaluate_by_path(self, path):
        '''
        透過路徑下之圖片與各 label 之距離, 回傳最近似之 label\n
        :param path: 圖片的路徑
        :return: (pred_label, min_distance, score)
        '''
        img = Image.open(path)
        result = self._img_simiarity_evaluate(img)
        if result == None: return None
        pred_label, min_distance, score = result

        return pred_label, min_distance, score

    def _emb_distance_evaluate(self, img):
        '''
        計算圖片與各 label 之歐氏距離\n
        :param img: 輸入圖片
        :return: distance_dict{label : distance}
        '''
        emb_dict = self.emb_vector

        if self.is_cropped:
            cropped_img = self.transform(img)
        else:
            cropped_img = self.mtcnn(img)

        if cropped_img == None:
            if self.test_mode: print('image cannnot find a human face')
            return None
        else:
            img_emb = self.model(cropped_img.unsqueeze(0)).detach().numpy()[0]

        distance_dict = {}
        for label in emb_dict:
            emb_distance = 0
            norm = len(emb_dict[label])

            #### 排除沒有抓到臉的 label
            if '_' in emb_dict[label]: continue

            for vec in emb_dict[label]:
                emb_distance += np.linalg.norm(img_emb - vec)
            emb_distance /= norm
            distance_dict[label] = emb_distance

        return distance_dict

    def show_labels(self):
        '''
        回傳目前有在 db 內的 label list\n
        :return: label_list
        '''
        label_list = []
        for label, _ in self.db.items():
            label_list.append(label)
        return label_list

    ''' 以下測試用 '''
    def test(self):
        '''
            測試 ./test_img 資料夾內的 image 辨識結果\n
        :return: pred_label:預測人員 <br><br>
                 vector_distance: 測試與預測 label embedding_vector 歐式距離 <br><br>
                 score: 用 vector_distance, self.distance_norm 算出來的標準化數值
        '''
        for target_pth in glob.glob(os.path.join(self.test_pth, '*[.jpg-.jpeg]')):
            #### show image
            # display(Image.open(target_pth))
            print('test_img path:', target_pth)

            result = self._img_simiarity_evaluate_by_path(target_pth)
            if result == None: return None
            pred_label, min_distance, score = result

            print('predict:', pred_label)
            print('vector_distance:', min_distance)
            print(f'score: {score} %')
            print('--'*10)

if __name__ == '__main__':
    face_recognition = Face_recognition()

    #### 解除 test_mode, 會操作 sqlite3 insert
    # face_recognition.test_mode = False

    #### 第一次跑或是有新增人員 (label) 時需要先 init
    # face_recognition.emb_init()

    #### 預測來源圖片路徑的 label
    # path = ''
    # face_recognition.predict_by_path(path)

    #### 檢視目前 db 上有哪些 label
    # label_list = face_recognition.show_labels()
    # print(label_list)

    #### 測試 ./test_img 路徑裡的 image 辨識結果
    face_recognition.test()
