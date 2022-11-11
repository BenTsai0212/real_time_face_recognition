'''
    goal: 實現影片即時臉部識別 (recognition), 目標使用在 webcam 上
    process:
        1. 使用 MTCNN 擷取 frame 中的臉部影像
        2. 使用 FaceNet 識別, 與預先存取的 label embedding vectors 進行比對, 輸出識別結果

    update: 2022/11/11
'''

from mtcnn_cv2 import MTCNN
import cv2
import time
import numpy as np
from face_recognition.face_recognition import Face_recognition
from PIL import Image

class Realtime_face_recognition:
    def __init__(self):
        self.detector = MTCNN()
        self.model = Face_recognition()
        self.model.test_mode = False

        #### video setting
        self.frame_rate = 90
        self.process_rate = 30    # 執行 process 的 frame rate, 建議 = self.frame_rate/3
        self.base_detection_confidence = 0.95
        self.base_recognition_score = 55
        self.to_draw = True    # 決定要不要在影像畫框
        self.exit_buttom = 'q'    # 定義影像中途結束按鍵

    def run(self, video_pth):
        '''
            Run face recognition process\n
            :param video_pth: target video path (web_cam: 0)
        '''
        cap = cv2.VideoCapture(video_pth)
        self._realtime_process(cap)

    def _realtime_process(self, video_cap):
        '''
            開啟影像並進行影像處理, 先透過 detection process 抓取人臉,\n
            再透過 recognition process 辨識臉部 label\n
             process_elps 定義影像處理 frame rate\n
            frame_elps 定義影像顯示 frame rate\n
            :param video_cap: video capture (cv2)
        '''
        prev = 0
        process_prev = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()

            #### video end, exiting
            if not ret: break

            frame_elps = time.time() - prev
            process_elps = time.time() - process_prev

            #### frame process
            if process_elps > 1./self.process_rate:
                process_prev = time.time()

                #### detection process
                detected_faces = self.detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for cnt, face in enumerate(detected_faces):
                    if len(detected_faces) == 0: break

                    if face['confidence'] < self.base_detection_confidence: continue
                    mat, size = self._affine_matrix(face['keypoints'])

                    #### recognition process
                    result = cv2.warpAffine(frame, mat, size)    # crop image
                    result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))   # opencv -> PIL.Image
                    predict = self.model.predict(result, is_cropped=True)

                    if predict == None: continue

                    pred_label, _, score = predict
                    if score < self.base_recognition_score: pred_label = 'Unknown'

                    #### draw process
                    if self.to_draw:
                        face_pos = face['box']
                        cv2.rectangle(frame, (face_pos[0], face_pos[1]), (face_pos[0]+face_pos[2],
                                                                          face_pos[1]+face_pos[3]), (0, 255, 0), 3)
                        text = f'{pred_label}'
                        if pred_label != 'Unknown': text += f', score:{int(score)}'
                        cv2.putText(frame, text, (face_pos[0], face_pos[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)

            #### show frame
            if frame_elps > 1./self.frame_rate:
                prev = time.time()
                cv2.imshow('frame', frame)

            #### exit buttom
            if cv2.waitKey(1) == ord(self.exit_buttom): break

        video_cap.release()
        cv2.destroyAllWindows()

    def _affine_matrix(self, lmks, scale=4.5):
        '''
            從 MTCNN output 的 dict['keypoints'] 取得臉部資訊及位置\n
            :param lmks: MTCNN output dict['keypoints']
            :param scale: 臉部框選範圍參數
            :return: np.array(face_position), (face_border_width, face_border_height)
        '''
        nose = np.array(lmks['nose'], dtype=np.float32)
        left_eye = np.array(lmks['left_eye'], dtype=np.float32)
        right_eye = np.array(lmks['right_eye'], dtype=np.float32)
        eye_width = right_eye - left_eye
        angle = np.arctan2(eye_width[1], eye_width[0])
        center = nose
        alpha = np.cos(angle)
        beta = np.sin(angle)
        w = np.sqrt(np.sum(eye_width**2)) * scale
        m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
            [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
        return np.array(m), (int(w), int(w))


if __name__ == '__main__':
    face_recognizer = Realtime_face_recognition()
    test_pth = './test/khaby.mp4'

    face_recognizer.run(test_pth)