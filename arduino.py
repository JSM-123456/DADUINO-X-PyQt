
import sys
from urllib import request
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QCloseEvent, QIcon, QPixmap, QImage, QKeyEvent, QFont
from PyQt5.QtCore import QCoreApplication, Qt, QTimer
import cv2
import numpy as np
import torch
import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Posix 에러 문제 해결 (윈도우 문제)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class App(QWidget):
    ip = "192.168.137.223"
    def __init__(self):
        super().__init__()

        # PyQt 프로그램에 동영상 처리하기 위한 밑작업
        self.stream = request.urlopen('http://' + App.ip +':81/stream')
        self.buffer = b''
        request.urlopen('http://' + App.ip + "/action?go=speed80")
  
        self.initUI()

        # OpenCV 객체검출을 위한 haarcascade 데이터와 라벨 읽어들이기
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detection_enabled = False
        self.line_tracing_enabled = False

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')


    def initUI(self):

        # 동영상 넣을 라벨
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(0, 0, 640, 480)
        self.setWindowTitle('OpenCV x PyQt')

        # 텍스트 넣을 라벨
        self.text_label = QLabel(self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setGeometry(0, 0, 640, 480)
        self.text_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_label.setText("DADUINO AI CAR")

        # 버튼 추가하기
        btn1 = QPushButton('Speed 40', self)
        btn1.resize(60, 30) # 버튼 크기 설정
        btn1.pressed.connect(self.speed40)
        # btn1.move(0, 175) # 창 기준으로 좌표 정하기
        
        btn2 = QPushButton('Speed 60', self)
        btn2.resize(60, 30)
        btn2.pressed.connect(self.speed60)
        # btn2.move(0, 210)

        btn3 = QPushButton('Speed 80', self)
        btn3.resize(60, 30)
        btn3.pressed.connect(self.speed80)
        # btn3.move(0, 245)

        btn4 = QPushButton('Speed 100', self)
        btn4.resize(60, 30)
        btn4.pressed.connect(self.speed100)
        # btn4.move(0, 280)

        btn5 = QPushButton('Forward', self)
        btn5.resize(60, 30)
        btn5.pressed.connect(self.forward)
        btn5.released.connect(self.stop)
        # btn5.move(70, 175)

        btn6 = QPushButton('Backward', self)
        btn6.resize(60, 30)
        btn6.pressed.connect(self.backward)
        btn6.released.connect(self.stop)
        # btn6.move(70, 210)

        btn7 = QPushButton('Left', self)
        btn7.resize(60, 30)
        btn7.pressed.connect(self.left)
        btn7.released.connect(self.stop)
        # btn7.move(70, 245)

        btn8 = QPushButton('Right', self)
        btn8.resize(60, 30)
        btn8.pressed.connect(self.right)
        btn8.released.connect(self.stop)
        # btn8.move(70, 280)

        btn9 = QPushButton('Stop', self)
        btn9.resize(60, 30)
        btn9.pressed.connect(self.stop)      
        # btn9.move(140, 175)

        btn10 = QPushButton('Turn Left', self)
        btn10.resize(60, 30)
        btn10.pressed.connect(self.turnleft)
        btn10.released.connect(self.stop)
        # btn10.move(140, 210)

        btn11 = QPushButton('Turn Right', self)
        btn11.resize(60, 30)
        btn11.pressed.connect(self.turnright)
        btn11.released.connect(self.stop)
        # btn11.move(140, 245)

        btn12 = QPushButton("Face", self)
        btn12.resize(60, 30)
        btn12.clicked.connect(self.haar)
        # btn12.released.connect(self.haaroff)

        btn13 = QPushButton("Self-Driving", self)
        btn13.resize(60, 30)
        btn13.clicked.connect(self.self_driving)


        # 버튼위치 조정을 위한 격자레이아웃 설정
        grid = QGridLayout()
        

        grid.addWidget(btn1, 0, 0)
        grid.addWidget(btn2, 1, 0)
        grid.addWidget(btn3, 2, 0)
        grid.addWidget(btn4, 3, 0)
        grid.addWidget(btn5, 0, 2) # forward
        grid.addWidget(btn6, 2, 2) # backward
        grid.addWidget(btn7, 1, 1) # left
        grid.addWidget(btn8, 1, 3) # right
        grid.addWidget(btn9, 1, 2) # stop
        grid.addWidget(btn10, 0, 4) # turn left
        grid.addWidget(btn11, 1, 4) # turn right
        grid.addWidget(btn12, 2, 4) # Face
        

        # 수직으로 위젯레이아웃 설정
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.label)       
        self.layout.addLayout(grid)

        
        # 업데이트된 프레임을 5ms마다 읽어들이기 (while문과 비슷한 효과)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

        


    # 버튼 눌렀을 때 동작하기 위한 메소드 정의하기
    def speed40(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed40")
        
    def speed60(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed60")
       
    def speed80(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed80")
       
    def speed100(self) :
        request.urlopen('http://' + App.ip + "/action?go=speed100")
        
    def forward(self) :
        request.urlopen('http://' + App.ip + "/action?go=forward")
        

    def backward(self) :
        request.urlopen('http://' + App.ip + "/action?go=backward")
        print("back")
        
    def left(self) :
        request.urlopen('http://' + App.ip + "/action?go=left")
        
    def right(self) :
        request.urlopen('http://' + App.ip + "/action?go=right")
        
    def stop(self) :
        request.urlopen('http://' + App.ip + "/action?go=stop")
        print("stop")

    def turnleft(self) :
        request.urlopen('http://' + App.ip + "/action?go=turn_left")
        
    def turnright(self) :
        request.urlopen('http://' + App.ip + "/action?go=turn_right")

    def haar(self) :
        self.face_detection_enabled = not self.face_detection_enabled
 
    # def haaroff(self) :
    # self.face_detection_enabled = False

    # 자율주행 클릭 시 동작 여부
    def self_driving(self) :
        self.face_detection_enabled = False
        self.line_tracing_enabled = not self.line_tracing_enabled
        if not self.line_tracing_enabled :
            self.stop()

    
    
    def update_frame(self) :
        self.buffer += self.stream.read(4096)
        head = self.buffer.find(b'\xff\xd8')
        end = self.buffer.find(b'\xff\xd9')
        try :
            if head > -1 and end > -1 :
                jpg = self.buffer[head : end+1]
                self.buffer = self.buffer[end+2 :]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                img = cv2.flip(img, 0)
                img = cv2.flip(img, 1)

                # YOLO 모델로 객체 검출하여 stop, slow동작
                if self.line_tracing_enabled :
                    car_state = "go"
                    yolo_state = "go"                    
  
                    thread_frame = img
                    self.line_tracing(thread_frame)
                    results = self.model(thread_frame)
                    detections = results.pandas().xyxy[0]
                    if not detections.empty :
                        for _, detection in detections.iterrows() :
                            x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                            label = detection['name']
                            conf = detection['confidence']
                            if "stop" in label and conf > 0.3 :
                                yolo_state = "stop"
                            elif "slow" in label and conf > 0.3 :
                                yolo_state = "go"

                            color = [0, 0, 0]
                            cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(thread_frame, f"{label} {conf : 2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if car_state == "go" and yolo_state == "go" :
                        self.forward()
                    elif car_state == "right" and yolo_state == "go" :
                        self.right()
                    elif car_state == "left" and yolo_state == "go" :
                        self.left()
                    elif yolo_state == "stop" :
                        self.stop()

                # 객체 검출 시, 필요한 작업
                elif self.face_detection_enabled :
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # scaleFactor : 이미지 크기를 얼마나 축소할지 결정
                    # minNeighbor : 얼굴로 인식되기 위한 최소 이웃 사각형 수, 값이 클수록 검출되는 얼굴이 더 정확하다
                    # minSize : 얼굴 탐지할 때 최소 크기
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50))
                    for (x, y, w, h) in faces :
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "FACE", ((2*x + w - 84) // 2, y-10), cv2.FONT_HERSHEY_PLAIN, 2, 5)


                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # OpenCV 이미지를 QImage로 변환하기
                height, width, channels = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # QImage를 QPixmap로 변환하여 라벨에 표시하기
                pixmap = QPixmap.fromImage(q_img)
                self.label.setPixmap(pixmap)

        except Exception as e :
            print(e)

    # 라인 검출과 중심점을 찾고 자율주행 동작 시키기
    def line_tracing(self, img) :
        global car_state
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        height, width = thresh.shape
        thresh = thresh[height // 2:, :]
        edges = cv2.Canny(thresh, 0, 50)
        cv2.imshow("edges", thresh)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lagest_contour = max(contours, key=cv2.contourArea) if contours else None

        if lagest_contour is not None :
            M = cv2.moments(lagest_contour)
            if M["m00"] != 0 :
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                height, width = img.shape[:2]
                error = cX - width // 2

                if error > 10 :
                    car_state = "left"
                elif error < -10 :
                    car_state = "right"
                else :
                    car_state = "go"
            else :
                cX, cY = 0, 0
                self.stop()
            cv2.circle(img, (cX, cY + height // 2), 10, (0, 255, 0), -1)

        else :
            print("No contour found")
            self.stop()        
        

    def closeEvent(self, event) :   
        event.accept()


    # 키보드 키를 눌렀을 때 동작하기 위한 메소드
    def keyPressEvent(self, event:QKeyEvent) :
        key = event.key()

        # 동작을 부드럽게 하기
        if event.isAutoRepeat() :
            return
    
        if key == Qt.Key_W :
            self.forward()
        elif key == Qt.Key_S :
            self.backward()
        elif key == Qt.Key_A :
            self.left()
        elif key == Qt.Key_D :
            self.right()
        elif key == Qt.Key_Escape :
            self.close()

    # 키보드 키를 뗐을 때 동작하기 위한 메소드
    def keyReleaseEvent(self, event: QKeyEvent) :
        key = event.key()
        if event.isAutoRepeat() :
            return

        if key in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D] :
            self.stop()



if __name__ == '__main__':
   print(sys.argv)
   app = QApplication(sys.argv)
   view = App()
   view.show()
   sys.exit(app.exec_())