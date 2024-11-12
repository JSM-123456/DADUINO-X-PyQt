
import sys
from urllib import request
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QCloseEvent, QIcon, QPixmap, QImage, QKeyEvent, QFont
from PyQt5.QtCore import QCoreApplication, Qt, QTimer
import cv2
import numpy as np




class App(QWidget):
    ip = "192.168.137.223"
    def __init__(self):
        super().__init__()
        self.stream = request.urlopen('http://' + App.ip +':81/stream')
        self.buffer = b''
        request.urlopen('http://' + App.ip + "/action?go=speed80")
        self.initUI()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detection_enabled = False

    def initUI(self):

                
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(0, 0, 640, 480)
        self.setWindowTitle('OpenCV x PyQt')

        self.text_label = QLabel(self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setGeometry(0, 0, 640, 480)
        self.text_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_label.setText("DADUINO AI CAR")

        btn1 = QPushButton('Speed 40', self)
        btn1.resize(60, 30) # 버튼 크기 설정
        btn1.pressed.connect(self.speed40)
        
        #btn1.move(0, 175) # 창 기준으로 좌표 정하기
        
        btn2 = QPushButton('Speed 60', self)
        btn2.resize(60, 30)
        btn2.pressed.connect(self.speed60)

        #btn2.move(0, 210)

        btn3 = QPushButton('Speed 80', self)
        btn3.resize(60, 30)
        btn3.pressed.connect(self.speed80)

        #btn3.move(0, 245)

        btn4 = QPushButton('Speed 100', self)
        btn4.resize(60, 30)
        btn4.pressed.connect(self.speed100)
    
        #btn4.move(0, 280)

        btn5 = QPushButton('Forward', self)
        btn5.resize(60, 30)
        btn5.pressed.connect(self.forward)
        btn5.released.connect(self.stop)
        #btn5.move(70, 175)

        btn6 = QPushButton('Backward', self)
        btn6.resize(60, 30)
        btn6.pressed.connect(self.backward)
        btn6.released.connect(self.stop)
        #btn6.move(70, 210)

        btn7 = QPushButton('Left', self)
        btn7.resize(60, 30)
        btn7.pressed.connect(self.left)
        btn7.released.connect(self.stop)
        #btn7.move(70, 245)

        btn8 = QPushButton('Right', self)
        btn8.resize(60, 30)
        btn8.pressed.connect(self.right)
        btn8.released.connect(self.stop)
        #btn8.move(70, 280)

        btn9 = QPushButton('Stop', self)
        btn9.resize(60, 30)
        btn9.pressed.connect(self.stop)      
        #btn9.move(140, 175)

        btn10 = QPushButton('Turn Left', self)
        btn10.resize(60, 30)
        btn10.pressed.connect(self.turnleft)
        btn10.released.connect(self.stop)
        #btn10.move(140, 210)

        btn11 = QPushButton('Turn Right', self)
        btn11.resize(60, 30)
        btn11.pressed.connect(self.turnright)
        btn11.released.connect(self.stop)
        #btn11.move(140, 245)

        btn12 = QPushButton("Face", self)
        btn12.resize(60, 30)
        btn12.clicked.connect(self.haaron)
        #btn12.released.connect(self.haaroff)


        
        
        
       
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
        

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.label)       
        self.layout.addLayout(grid)

        



        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

        

        



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

    def haaron(self) :
        self.face_detection_enabled = not self.face_detection_enabled

    #def haaroff(self) :
    #   self.face_detection_enabled = False

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

                if self.face_detection_enabled :
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # scaleFactor : 이미지 크기를 얼마나 축소할지 결정
                    # minNeighbor : 얼굴로 인식되기 위한 최소 이웃 사각형 수, 값이 클수록 검출되는 얼굴이 더 정확하다
                    # minSize : 얼굴 탐지할 때 최소 크기
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50))
                    for (x, y, w, h) in faces :
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "FACE", ((2*x + w - 84) // 2, y-10), cv2.FONT_HERSHEY_PLAIN, 2, 5)


                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # OpenCV 이미지를 QImage로 변환
                height, width, channels = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # QPixmap을 라벨에 표시
                pixmap = QPixmap.fromImage(q_img)
                self.label.setPixmap(pixmap)

        except Exception as e :
            print(e)
            
        


    def closeEvent(self, event) :
        
        event.accept()

    def keyPressEvent(self, event:QKeyEvent) :
        key = event.key()
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