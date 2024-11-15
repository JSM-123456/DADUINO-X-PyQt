## 설명
파이썬 프로그램으로 아두이노 자동차를 제어하는 GUI를 만들었습니다.
수동 모드로서 키보드와 마우스 클릭으로 속도 조절 및 주행 방향을 조종할 수 있고 카메라를 이용한 얼굴 검출 할 수 있습니다.
자율주행 모드로 트랙따라 자동차가 움직이고 YOLO 모델을 학습시켜 학습된 객체에 맞춰 반응할 수 있습니다.

## 사용방법
1. 코드 사용에 필요한 라이브러리를 pip install을 통해 설치해줍니다.
pip install opencv -- python PyQt5 torch yolov5
2. 얼굴 검출을 위해 https://github.com/opencv/opencv/tree/master/data/haarcascades 에서 haarcascade_frontalface_default.xml 파일을 다운받고 코드 실행할 폴더에 저장시킵니다.
3. 학습된 YOLO 모델을 사용하기 위해 best.pt 파일을 코드 실행할 폴더에 저장시킵니다.
4. 아두이노 자동차를 윈도우 모바일 핫스팟에 연결하여 ip주소를 받아 파이썬 코드에 ip주소를 수정합니다.

## Version log
1. 버튼마다 동작하는 것과 OpenCV동영상을 PyQt 위젯에 나타태는 기능을 만들었습니다.
2. haar 데이터 파일을 활용하여 얼굴 검출 기능을 추가하였습니다.
3. 라인 검출하여 자율주행하는 기능을 추가하였습니다.
4. YOLO모델 학습한 파일로 객체 검출하여 동작하는 기능을 추가하였습니다.
