import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

# 레이블링 정보
LABELS = ['a', 'b', 'c']

# 딥러닝 모델 불러오기
model = keras.models.load_model('models/abc.h5')

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# 카메라 열기
cap = cv2.VideoCapture(1)

# 입력 데이터 준비
x_test = []

# 프레임 처리
while True:
    # 카메라에서 프레임 가져오기
    ret, frame = cap.read()

    # Mediapipe를 사용하여 감지
    # 좌표는 정규화(normalized) 된 좌표값을 사용합니다.
    # 정규화 좌표값을 다시 픽셀 좌표값으로 변환하기 위해서 화면 크기를 가져와야 합니다.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # 손 인식된 경우 좌표 추출
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 손 인식 좌표 추출 (18개)
        landmark_list = []
        for i in range(18):
            landmark_list.append(hand_landmarks.landmark[i].x)
            landmark_list.append(hand_landmarks.landmark[i].y)
            landmark_list.append(hand_landmarks.landmark[i].z)

        # 데이터 저장
        x_test.append(landmark_list)

        # 입력 데이터 준비
        if len(x_test) >= 30:
            x_test_array = np.array([x_test[-30:]])
            x_test_array = x_test_array.reshape((-1, 30, 3*18))

            # 딥러닝 모델 예측
            y_predict = model.predict(x_test_array)

            # 결과 출력
            label = LABELS[np.argmax(y_predict)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (10, 40), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 프레임 출력
    cv2.imshow('Hand Gesture Recognition', frame)

    # 영상이 출력되도록 해주는 코드
    if cv2.waitKey(1) == ord('q'):
        break

# 자원 반환
cap.release()
cv2.destroyAllWindows()
