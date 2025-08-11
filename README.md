# Link 
- Backend : Spring Boot
  - [github.com/k1sihyeon/ModuCare-SpringBoot](https://github.com/k1sihyeon/ModuCare-SpringBoot)
- Frontend : Android
  - [github.com/k1sihyeon/ModuCare-Android](https://github.com/k1sihyeon/ModuCare-Android)

<br><br>

# 1. 서론
## 개발 배경
 최근 몇 년간 사회복지시설에서의 안전사고가 증가하고 있다. 특히, 노약자 및 장애인이 대부분을 차지하는 사회복지시설 이용자들은 신체적 취약성으로 인해 위험 상황에 빠질 가능성이 크며 신속한 도움을 필요로 한다. 그러나, 대부분의 사회복지시설은 24시간으로 운영되고 있음에도 불구하고 인력 부족 및 안전 관리에 어려움을 겪고 있다.

- [그림 1] KBS MBC, SBS, 사회복지시설에서의 안전사고 발생 뉴스
  ![image](https://github.com/k1sihyeon/ModuCare/assets/96001080/ce203a45-65a5-407a-880e-13df0ce5c48e) <br>
 [그림 1]에서 볼 수 있듯이 치매 노인의 폭행 사건이나 요양원에서의 방임 사례가 뉴스에 보도되고 있다.  이러한 문제들은 사회복지시설의 인력 부족과 관련되어 있다. 24시간으로 운영되는 시설에서 제한된 인력만으로 모든 위험 상황에 신속하게 대응하는 것은 어렵다.

<br>

- [그림 2] 통계청, 65세 이상 고령 인구 비중 전망
  ![image](https://github.com/k1sihyeon/ModuCare/assets/96001080/61b88478-2f49-412c-a7c9-0a3ee054c51c) <br>
[그림 2]는 통계청에서 발표한 65세 이상 고령 인구 비중 전망이다. 자료에 따르면 65세 이상 고령인구의 비중이 급격히 증가하고 있다. 2023년에는 18.4%였던 고령인구 비중이 2030년에는 25.5%, 2040년에는 34.4%로 증가할 것으로 예상된다. 이로 인해 사회복지시설의 수요 또한 크게 증가할 것으로 보인다. 따라서, 이러한 문제를 해결하기 위해 위험 상황을 실시간으로 감지하여 근무자가 신속하게 대응할 수 있는 시스템을 제안하게 되었다.

 본 프로젝트에서는 사회복지시설의 거주자 또는 이용자가 위험 상황에 처했을 때 해당 시설의 근무자에게 애플리케이션을 통해 실시간으로 알림을 전송하는 시스템을 개발하였다. 위험 상황은 시설 이용자의 낙상, 위험 물건 소지 등으로 가정한다. 근무자는 위험 상황에 대한 실시간 알림을 받을 수 있고 해당 상황이 처리되었는지 확인할 수 있으며 다른 근무자들과 정보를 공유할 수 있다.

<br>

# 2. 작품 개요
## 전체 시스템 구조도
![image](https://github.com/k1sihyeon/ModuCare/assets/96001080/6e855b50-3b5a-410c-8a62-afbfa0897a10)

 시스템은 객체 및 행동 탐지를 위해 카메라로 Jetson TX 보드를 사용한다. TX 보드는 카메라로부터 받아온 실시간 영상을 일정 프레임 단위로 분할한다. 딥러닝 기반 실시간 객체 탐지 모델인 YOLO와 인간 행동 추정 모델인 PoseNet을 이용하여 위험 상황을 탐지한다. <br>
 탐지 정보 관리 및 근무자 정보 공유를 위해 스프링 부트와 MySQL DB 그리고 파이어 베이스 클라우드 메시징 서비스를 사용한다. 위험 상황이 감지되면 TX 보드는 서버로 탐지 내용, 시간, 장소, 스냅샷 등을 전송한다. 서버는 다시 근무자가 해당하는 상황을 확인할 수 있도록 파이어 베이스 클라우드 메시징 서비스를 통해 안드로이드 클라이언트에게 푸시 알람을 전송한다. 알림을 받은 근무자는 탐지 내용, 시간, 장소, 스냅샷 등을 확인할 수 있다. <br>
 또한, 안드로이드 애플리케이션과 네이버 지도 API를 통해 자신의 위치와 위험 상황이 발생한 위치를 함께 확인할 수 있다. <br><br>

 - DB ERD <br>
   <img width="855" height="538" alt="DB ERD" src="https://github.com/user-attachments/assets/7527b4ec-9ebe-41fd-be13-6fb75b830237" />

<br>

 - [그림 5] 애플리케이션 순서도 <br>
   <img width="922" height="411" alt="application chart" src="https://github.com/user-attachments/assets/173fdd70-218c-4855-bbdd-556d5cd1deac" /> <br>
   서버와 파이어 베이스 클라우드 메시징 서비스를 통해 위험 탐지 푸시 알림(1)을 받은 사용자는 [그림 5]의 로그 리스트 화면(2)으로 이동한다. 상단의 최신 로그를 선택하면 로그 상세정보 화면(3)으로 이동하며, 위험 탐지 스냅샷, 로그 id, 탐지 위치, 탐지 날짜 및 시간, 탐지 내용 등을 확인할 수 있다. <br>
사용자(근무자)가 환자의 상태를 확인한 이후 “확인했어요” 버튼을 누르면 세부 정보 입력을 위한 팝업창(4)이 나타난다. 환자의 세부 상태를 입력하고 제출하면 “확인했어요” 버튼이 “확인 완료됨” 버튼으로 변경된다. 이후에는 로그 상세정보 화면(5)에서 환자의 상태를 확인한 사용자의 프로필 사진, 이름, 직책, 시간, 환자의 세부 상태 내용 등을 아래에서 확인할 수 있다.

<br>

 - [그림 6] 위치 기반 서비스 <br>
   <img width="866" height="423" alt="location based system" src="https://github.com/user-attachments/assets/2d5b421c-8caf-44fc-9043-993ea55073a5" /> <br>
   [그림 6]의 탐지 위치 정보 화면에서는 위험을 탐지한 위치를 지도상에서 확인할 수 있다. 사회복지시설의 크기와 다수의 카메라가 작동되고 있는 점을 고려하여 위험 상황이 발생한 곳을 한눈에 확인할 수 있도록 마커로 표기한다. 마커를 누르면 해당 위험 상황의 로그 상세정보 화면으로 이동하여 세부 정보를 확인할 수 있다. 또한, 근무자가 확인하지 않는 로그들만 지도에 표시할 수 있도록 하여 혼돈을 줄일 수 있다.

<br><br>

## 시스템 동작
- [그림 7] YOLO 객체 탐지 <br>
  ![image](https://github.com/k1sihyeon/ModuCare/assets/96001080/c9117c6d-4327-4be2-a605-3d0ff72efa37) <br>
  TX 보드의 카메라는 실시간 영상을 일정 시간마다 프레임별로 분할하고 YOLO 모델을 통해 객체를 탐지한다. 사회복지시설 내의 위험 물건을 탐지하기 위해 영상을 [그림 7]와 같이 분석한다. 위험 물건은 칼, 가위 등 예리한 물품으로 가정한다.

<br>

- PoseNet 넘어짐 탐지 <br>
  <img width="1076" height="400" alt="image" src="https://github.com/user-attachments/assets/5c621bee-d664-45aa-a98c-a9685e7d4568" /> <br>
  <img width="486" height="633" alt="flow chart" src="https://github.com/user-attachments/assets/3673abe4-a65c-4121-affa-87e402631f4c" /> <br>
실시간 인간 자세 추정 모델인 PoseNet을 통해 17개의 인간 keypoint를 바탕으로 최상단, 하단, 우측, 좌측 키포인트를 먼저 추출한 다음 Pose의 너비와 높이를 계산한다. 이때, 너비가 높이보다 더 길면 넘어진 것으로 판단한다.

<br>

# 3. 결론
개발한 시스템은 다양한 위험 상황을 실시간으로 탐지하고 위험 상황 발생 시 근무자에게 실시간으로 알림을 전송하여 신속하게 대응을 가능하게 하며 위험 상황의 상세정보를 제공하여 근무자의 상황 판단을 돕는다. <br>
또한, 근무자가 위험 상황을 확인하고 환자 상태를 입력할 수 있도록 하여 사후 관리를 돕는다. 고령 인구가 증가하고 있는 만큼 사회복지시설 뿐만 아니라 다양한 분야에서의 수요가 증가할 것으로 예상되며 특히 의료 분야에서의 인력 부족 문제를 해결할 수 있을 것으로 기대한다.

<br>

# 4. 데모 영상
 - 애플리케이션 데모 <br>
    [![Application Demo](http://img.youtube.com/vi/dJYOpcYKwLI/0.jpg)](https://youtu.be/dJYOpcYKwLI)

 - Object Detection 데모 <br>
    [![Object Detection Demo](http://img.youtube.com/vi/kk1bruN_Pu4/0.jpg)](https://youtu.be/kk1bruN_Pu4)
   
 - Fall Detection 데모 <br>
   [![Fall Detection Demo](http://img.youtube.com/vi/oH1hj0IUWIs/0.jpg)](https://youtu.be/oH1hj0IUWIs)
