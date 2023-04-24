## 마스크 착용 상태 분류 프로젝트(Naver BoostCamp AI Tech 5기 CV-7조)


### 📌  프로젝트 개요
-------------
> - 프로젝트 주제 : 주어진 사람의 이미지 데이터와 메타 데이터를 이용하여 마스크 착용 여부 / 성별 / 연령대를 예측하는 muliti-label classification 문제 해결
![260720E3-F60A-493C-B653-6E69A9D2356A](https://user-images.githubusercontent.com/99079272/233901727-0eb91e67-add0-4436-90a7-fc014d9969ee.png)

> - 📆 대회 일정 : 2023.04.12 ~ 2023.04.20 (11일)

### 🗂️ Dataset
-------------
- image dataset
    - **전체 사람 명 수** : 4,500
    - **한 사람 당 사진의 개수** : 7 [마스크 정상 착용 5장, 오착용(코스크 or 턱스크) 1장, 미착용 1장]
    - **이미지 크기** : (Width=384, Height=512)

- 예측 label
    - **Mask** : Wear / Incorrect / Not Wear
    - **Gender** : Male / Female
    - **Age** : <30 (Young), ≥30 and <60 (Middle), ≥60 (Old)
    
### 🛠️ 개발 환경
-------------
| 서버 OS | Ubuntu 18.04.5 LTS | 편집기 | VS Code |
| --- | --- | --- | --- |
| GPU | Tesla V100 | 언어 | Python 3.x |

### 🍀 Folder Structer
-------------
```bash
├── analysis : 실험 계획 및 결과 분석 정리
├── architecture : 공통으로 사용하지만, 변경이 자주 발생하며 학습할 모델의 구조를 정의
├── common : 변경이 적으면서, 공통으로 사용하는 코드들
│   ├── augmentation.py
│   ├── dataset.py
│   ├── loss.py
│   └── pytorchtools.py
├── model : 학습한 모델의 weight와 log를 저장, architecture 단위로 관리
├── inference.py
├── requirements.txt
├── sample_submission.ipynb
└── train.py
```

### 🔍 최종 모델 선정
- #### Beit v2 + CutMix 
    - **Beit v2** : Backbone Model로 vision transformer와 codebook embedding을 활용한 모델인 Beit v2 활용해 데이터가 적어서 발생하는 overfitting이나 generalization issue를 해결함
    - **CutMix** : 여러 이미지 데이터를 자르고, 섞어서 새로운 데이터를 만드는 기법으로 높은 일반화 성능과 데이터 증강 효과를 얻을 수 있음
- #### ResNet 152 + Imbalanced Dataset Sampler + Custom Augmentation
    - **ResNet 152** : BackBone Model로 깊은 모델인 ResNet152 활용
    - **Imbalanced Dataset Sampler** : 데이터 불균형으로 인해 학습 시 특정 label이 너무 큰 비율을 차지하는 것을 방지하기 위함

### 👨🏻‍💻 👩🏻‍💻  Members
| [김주성](https://github.com/kjs2109) | [김지현](https://github.com/codehyunn) | [박수영](https://github.com/nstalways) |[오동혁](https://github.com/97DongHyeokOH) |[조수혜](https://github.com/suhyehye) |
| --- | --- | --- | --- |  --- |
