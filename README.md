# level1_bookratingprediction_recsys-level-recsys-07

![main](https://user-images.githubusercontent.com/50396533/147069300-5038c779-faa4-404b-b1fd-e9e3896f06b4.png)
# 마스크 착용 상태 분류
## 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task
### Overview
박우석

### Requirements
```bash
pip install -r requirements.txt
```
### Dataset (저작권 이슈로 깃헙 업로드가 불가능합니다)
채우기




### EDA 

### 전처리

### Model
최종 모델 ok 

### Result
Model, Task에 따라 5-Fold Cross Validation 했을 때 나온 Validation Accuracy의 평균값입니다.  
Data leakage를 방지하기 위해 동일한 사람에 대한 7장의 이미지들이 Train 데이터셋, Validation 데이터셋 둘 중 하나에만 포함되도록 하였습니다.
|Model \ Task|Mask|Gender|Age|
|---|---|---|---|
|NCF(5-Fold)|97.68|94.98|89.57|
|어쩌고저쩌고(5-Fold)|97.67|96.51|90.16|
|EfficientnetB3(5-Fold)|97.75|96.04|90.20|
| **Ensemble** | **100** | **100** | **100**|


### 회고
- Wrap-up report 내용 중 몇개 뽑아서 정리?
