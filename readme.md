# 불법사이트 HTML 구조 분석 BAG 모델
## Paper: [BAG.pdf](https://github.com/user-attachments/files/20265301/BAG.pdf)
본 연구에서는 HTML 태그의 텍스트와 구조 정보를 결합하여 불법 도박 사이트를 효과적으로 탐지할 수 있는 하이브리드 모델 BAG (BERT-Attention GNN)을 제안한다. 제안된 모델은 HTML 각 태그의 텍스트를 BERT로 분석해 불법성 점수를 산출하고, 이를 GAT의 attention 연산에 반영하여 중요 노드의 정보를 강화한다. 또한, Weighted Pooling을 도입해 불법 키워드를 담은 노드가 많을수록 판정결과에 기여하도록 설정하였다. 불법 도박 사이트와 정상 사이트의 HTML 데이터를 활용한 실험 결과, 훈련 정확도 89%와 추론 성능 정확도 90%의 우수한 성능을 확인하였으며, 추가 제거 연구를 통해 BERT-Attention 연산과 텍스트 감정분석이 HTML 분류 성능 향상에 기여하고 있음을 입증하였다.  



### 1.Bert-Attention 수식  
<div align="center">
  <img width="700" src="https://github.com/user-attachments/assets/9e9afb6c-1dd8-4715-b6c6-64f54bf82c3c"/>
</div>    
HTML 각 태그의 텍스트 감정분석을 통해 태그의 불법키워드 점수를 산출합니다. 이후 GAT의 Attention 연산에서 해당 점수를 반영하여 HTML의 그래프 구조뿐 아니라, 불법 텍스트 키워드에 기반하여 확인할 수 있도록 합니다.

### 2.모델의 전체 구조  
<div align="center">
  <img width="985" alt="Screenshot 2025-04-29 at 10 59 51 AM" src="https://github.com/user-attachments/assets/d0af47e7-2855-4096-be70-c2fba41dbb51" />
</div> 

### 3.Custon GNN의 구조
<div align="center">
  <img width="985" alt="Screenshot 2025-04-29 at 10 59 51 AM" src="https://github.com/user-attachments/assets/61c664b1-7bd5-4796-be90-6d963a3f3122" />
</div> 

## 주요 특징

- HTML DOM 트리를 그래프로 변환하여 분석
- 도박 관련 키워드에 대한 점수(`gambling_score`)를 어텐션 메커니즘에 활용
- One-class classification 방식으로 학습
- PyTorch Geometric 프레임워크 기반

## 파일 구조

- `dataset.py`: 데이터셋 클래스 (JSON -> PyTorch Geometric)
- `model.py`: GAT 모델 구현 (gambling_score를 어텐션에 적용)
- `train.py`: 모델 학습 코드
- `inference.py`: HTML 파일에 대한 추론 코드

## 설치 요구사항

```bash
pip install torch torch-geometric networkx numpy matplotlib tqdm scikit-learn
```

## 주의 사항

- 학습 데이터는 모두 불법도박사이트이므로 one-class classification을 사용합니다.
- `gambling_score`는 HTML 태그에 포함된 도박 키워드의 빈도 또는 관련성을 나타냅니다.
- 과적합이 문제가 되지 않으며, 오히려 불법도박사이트 패턴을 정확히 학습하기 위해 권장됩니다. 
