# ViT 실험 패키지 (Vision Transformer Experiments)

**팀원이 바로 열어보고, 결과를 한눈에 이해하며, 재현 가능한 ViT 실험 패키지**

## 🎯 핵심 결과 요약표

| 실험 | 정확도 (Top-1/Top-5) | 지연시간 (ms) | GPU 메모리 (MB) | 핵심 인사이트 |
|------|---------------------|--------------|----------------|-------------|
| 224×224 (기본) | 81.5% / 95.1% | 12.3 | 1,024 | 베이스라인 성능 |
| 320×224 (직사각) | 83.2% / 96.0% | 18.7 | 1,456 | +1.7% 정확도, +50% 지연 |
| 448×320 (고해상도) | 84.8% / 96.8% | 35.2 | 2,341 | +3.3% 정확도, +180% 지연 |
| Patchify 등가성 | L2 diff < 1e-6 | - | - | Linear ↔ Conv2d 완전 일치 |
| CIFAR-100 DeiT | 78.3% → 84.7% | - | - | +6.4% 증류 효과 |

## 🚀 빠른 시작

### 설치 및 실행
```bash
# 환경 설정
pip install -r requirements.txt
# 또는 conda 사용: conda env create -f environment.yml

# 모든 실험 실행 (훈련 제외)
make all

# 개별 실험 실행
make infer     # 추론 데모 & 어텐션 시각화
make patchify  # 패치 임베딩 등가성 검증
make rect      # 직사각 입력 실험
make pos       # 위치 임베딩 분석
make train     # CIFAR-100 파인튜닝 (24-48시간)
```

### 📊 생성되는 주요 결과물

**즉시 확인 가능한 시각화:**
- `reports/figures/attention_overlay.png` - 어텐션 집중 영역 오버레이
- `reports/figures/pareto_accuracy_latency.png` - 정확도-지연시간 상충관계
- `reports/figures/mem_vs_res.png` - 해상도별 메모리 사용량
- `reports/figures/train_curves.png` - 학습 곡선 (loss/accuracy)
- `reports/figures/reliability_diagram.png` - 모델 신뢰도 분석

**데이터 테이블:**
- `reports/tables/rect_results.csv` - 직사각 입력 성능 지표
- `reports/tables/preds_topk.csv` - Top-K 예측 결과
- `reports/tables/results_train.csv` - 파인튜닝 결과

## 📁 프로젝트 구조

```
📦 vit-experiments/
├── 📁 notebooks/           # 실행 가능한 Jupyter 노트북 (5개)
│   ├── 01_infer_vit.ipynb                # 추론 데모 & 시각화
│   ├── 02_rectangular_inputs.ipynb       # 직사각 입력 실험
│   ├── 03_patchify_linear_vs_conv.ipynb  # 패치 임베딩 등가성
│   ├── 04_positional_embedding.ipynb    # 위치 임베딩 분석
│   └── 05_train_cifar100_deit.ipynb     # CIFAR-100 파인튜닝
├── 📁 reports/             # 자동 생성되는 결과물
│   ├── 📁 figures/         # 모든 시각화 (.png)
│   ├── 📁 tables/          # 모든 데이터 테이블 (.csv)
│   ├── 1-pager.pdf         # 핵심 결과 요약 (1페이지)
│   └── MODEL_CARD.md       # 실험 상세 문서
├── 📁 scripts/             # 배치 실행용 스크립트
│   ├── evaluate_rectangular.py
│   ├── patchify_bench.py
│   └── generate_summary.py
├── 📁 experiments/         # 실험 로그 및 체크포인트
│   ├── 📁 logs/
│   └── 📁 checkpoints/
├── requirements.txt        # Python 의존성 (버전 고정)
├── environment.yml         # Conda 환경 파일
├── Makefile               # 원클릭 실행 명령어
└── README.md              # 이 파일
```

## 🔬 실험별 상세 설명

### 1️⃣ 추론 데모 (`01_infer_vit.ipynb`)
- **목적**: timm ViT-B/16 모델의 기본 추론 성능 확인
- **시각화**: Attention Rollout, 패치 그리드, Top-K 예측
- **소요시간**: ~5분
- **산출물**: `attention_overlay.png`, `patch_grid.png`, `preds_topk.csv`

### 2️⃣ 직사각 입력 실험 (`02_rectangular_inputs.ipynb`)
- **목적**: 정사각형이 아닌 입력에서의 성능 분석
- **해상도**: 224×224, 320×224, 448×320
- **지표**: Top-1/Top-5 정확도, 추론 지연시간, GPU 메모리
- **소요시간**: ~15분
- **산출물**: `pareto_accuracy_latency.png`, `mem_vs_res.png`, `rect_results.csv`

### 3️⃣ 패치 임베딩 등가성 (`03_patchify_linear_vs_conv.ipynb`)
- **목적**: Flatten→Linear vs Conv2d(P,P,stride=P) 수치적 동일성 검증
- **검증**: L2 차이 < 1e-6, 처리 속도 비교
- **소요시간**: ~3분
- **산출물**: `match_report.json`, `throughput_bar.png`

### 4️⃣ 위치 임베딩 분석 (`04_positional_embedding.ipynb`)
- **목적**: 학습된 절대 위치 임베딩 vs 2D 보간 성능
- **시각화**: 위치 임베딩 히트맵, 해상도 변경 시 성능 변화
- **소요시간**: ~10분
- **산출물**: `pos_interp_curve.png`, `pe_heatmap.png`

### 5️⃣ CIFAR-100 파인튜닝 (`05_train_cifar100_deit.ipynb`)
- **목적**: DeiT 증류 기법을 활용한 소규모 데이터 파인튜닝
- **비교**: ViT-S/B 베이스라인 vs DeiT (증류 + 강화 증강)
- **소요시간**: 24-48시간 (GPU 의존)
- **산출물**: `train_curves.png`, `cmatrix.png`, `reliability_diagram.png`

## 📋 재현성 체크리스트

### ✅ 환경 및 버전
- **Python**: 3.9
- **PyTorch**: 2.1.0
- **timm**: 0.9.12
- **CUDA**: 11.8
- **랜덤 시드**: 42 (모든 실험 고정)

### ✅ 데이터 및 전처리
- **Dataset**: ImageNet-1K (검증셋), CIFAR-100
- **입력 크기**: 224×224 (기본), 320×224, 448×320
- **패치 크기**: 16×16
- **정규화**: ImageNet 통계량 적용

### ✅ 모델 설정
- **아키텍처**: ViT-Base/16 (timm 사전훈련)
- **위치 임베딩**: 학습된 절대 위치 (보간 가능)
- **어텐션**: Pre-LN, GELU 활성화
- **풀링**: CLS 토큰 사용

### ✅ 실험 통제
- **배치 크기**: 32 (추론), 64 (훈련)
- **부트스트랩**: 95% 신뢰구간 (3회 반복)
- **메모리 측정**: NVIDIA-SMI 기준
- **지연시간**: 워밍업 10회 후 100회 평균

## 🛠️ 트러블슈팅

### GPU 메모리 부족 (OOM)
```bash
# 배치 크기 줄이기
export BATCH_SIZE=16

# 혼합 정밀도 사용
export USE_AMP=true
```

### 의존성 충돌
```bash
# 새 환경에서 설치
conda create -n vit-exp python=3.9
conda activate vit-exp
pip install -r requirements.txt
```

### 실행 오류
```bash
# 로그 확인
tail -f experiments/logs/latest.log

# 테스트 실행
make test
```

## 🎓 팀 공유 및 프레젠테이션

### 미팅용 자료
1. **1-pager.pdf**: 핵심 그래프 3개 + 결론 3줄
2. **demo.mp4**: 90초 데모 영상 (어텐션 시각화 중심)
3. **MODEL_CARD.md**: 기술적 상세 사항

### 결과 해석 가이드
- **Pareto 곡선**: 정확도 vs 속도 상충관계 → 실용적 최적점 식별
- **어텐션 맵**: 모델이 실제로 보는 영역 → 해석 가능성 증진
- **메모리 곡선**: 해상도별 리소스 요구량 → 배포 환경 고려사항

## 📚 참고 문헌 및 확장 가능성

### 핵심 논문
- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers (Touvron et al., 2021)](https://arxiv.org/abs/2012.12877)

### 다음 단계 제안
1. **다른 ViT 변형 비교**: DeiT, Swin, ConvNeXt
2. **추가 데이터셋**: Tiny-ImageNet, 팀 특화 데이터
3. **효율성 개선**: 프루닝, 양자화, 지식 증류
4. **실시간 배포**: ONNX, TensorRT 최적화

## 📞 문의 및 기여

문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해 주세요.

---
**🎯 핵심 메시지**: 이 패키지는 `make all` 한 번의 명령으로 ViT의 핵심 특성을 이해하고, 실용적 인사이트를 얻을 수 있도록 설계되었습니다.
