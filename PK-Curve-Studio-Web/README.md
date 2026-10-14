# PK Curve Studio

랫드 PK 데이터셋의 최종 5-fold scaffold-CV 결과를 시각화하는 Streamlit 웹 앱입니다.

약물명을 입력하면 PubChem에서 분자구조를 확인합니다. 구조가 데이터셋에 일치하면, 그 약물의 IV 또는 PO profile을 선택해 다음을 한 화면에서 확인할 수 있습니다.

- 문헌에서 수집한 관측 농도점(Observed)
- 동일 scaffold를 학습에서 제외한 fold가 만든 out-of-fold 예측곡선(Predicted)
- 약물·경로·용량별 AAFE, 2-fold, 3-fold, R²-log

이 웹 앱은 새로운 약물의 PK를 자동 예측하지 않습니다. 학습 데이터 성능을 새 예측처럼 표시하지 않기 위해, 사전에 계산된 out-of-fold 결과만 보여줍니다.

## Local launch

```bash
cd /Users/mac/Desktop/PK-AI/pk_curve_studio
/Users/mac/Desktop/PK-AI/.venv/bin/streamlit run app.py
```

## Public deployment

GitHub 저장소에 이 폴더를 올린 뒤 Streamlit Community Cloud에서 `app.py`를 엔트리 포인트로 배포합니다. 공개 사이트에는 `demo_data/`의 OOF 결과만 포함하고, `models/` 및 로컬 원본 데이터베이스는 업로드하지 않습니다.

## Desktop reference-PK simulator

`desktop_app.py`는 별도 로컬 도구입니다. 사용자가 CL, Vss, 그리고 PO의 경우 F를 직접 입력하면, 발표에 사용한 final dual-absorption neural ensemble으로 C–t curve를 계산합니다.

```bash
/Users/mac/Desktop/PK-AI/.venv/bin/python desktop_app.py
```
