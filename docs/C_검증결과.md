# C 담당 검증 결과

확인일은 2026-11-03(KST), 확인자는 Codex입니다. Python 3.11.15 CPU 환경에서 자동검사 32개와 실제 CLIP 검사 5개, Python 3.13.5 CUDA 환경에서 실제 CLIP 검사 5개를 통과했습니다. 실제 CCTV 검색 성능이나 A/B 전체 연결을 검증한 결과는 아닙니다.

## 실행 명령과 관측 결과

작업 위치는 `C:\컴퓨터비전\cctv-search`입니다.

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\.venv311\Scripts\python.exe -m pip check
.\.venv311\Scripts\python.exe -m compileall -q embedding fixtures tools tests
.\.venv311\Scripts\python.exe -m tools.verify_embedding --output runs/c-verified-cpu-20261103
.\.venv\Scripts\python.exe -m tools.verify_embedding --output runs/c-verified-gpu-20261103
.\.venv311\Scripts\python.exe -m embedding validate runs/c-verified-cpu-20261103/cold
.\.venv\Scripts\python.exe -m embedding validate runs/c-verified-gpu-20261103/cold
```

모든 명령 exit code=0. unittest는 `Ran 32 tests`, `OK`, pip check는 `No broken requirements found.`를 출력했습니다. 실제 모델 검사는 환경별 `PASS: 5 real-model checks`, 두 인덱스 검사는 `status=success, row_count=4`를 출력했습니다.

자동검사 원문은 `docs/verification/unit-tests-python311.txt`, CPU 환경의 전체 패키지 버전은 `docs/verification/python311-cpu.freeze.txt`입니다. 실제 모델의 입력 해시·기대값·실측값·PASS·확인자·환경은 각 실행 폴더의 verification_report.json에 있습니다. 같은 명령을 재실행할 때 출력 폴더는 새 이름으로 지정합니다.

## 대표 경계 및 실패 검사

- 소수 박스 `[10.2,10.5,30.8,40.1]`, 원본 100×80, 여백 10%의 기대 crop은 `[8,7,33,44]`. 실측 일치, PASS.
- 원본 전체 박스 `[0,0,100,80]`의 기대 crop은 같은 원본 경계. 실측 일치, PASS. 음수·0면적·경계 초과·NaN 박스 거부도 PASS.
- 실패 object 1개와 정상 scene 1개 입력에서 기대값은 성공 1·실패 1·vector_row=[0]·stage=failed. 실측 일치, PASS. 기본 validate는 해당 묶음을 거부함.
- 메모리 오류를 주입했을 때 요청 배치의 기대 순서는 16→8→4→1. 실측 일치, PASS. 배치 1에서도 실패하면 두 번 시도 후 제외, PASS. 실제 GPU OOM을 강제로 유발한 검사는 아님.
- 원본 해시·디코더·여백·가중치·전처리 변경, 캐시 손상, 결과 파일/미디어 손상, split 혼합, 실제 프레임 불일치, JSONL 형식·경로 이탈·중복 ID를 검사. 기대한 재생성 또는 거부 확인, PASS.

## 실제 CLIP 검사

입력은 합성 장면 2장과 scene/object samples 4개입니다. 공식 CLIP ViT-B/32와 공식 전처리를 사용했습니다. 두 환경 모두 items 4행·vectors float32 `[4,512]`이며 모든 값이 유한합니다.

CPU의 최대 정규화 오차는 `5.960464477539063e-08`, GPU는 `0.0`으로 허용값 `0.0001` 이하입니다. warm 실행은 벡터 캐시 4/4 hit, 인코딩 배치 0회, cold 벡터와 배열 값이 정확히 같았습니다. 빈 입력은 `[0,512]`, status=success였습니다.

이미지 2장과 영어 문장 2개를 각각 `[2,512]`로 인코딩했고, 이미지 질의와 같은 scene의 인덱스 벡터는 오차 0.0001 이내에서 일치했습니다. 빈 문장·토큰 한도 초과 문장을 거부하고 빈 질의 목록은 `[0,512]`로 반환했습니다.

합성 예제 4개 처리 구간의 실측은 CPU cold 약 257.6ms·warm 약 48.8ms, GPU cold 약 294.2ms·warm 약 32.1ms입니다. 모델 로딩·원본 해시 검사·최종 직렬화/공개를 제외한 값이며, 실행마다 변동합니다. 이 수치로 실제 영상 처리속도나 CPU/GPU 우열을 판단하지 않습니다.

GPU는 RTX 3050 Ti Laptop 4GB, PyTorch 2.7.1+cu118을 사용했습니다. CPU는 PyTorch 2.7.1+cpu입니다. 모델 소스 리비전은 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, 가중치 SHA256은 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`입니다.

## 실제 통합 시 남은 확인

실제 A samples와 B RGB 프레임 함수, 원본 CCTV가 제공되지 않아 해당 연결·실영상 1분 성능 측정은 아직 하지 않았습니다. Python 3.11 GPU 조합도 별도로 실행하지 않았습니다. 대상 환경이 고정되면 G가 설치/통합을 실행하고 B의 시각·해상도·offset 및 실제 모델 메모리 사용량을 확인해야 합니다.

D·F·G의 인수 실행은 팀원이 수행해야 하며, 현재 보고서의 확인자는 Codex입니다. 실제 데이터의 검색 Recall·MRR·보존율, PPT 편집과 전체 웹 시연은 각 담당자의 후속 통합 범위입니다.
