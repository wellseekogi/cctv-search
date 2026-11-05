# CCTV 검색 C 담당 구현

## 팀원 공유와 브랜치 선택

공개 저장소는 [wellseekogi/cctv-search](https://github.com/wellseekogi/cctv-search)입니다. 이 `main` 브랜치는 CLIP 기준선 코드입니다. 기존 작업은 `work/clip-baseline`, CLIP/SigLIP2 모델 실험과 최신 팀원용 설치 안내는 [feature/siglip2](https://github.com/wellseekogi/cctv-search/tree/feature/siglip2)에 있습니다.

```powershell
git clone --branch feature/siglip2 https://github.com/wellseekogi/cctv-search.git
cd cctv-search
```

모델 가중치·가상환경·캐시·실행 결과는 Git에 포함하지 않습니다. 각 PC에서 설치하고 모델을 내려받습니다. 아래 내용은 CLIP 기준선의 실행 설명과 검증 기록입니다.

최종 7인 실행명세서 v5.0(2026-11-02)의 C 범위를 구현했습니다. scene·object 이미지 생성, 공식 OpenAI CLIP ViT-B/32 이미지·텍스트 인코더, 캐시, 재시도, 인덱스 검증을 제공합니다. A의 선택 정책과 B의 영상 디코더는 각 담당자의 모듈을 연결합니다.

## 현재 PC에서 실행

프로젝트 폴더 `C:\컴퓨터비전\cctv-search`에서 PowerShell을 엽니다. 준비된 Python 3.11 환경은 `.venv311`, 기존 GPU 패키지를 사용하는 Python 3.13 환경은 `.venv`입니다.

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\.venv311\Scripts\python.exe -m tools.verify_embedding --output runs/my-c-check
```

검증 도구는 실제 CLIP을 사용해 cold·warm·empty 인덱스와 이미지 2장·영어 문장 2개를 생성합니다. 출력 경로는 매번 새 이름을 지정합니다. 원본·가중치는 ZIP에 포함하지 않으며 첫 실행 시 공식 가중치 약 338MiB를 `models/ViT-B-32.pt`에 내려받습니다.

## 새 환경 설치

기준은 Python 3.11입니다. CPU 재현 명령은 다음과 같습니다. GPU용 PyTorch 설치는 G가 장비에 맞게 결정한 뒤 나머지 요구사항을 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-embedding.txt
```

공통 `requirements.lock`은 G 담당이며, C가 고정한 직접 의존성은 `requirements-embedding.txt`, 실측 패키지 버전은 검증 보고서에 기록됩니다. CLIP 소스 리비전과 가중치 SHA256은 `configs/embedding.yaml`에 고정했습니다. 공식 전처리는 반환된 transform을 그대로 사용하며 실제 전처리 해시는 실행 폴더의 `embedding_config.json`에 저장합니다.

## 입력과 B 연결 계약

입력은 F의 `manifest.jsonl`, A의 `samples.jsonl`, 원본 파일을 담은 `data_root`, B의 `read_frame(video_id, requested_timestamp_ms)`입니다. schema는 `cctv-3.0`, 시각은 원본 기준 정수 ms, bbox는 여백 전 원본 좌표입니다.

B 함수 반환값은 아래 dictionary로 맞춥니다. B의 반환 형태가 다르면 해당 모듈에 얇은 변환 함수를 두고 CLI에 그 함수를 전달합니다.

```python
{
    "image_rgb": rgb_uint8_array,  # [height, width, 3] 또는 RGB PIL.Image
    "actual_timestamp_ms": 1200,
    "frame_index": 6,
    "width": 1920,
    "height": 1080,
    "time_source": "pts",
    "source_offset_ms": 0,
}
```

읽기 실패는 `{"error": {"code": "FRAME_NOT_FOUND", "message": "..."}}`로 반환하거나 예외를 발생시킵니다. C는 오류 코드를 보존하고 한 번 재시도합니다. 예외에 `code` 속성이 있으면 그 값을 기록합니다.

A의 samples는 이미 실제 프레임을 선택한 결과이므로 반환 시각·프레임 번호가 samples와 정확히 같아야 합니다. B의 일반 요청 허용차 100ms를 이용해 다른 프레임으로 바꾸지 않습니다. 크기·offset 불일치, 반개구간 `[available_start_ms, available_end_ms)` 밖 시각, 잘못된 bbox는 실패 항목으로 남깁니다. GT·pred의 출처, event_ids, reasons, policy, source_run_id와 원래 item_id를 유지합니다.

## 작은 인덱스 생성

```powershell
.\.venv311\Scripts\python.exe -m fixtures.embedding.generate
.\.venv311\Scripts\python.exe -m embedding build --samples fixtures/embedding/samples.jsonl --manifest fixtures/embedding/manifest.jsonl --data-root fixtures/embedding --output runs/c-example --cache .cache/embedding --frame-reader fixtures.embedding.reader:read_frame --decoder-id fixture-rgb-v1
.\.venv311\Scripts\python.exe -m embedding validate runs/c-example
```

실제 입력으로 전환할 때 samples·manifest·data-root를 바꾸고, `--frame-reader`에 B의 `모듈:함수`를 지정합니다. `--decoder-id`에는 디코더 이름·버전·시각/색 변환 규칙을 구별할 수 있는 고정 문자열을 넣습니다. 실행 위치가 달라지면 `--config`도 지정합니다. `model_root`는 작업 디렉터리 기준입니다.

## 출력과 D 연결

`items.jsonl`과 `vectors.npy`는 성공 항목만 같은 순서로 저장합니다. vector_row는 `0..N-1`, vectors는 float32 `[N,512]`, 모든 값은 유한하며 L2 정규화 오차는 0.0001 이하입니다. 빈 입력은 items 0행·vectors `[0,512]`·status=success입니다.

`media/frames`에 전체 프레임을, `media/crops`에 객체 이미지를 무손실 PNG로 저장합니다. object는 가로·세로 각각 10%를 양쪽에 추가하고 시작 좌표 floor·끝 좌표 ceil 후 원본 경계로 맞춥니다. items의 `bbox`는 원래 박스, `crop_bbox`는 실제 자른 정수 박스입니다. `full_frame_path`는 E의 박스 표시용 전체 프레임입니다. 생성 경로는 출력 폴더 기준 `/` 구분자를 사용합니다.

`index_manifest.json`에는 파일 해시·모델·전처리·차원·dtype·정규화·분할·관측 출처가 들어갑니다. dev와 eval은 별도 인덱스로 생성합니다. D는 읽기 전에 다음을 실행합니다.

```python
import numpy as np
from embedding import ClipEncoder, validate_index

encoder = ClipEncoder(model_root="models", device="auto")
validate_index("runs/c-example", expected_metadata=encoder.metadata)
query_vectors = encoder.encode_text(["a white car near an intersection"])
# encoder.encode_image([rgb_image])도 같은 float32 [N,512]를 반환합니다.
gallery_vectors = np.load("runs/c-example/vectors.npy", allow_pickle=False)
scores = query_vectors @ gallery_vectors.T
```

문장 토큰 길이를 초과하면 잘라 쓰지 않고 예외를 반환합니다. 실제 검색 순위, 트랙 집계, API, 이미지 업로드 검사와 EXIF 보정은 D 담당입니다. 이미지 인코더 입력은 이미 RGB여야 합니다.

## 캐시와 실패 처리

이미지 키에는 원본 SHA256·실제 시각·프레임·bbox·여백·디코더·RGB/PNG 규칙을 넣습니다. 벡터 키는 이미지 식별과 실제 이미지 해시, 모델 ID·리비전·가중치 해시·전처리·구현 해시·dtype·정규화를 포함합니다. 캐시 데이터와 성공 메타데이터를 임시 파일로 작성한 뒤 확정하고, 로드할 때 해시와 이미지/벡터 유효성을 재검사합니다. 손상되거나 조건이 바뀐 캐시는 재생성합니다. warm 실행에서도 B 반환 메타데이터 확인을 위해 프레임을 읽습니다.

읽기·crop·일반 인코딩 오류는 같은 설정으로 한 번 재시도합니다. GPU 시작 배치는 16, CPU는 4입니다. OOM이면 16→8→4→1로 축소하고 배치 1에서도 한 번 재시도합니다. 일반 배치 오류가 두 번 나면 항목별로 분리해 정상 항목을 보존합니다. 배치별 요청 크기·실제 크기·시도·결과를 기록합니다.

실패 항목은 items와 vectors에서 제외하고 `errors.jsonl`에 오류 코드·시도·복구 여부·run_id를 기록합니다. 일부라도 최종 실패하면 bundle의 status=failed이며 `validate_index` 기본 검사에서 거부합니다. 원인 확인 후 새 출력 폴더로 재실행하면 정상 캐시는 재사용합니다. G는 배치 1의 최종 실패 항목을 인수합니다.

`embedding_status.json`에는 요청·성공·실패 수, 캐시 hit/miss, 배치 기록, 처리시간이 들어갑니다. wall_ms는 모델 준비·원본 해시 검사·최종 직렬화/공개를 제외한 이미지/벡터 처리 구간이며 read_crop_ms와 encode_success_ms도 제공합니다. G의 전체 단계 시간은 호출 전후로 별도 측정합니다.

출력은 임시 폴더에서 검사하고 `index_manifest.json`을 마지막에 공개합니다. 기존 결과가 있으면 덮어쓰지 않습니다. CLI 종료 코드는 정상 0, 입력/검사 오류 2, 항목 처리 또는 내부 오류 1입니다. C의 오류 파일을 다른 단계의 파일과 합치는 작업은 G가 수행합니다.

## 검사와 인수인계

자동검사는 경계·소수 좌표 crop, 빈 입력, 프레임 불일치, 입력 해시/경로, 캐시 손상/무효화, 재시도, OOM 주입, 잘못된 벡터, 행 대응과 결과 손상을 확인합니다. 실제 CLIP 검증은 `tools.verify_embedding`에서 별도로 수행합니다. 검사 입력·기대값·실측값·PASS·확인자는 실행 폴더의 `verification_report.json`에 기록합니다.

검사 실측과 아직 연결하지 않은 항목은 `docs/C_검증결과.md`, D·G 전달 절차와 발표 대본은 `docs/C_인수인계.md`에 정리했습니다. 실제 CCTV와 B 디코더, A 최종 samples 연결·1분 영상 성능 측정·다른 팀원의 인수 실행은 해당 입력이 준비되면 수행해야 합니다.

참조: [최종 명세서의 CLIP 공식 구현](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6).
