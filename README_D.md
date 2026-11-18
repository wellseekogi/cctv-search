# D 모듈: 검색 순위 · API · 영상 제공

저장된 CCTV 인덱스에서 텍스트·이미지 질의로 트랙을 찾고, 대표 시각과 재생 구간을 붙여 API 로 제공합니다.
안내서(버전 5.0) D 1~D 4 를 구현했습니다. 담당 폴더는 `retrieval/` 과 `api/` 입니다.

> **현재 상태**: C·F·B 의 실제 자료 대신 `fixtures/d_small` (벡터 10개, 4차원) 과 검사용 인코더로 확인했습니다.
> 실제 CLIP 인코더·실제 인덱스 연결은 "남은 작업" 을 보세요.

## 1. 설치와 실행

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-d.txt

# 준비 상태 검사 (정상 0, 미준비 2)
.\.venv\Scripts\python.exe -m retrieval check --config configs/retrieval.yaml

# API 서버 (127.0.0.1:8000, worker 1)
.\.venv\Scripts\python.exe -m api --config configs/retrieval.yaml

# 질의 목록 전체의 트랙 순위 저장
.\.venv\Scripts\python.exe -m retrieval predict --config configs/retrieval.yaml --gallery gallery_dev `
    --queries fixtures/d_small/queries.jsonl --out runs/d_small/predictions.jsonl --representation all

# 자동 검사
.\.venv\Scripts\python.exe -m pytest tests -q
```

exit code 는 정상 0, 입력·검사 오류 2, 내부 오류 1 입니다(G 5).

## 2. 입력 파일

| 파일 | 작성자 | D 가 쓰는 필드 |
|---|---|---|
| `<run_dir>/items.jsonl` | C | item_id · video_id · representation · track_uid · bbox · vector_row · timestamp_ms/actual_timestamp_ms · media_path · full_frame_path |
| `<run_dir>/vectors.npy` | C | float32 [N, d], L2 정규화 |
| `<run_dir>/index_manifest.json` | C | files.{items.jsonl,vectors.npy}.{sha256,rows} · embedding_dim · dtype · normalized · model_id · model_revision · preprocess_hash · index_version |
| `manifest.jsonl` | F | video_id · relative_path · width · height · duration_ms · source_offset_ms · available_start_ms · available_end_ms · split |
| `tracks.jsonl` | B(pred) 또는 F(GT) | 관측 행 전체 (2.1 참고). scene↔track 연결, scene 대표 박스, 결과의 class_name·confidence·quality_score |
| `queries.jsonl` | F | query_id · modality · text · image_path · gallery_id · split |

서버와 CLI 는 시작할 때 인덱스를 검사합니다. 하나라도 어긋나면 검색하지 않고 `INDEX_NOT_READY`(503) 로 응답하며, 사유는 `GET /health` 의 `galleries[].problems` 에 나옵니다.

- items.jsonl · vectors.npy 의 SHA256 과 행 수가 index_manifest 와 같은지
- items N행 = vectors [N, d], vector_row 가 0부터 연속, item_id 고유
- dtype float32, 모든 값 유한, `abs(norm-1) ≤ 0.0001`
- 질의 인코더의 model_id · model_revision · preprocess_hash · 차원이 인덱스와 같은지
- 관측 파일 검사 통과, object 항목의 track_uid 가 모두 관측 파일에 있는지 (2.1)

### 2.1 관측 파일 형식 (회의에서 정한 B 출력)

한 줄이 "한 영상의 한 시각(200ms 간격)에 검출된 객체 하나"입니다. A 는 이 행으로 event 와 저장 후보를 정하고, C 는 확정된 후보로 이미지·벡터를 만듭니다. D 는 같은 파일을 다시 읽어 scene 을 트랙에 연결합니다.

```json
{"schema_version": "cctv-3.0", "source": "pred", "track_source_id": "det01_b_colab_001_000",
 "video_id": "cityflow_S02_c007", "frame_index": 0, "timestamp_ms": 0, "local_track_id": "t000001",
 "track_uid": "det01_b_colab_001_000:cityflow_S02_c007:t000001", "class_name": "car",
 "bbox": [1325.21435546875, 370.3994140625, 1668.09130859375, 504.34686279296875],
 "confidence": 0.8827624320983887,
 "quality": {"area_ratio": 0.022148675298789677, "blur_var": 4652.25025543198, "border_touch": false,
             "score": 0.7109654215527837}}
```

안내서 B 3 예시와 다른 점은 `quality.score` 하나입니다. A 4 품질식 `0.5·min(area_ratio/0.05, 1) + 0.5·blur_var/(blur_var+100)` (경계 접촉 시 ×0.8) 의 값이며, 위 행은 이 식과 정확히 일치합니다. 이 행은 `fixtures/b_sample/tracks_sample.jsonl` 에 그대로 넣어 검사합니다.

| 구분 | 검사 내용 | 처리 |
|---|---|---|
| 오류 | JSON 형식, schema_version, source(pred·gt), track_uid = `track_source_id:video_id:local_track_id`, timestamp_ms 정수, bbox 숫자 4개·x1<x2·y1<y2, pred confidence 0~1, score 0~1 | 인덱스 미준비 (503) |
| 오류 | 같은 video·track·시각 중복, pred·gt 혼합, index_manifest.track_source 와 출처 불일치, 관측 파일 없음, items 의 track_uid 가 관측에 없음 | 인덱스 미준비 (503) |
| 경고 | score 가 품질식과 1e-6 넘게 다름, area_ratio 가 bbox 면적/영상 면적과 다름, bbox 가 영상 범위 밖, 트랙 안 시각 역순(다시 정렬), gallery 밖 영상의 관측 | 검색 가능, `/health` 의 `warnings` 에 표시 |
| 보완 | score 가 없는 행 | D 가 품질식으로 계산 (`observations.score_recomputed_rows`) |

오류·경고는 종류별로 한 줄에 모아 `tracks.jsonl: track_uid 가 ... (3행, 예: 4·9·12행)` 처럼 행 번호를 보여 줍니다. `/health` 의 `galleries[].observations` 에 행 수·트랙 수·출처·track_source_id·클래스별 행 수가 나옵니다.

## 3. 순위 계산 (D 1)

1. 영상 필터(`video_ids`)를 먼저 적용합니다. 생략하면 gallery 전체, `[]` 이면 후보 0개입니다.
2. `score = q · v` (정규화 벡터의 내적).
3. 같은 트랙의 object 점수는 최댓값 하나로 모읍니다.
4. scene 은 같은 영상에서 시각 차이 100ms 이내에 관측이 있는 트랙에 연결하고, 트랙별 scene 점수도 최댓값으로 모읍니다.
5. 표현별 순위는 점수 내림차순, 동률은 track_uid 사전순입니다.
6. combined 는 `1/(60+scene 순위) + 1/(60+object 순위)` 이고, 없는 항은 0, 동률은 track_uid 사전순입니다.
7. 대표 항목은 최고 object, 없으면 최고 scene 입니다. 재생 구간은 대표 시각 ±1000ms 를 `[available_start_ms, available_end_ms]` 로 맞춥니다.

API 는 상위 K 만, `predictions.jsonl` 은 전체 트랙 순위를 저장합니다.

### 손계산 예제 (질의 "a white car" = [1,0,0,0])

| 항목 | 표현 | 트랙 / 연결 트랙 | 점수 |
|---|---|---|---|
| obj001 @1200 | object | v001:t007 | 0.80 |
| obj002 @0 | object | v001:t007 | 0.60 |
| obj003 @600 | object | v001:t008 | 0.80 |
| obj004 @400 | object | v002:t001 | 0.96 |
| obj005 @4000 | object | v002:t002 | 0.28 |
| scn001 @1200 | scene | v001: t007·t008·t009 | 0.60 |
| scn002 @2600 | scene | v001: t007·t008 | 1.00 |
| scn003 @400 | scene | v002: t001 | 0.00 |
| scn004 @2500 | scene | (연결 트랙 없음) | 0.00 |
| scn005 @4000 | scene | v002: t002 | 0.60 |

| 트랙 | object 최댓값 → 순위 | scene 최댓값 → 순위 | 결합 점수 | 최종 |
|---|---|---|---|---|
| v001:t007 | 0.80 → 2 (t008 과 동점, 사전순) | 1.00 → 1 | 1/61 + 1/62 = 0.032522 | 1 |
| v001:t008 | 0.80 → 3 | 1.00 → 2 | 1/62 + 1/63 = 0.032002 | 2 |
| v002:t001 | 0.96 → 1 | 0.00 → 5 | 1/65 + 1/61 = 0.031778 | 3 |
| v002:t002 | 0.28 → 4 | 0.60 → 4 | 1/64 + 1/64 = 0.031250 | 4 |
| v001:t009 | 없음 → 0 | 0.60 → 3 | 1/63 = 0.015873 | 5 |

1위 t007 의 대표 항목은 obj001(1200ms), 재생 구간은 200~2200ms, 박스는 [172,200,372,360] 으로 안내서 D 3 예제와 같습니다.

## 4. API (D 2 · D 3)

| 요청 | 설명 |
|---|---|
| `POST /search` | 텍스트 검색. `application/json` |
| `POST /search/image` | 이미지 검색. `multipart/form-data` (image 1개 + gallery_id · top_k · representation · video_ids) |
| `GET /health` | 모델·인덱스 준비 상태 |
| `GET /videos/{video_id}` | 영상 크기·길이·offset·재생 URL (`?gallery_id=` 선택) |
| `GET /media/{media_id}` | 등록한 이미지·영상. HTTP Range 지원 |

```json
{"modality":"text","text":"a white car","top_k":10,
 "gallery_id":"gallery_dev","video_ids":["v001"],"representation":"combined"}
```

- `text`: 앞뒤 공백 제거·연속 공백 한 칸 처리 후 1~500자, 모델 토큰 길이 이하
- `top_k`: 1~50 정수 (기본 10), `representation`: scene · object · combined (기본 combined)
- 이미지: JPEG·PNG, 5,000,000 bytes 이하, 1600만 픽셀 이하, 가로·세로 각 32픽셀 이상. EXIF 회전 후 RGB 변환
- 이미지 검색의 `video_ids` 는 JSON 배열 문자열(`["v001"]`)

응답의 `results[]` 는 안내서 D 3 의 필드를 모두 포함합니다. 아래는 추가한 필드입니다(기존 필드는 그대로라 무시해도 됩니다).

| 추가 필드 | 내용 |
|---|---|
| `results[].crop_url` | 객체 crop 이미지. `thumbnail_url` 은 박스를 그릴 수 있게 **전체 프레임**을 가리킵니다 |
| `results[].item_representation` | 대표 항목이 object 인지 scene 인지 |
| `results[].class_name` · `confidence` · `quality_score` | 대표 시각(±100ms)에 가장 가까운 해당 트랙 관측의 값. GT 관측이면 confidence 는 null |
| `results[].object_rank` · `scene_rank` · `object_score` · `scene_score` | 결합 전 순위와 점수 |
| `scene_results[]` | 장면 결과 목록(`result_type="scene"`, `track_uid=null`, `linked_track_uids`). representation 이 scene·combined 일 때 상위 K |

### 상태와 오류

| 상태 | error.code | 발생 조건 |
|---|---|---|
| 200 | - | 결과 목록 또는 `results=[]` |
| 422 | INVALID_INPUT | 빈 text, top_k 범위·형식, representation, video_ids 형식, JSON 오류 |
| 422 | TEXT_TOO_LONG | 500자 초과 또는 모델 토큰 길이 초과 |
| 413 | IMAGE_TOO_LARGE | 5,000,000 bytes 초과 또는 1600만 픽셀 초과 |
| 415 | INVALID_IMAGE | JPEG·PNG 아님, 손상, 한 변 32픽셀 미만 |
| 404 | UNKNOWN_GALLERY · UNKNOWN_VIDEO · UNKNOWN_MEDIA | 등록되지 않은 ID |
| 503 | INDEX_NOT_READY | 모델 또는 인덱스 미준비 |
| 500 | INTERNAL_ERROR | 그 밖의 내부 오류 |

오류 본문은 `{"schema_version","request_id","error":{"code","message"}}` 입니다. 모든 응답에 `X-Request-ID` 헤더가 붙고, 같은 ID 로 `logs/api.jsonl` 에 한 줄씩 기록됩니다.
검사 순서는 입력 형식(422) → gallery(404) → 준비 상태(503) → video_ids(404) 입니다.

## 5. 설정

`configs/retrieval.yaml` 에 D 4 의 값과 gallery 등록 정보가 있습니다.

- 실제 인코더 연결: `encoder.factory` 를 `retrieval.encoder:FunctionEncoder`, `options.module` 을 C 의 모듈 경로로 바꿉니다.
- gallery 추가: `galleries.<gallery_id>` 에 run_dir · data_root · manifest · tracks 를 등록합니다.
- 재인코딩한 재생용 파일: `playback_overrides.<video_id>` 에 path · source_offset_ms · codec · sha256 을 등록합니다.

## 6. 출력 파일

| 파일 | 내용 | 받는 사람 |
|---|---|---|
| `runs/<run>/predictions.jsonl` | 질의·표현별 전체 트랙 순위. 한 줄에 (질의, 트랙) 하나 | F · G |
| `runs/<run>/predictions_summary.json` | status · 질의 수 · 행 수 · 사용한 순위 설정 | F · G |
| `docs/d_handoff/examples/*.json` | 정상·빈 결과·오류 응답 예제 16개 | E |
| `docs/d_handoff/check_log.txt` | 자동 검사 목록과 실제 서버 확인 기록 | F · G |
| `logs/api.jsonl` | request_id 별 요청 로그 | D |

predictions 한 줄: `schema_version · query_id · gallery_id · index_version · representation · score_type · rank · track_uid · video_id · score · item_id · representative_ms · start_ms · end_ms · object_rank · scene_rank`

## 7. 검사 결과

입력은 모두 `fixtures/d_small` 입니다. 자동 검사 93건이 통과했고 상세 목록은 `docs/d_handoff/check_log.txt` 에 있습니다. 확인자는 직접 실행한 뒤 적어 주세요.

| 검사 | 입력 | 기대값 | 실측값 | 결과 | 확인자 |
|---|---|---|---|---|---|
| 벡터 10개 손계산 | 질의 [1,0,0,0] | 3장의 표 | 표와 일치 | PASS | |
| 동점 | t007·t008 object 0.80 | track_uid 사전순 | t007 → t008 | PASS | |
| 중복 트랙 | t007 object 2개 | 최댓값 0.80 한 건 | 0.80, 대표 obj001 | PASS | |
| 표현 누락 | t009 object 없음 | 1/63 = 0.015873 | 0.015873 | PASS | |
| scene 연결 경계 | 시각 차 100 / 101ms | 연결 / 미연결 | 연결 / 미연결 | PASS | |
| 재생 구간 | 대표 1200 / 400 / 4000ms | 200~2200 / 0~1400 / 3000~4800 | 같음 | PASS | |
| 빈 필터 | `video_ids: []` | 200, `results=[]` | 200, `[]` | PASS | |
| 미준비 상태 | 손상된 vectors.npy | /health not_ready, 검색 503 | 같음 | PASS | |
| 입력 오류 | text·top_k·representation 13종 | 422 INVALID_INPUT | 422 | PASS | |
| 텍스트 길이 | 500자 / 501자 | 200 / 422 TEXT_TOO_LONG | 같음 | PASS | |
| 이미지 조건 | 5MB 초과, 4001×4000, 31×64, GIF, 손상 | 413·413·415·415·415 | 같음 | PASS | |
| 재현성 | 같은 요청 2회, predict 2회 | 순위·구간·상태 코드·파일 동일 | 동일 | PASS | |
| HTTP Range | `bytes=100-199`, `-50`, 범위 밖 | 206·206·416 | 같음 | PASS | |
| MP4 재생 | 서버 URL 을 ffprobe·ffmpeg 로 열기 | h264, 1.2초 탐색 성공 | 성공 | PASS | |
| 팀 관측 행 | `fixtures/b_sample/tracks_sample.jsonl` | 오류·경고 0, score 0.7109654 | 같음 | PASS | |
| 품질식 | 안내서 A 4 예제 (blur 100 / 900, 경계 접촉) | 0.404321 / 0.604321 / ×0.8 | 같음 | PASS | |
| 관측 오류 | track_uid 구성·bbox·시각·출처·중복·파일 없음 등 12종 | 인덱스 미준비 | 12종 모두 | PASS | |
| 관측 경고 | score·area_ratio 불일치, 범위 밖 박스, 역순 | 검색 가능 + 경고 | 같음 | PASS | |
| 결과 관측 정보 | t007 대표 obj001 @1200ms | car · 0.9 · 0.404321 | 같음 | PASS | |
| 처리량 | 카메라 4대, 관측 126,959행 | 시작 시 몇 초 이내 | 읽기 1.3초, scene 4,600개 연결 0.9초 | PASS | |

## 8. 남은 작업과 확인이 필요한 결정

**남은 작업 (다른 담당의 자료가 필요)**

1. C 의 실제 인코더·인덱스 연결. `FunctionEncoder` 가 기대하는 상수 이름(MODEL_ID 등)과 `count_tokens` 제공 여부, index_manifest 의 키 이름을 C 와 맞춰야 합니다.
2. F 의 실제 queries·gallery 로 dev predictions 생성 후 F 의 지표 손계산과 대조.
3. E 화면에서 브라우저 `<video>` 재생과 seek 확인. 지금은 ffprobe·ffmpeg 로만 확인했습니다.
4. dev 검증 후 최종 순위 설정을 G 에게 전달.

**안내서에 명시되지 않아 D 가 정한 것 (팀 확인 필요)**

| 항목 | 정한 내용 | 확인할 사람 |
|---|---|---|
| "track_uid=null 인 scene 은 장면 결과 목록으로 제공" | `scene_results[]` 를 `results[]` 와 따로 반환 | E · F |
| predictions 형식 | (질의, 트랙) 한 줄, 표현별로 구분 | F · G |
| index_manifest 키 이름 | 2장의 표 | C |
| `thumbnail_url` | 전체 프레임을 가리키고 crop 은 `crop_url` 로 분리 | E |
| 한 변 32픽셀 미만 이미지 | 415 INVALID_IMAGE | E |
| 미디어 404 | `UNKNOWN_MEDIA` 코드 추가 | E |
| 순위에 들어가는 트랙 | 인덱스에 항목이 하나 이상 있는 트랙만 | F |
| `quality.score` | 관측 형식의 공식 필드로 등록, 없으면 D 가 A 4 식으로 계산 | A · B |
| 관측 파일 오류·경고 기준 | 2.1 의 표 | B · F |
