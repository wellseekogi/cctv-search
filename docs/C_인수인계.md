# C 담당 인수인계

C는 A가 확정한 저장 후보를 이미지와 512차원 CLIP 벡터로 변환합니다. 검색 품질은 실제 영상/질의로 별도 평가하며, 현재 예제 결과는 데이터셋 성능 지표가 아닙니다.

## A와 B에서 받을 자료

- A: schema cctv-3.0 samples, event_ids·reasons·policy·source_run_id·track_source_id, 실제 frame_index·timestamp_ms.
- B: README 계약의 RGB 프레임 함수, 실제 PTS/offset, 디코더·버전, 원본 해상도.
- F/G: manifest, 원본 위치와 SHA256, split, 원본 사용 가능 구간, 실행 ID와 환경.

반환 frame_index/시각이 samples와 다르면 C는 FRAME_MISMATCH로 거부합니다. A/B가 선택 프레임을 확인한 뒤 다시 실행합니다. 다른 프레임을 crop하는 방식으로 우회하지 않습니다.

## D와 F에 전달할 것

`items.jsonl`, `vectors.npy`, `index_manifest.json`, `media` 전체, `embedding_config.json`, `embedding_status.json`, `errors.jsonl`을 같은 출력 폴더로 전달합니다. D는 `validate_index(output, encoder.metadata)`를 먼저 실행합니다. 동일 모델·전처리를 쓰는 `ClipEncoder.encode_text`와 `encode_image`를 재사용합니다.

최종 소스 기준 검증 예제는 `runs/c-verified-cpu-20261103/cold`와 `runs/c-verified-gpu-20261103/cold`입니다. 각 검증 폴더의 image_queries.npy·text_queries.npy는 이미지 2장과 영어 문장 2개의 벡터입니다. 원본 프레임은 fixtures/embedding에 있습니다. object의 여백 전 bbox와 전체 프레임 경로를 E에게 그대로 전달합니다.

F는 성공한 items와 A samples를 item_id로 대조해 encoding_missing을 분류할 수 있습니다. errors의 recovered=true는 재시도 후 성공했다는 뜻이며 최종 실패 수와 구분합니다.

## G에 전달할 것

코드 embedding/, configs/embedding.yaml, requirements-embedding.txt, fixtures/embedding, tests, 실행/검증 도구와 README를 전달합니다. Git 저장소가 제공되지 않아 커밋은 만들지 않았습니다. 납품 ZIP의 source_hashes.json으로 전달 코드의 SHA256을 확인할 수 있습니다.

명령, 입력 해시, 모델/전처리 식별, 실제 버전과 검사 결과는 README 및 verification_report.json에 있습니다. weights는 공식 URL에서 받으며 SHA256을 확인합니다. 원본 CCTV와 가중치 자체는 소스 ZIP에서 제외합니다. 아직 생성하지 않은 공통 requirements.lock, pipeline, 검색 API, UI는 각 담당자 범위입니다.

모든 소비자는 다음 작업일 이내에 예제 명령을 실행하고 실행 ID·확인자·실제 결과·PASS/FAIL을 남깁니다. 오류는 재현 명령과 errors를 C에 돌려줍니다. 실제 A/B 연결 확인일은 입력을 받은 다음 작업일로 잡으며, 다른 팀원의 확인은 아직 수행되지 않았습니다.

## 발표 슬라이드용 내용 2장

### 1장 이미지에서 벡터까지

- 입력: A samples와 B RGB 프레임. 선택 프레임·원본 크기·offset 검사.
- scene 전체 프레임, object 양쪽 10% 여백. floor/ceil과 경계 clipping.
- 공식 CLIP ViT-B/32 전처리, float32 512차원 L2 벡터.
- 원래 bbox와 실제 crop_bbox를 함께 보관해 화면과 평가에 연결.

### 2장 오류가 있어도 행 대응 유지

- items N행 ↔ vectors [N,512], vector_row 0..N-1. 실패 항목은 오류 파일에 분리.
- 해시 기반 캐시, 손상 검출, 모델/전처리 변경 시 무효화.
- 한 번 재시도, 메모리 부족 시 16→8→4→1, 배치 1 최종 실패 기록.
- 실제 수치와 실행 ID는 C_검증결과.md와 검증 보고서를 인용.

### 약 45초 대본

“C는 선택된 프레임을 검색 가능한 벡터로 만드는 역할입니다. 전체 장면은 그대로 저장하고 객체는 양쪽으로 10퍼센트 여백을 더해 잘랐습니다. 같은 CLIP 모델로 이미지와 텍스트를 512차원 벡터로 바꾸고 길이를 1로 정규화했습니다. 실패한 항목을 빼더라도 이미지 메타데이터와 벡터 행이 정확히 맞도록 했고, 오류는 별도 기록합니다. 캐시는 영상과 모델, 전처리의 해시를 확인한 뒤 재사용합니다. 메모리가 부족하면 배치를 줄이고 재시도합니다. 현재 작은 예제에서 생성과 질의 인코딩을 확인했으며 실제 영상 연결 검사가 다음 단계입니다.”

### 예상 질문과 답변

Q. 모델을 바꾸고 예전 캐시를 그대로 써도 되나요?

A. 안 됩니다. 모델 ID·리비전·가중치·전처리 등이 벡터 캐시 키에 들어가 새 벡터를 생성합니다. D도 질의 인코더와 인덱스의 식별을 검사합니다.

Q. 메모리 부족이나 crop 실패가 있으면 벡터 번호가 어긋나지 않나요?

A. 성공 항목만 순서대로 다시 번호를 매깁니다. 오류 기록과 최종 실패 수를 남기고, 실패가 있는 묶음은 기본 검증에서 검색용으로 승인하지 않습니다.
