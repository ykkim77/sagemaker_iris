# 11주차 실습(재개정): SageMaker SDK 3.x Iris 파이프라인·모델 등록·S3 추론 코드 연결·실시간 배포·추론


## 1. 실습 목표

- Kubeflow Pipelines와 SageMaker Pipelines의 컴포넌트/Step, 아티팩트/S3, Run/Execution을 비교함.
- Iris 데이터셋을 전처리·학습·평가하고 정확도가 기준 이상일 때만 모델 레지스트리에 등록함.
- 서로 다른 `NEstimators` 및 `AccuracyThreshold`로 3회 실행하고 실행별 평가 결과와 등록 여부를 비교함.
- 등록된 특정 모델 버전을 SageMaker Studio GUI에서 수동 승인하고, 학습된 모델을 재사용하여 추론 스크립트를 포함한 일반 Real-time Endpoint에 배포·추론·삭제함.

## 2. 환경 및 사전 점검 (필수)

| 항목 | 권장 설정 / 검증 항목 |
|---|---|
| 계정 | SageMaker Studio, Processing, Training, Model Registry, Real-time Inference 생성 권한 확인 |
| Studio | JupyterLab 스페이스 `ml.t3.medium` (사용 가능할 때) |
| 작업 인스턴스 | Processing/Training `ml.m5.large` 예시. 리전별 지원 및 계정별 할당량 확인 후 변경 |
| SDK | `sagemaker==3.26.0`을 예시 검증 후보로 고정. 설치 가능성과 API 동작 사전 검증 필요 |
| 이미지 | SageMaker scikit-learn `1.2-1` (공식 v3 예제의 이미지 검색 패턴 기준). 오래된 이미지이므로 수업 전 이미지 URI 조회·실행 테스트 필수 |
| IAM | SageMaker 실행 역할의 S3, ECR, CloudWatch, SageMaker 작업 생성·모델 등록·배포 권한 확인 |


## 3. 비용 관리

- Processing 2회 + Training 1회가 파이프라인 실행마다 발생함. 기본 3회 실행 시 총 9개 작업임.
- `ml.m5.large`는 작업 시간만큼 과금되지만 Studio JupyterLab은 별도로 실행 시간 과금됨.
- 이번 개정 실습에서는 Serverless Inference를 사용하지 않음. 이전 실습에서 서버리스 초기화 180초 제한 오류가 발생했음.
- Real-time Endpoint는 `ml.m5.large` 인스턴스가 상시 실행되므로 **추론 후 즉시 삭제**해야 함. 기존 GUI 배포 Endpoint/Inference Component도 별도로 정리함.
- 실습 시작 전에 AWS Budgets 경보를 설정하고, 과금·크레딧 현황을 확인함. 예산 경보가 서비스를 자동으로 중단하지는 않음.
- 가격은 리전·시점에 따라 달라지므로 수업자료에 고정된 금액을 확정 비용으로 기재하지 않음.

## 4. Step 1 — Studio 접속 및 SDK 설치

AWS 콘솔 → SageMaker AI → Studio 도메인 생성(계정에서 지원하는 설정 경로 사용) → JupyterLab 스페이스 생성 → 노트북 실행.

```python
%pip install "sagemaker==3.26.0" "boto3>=1.35,<2" "scikit-learn>=1.4,<2" pandas joblib
```

커널을 재시작한 후:

```python
import boto3
from importlib.metadata import version

# 설치된 배포 패키지 메타데이터로 버전 조회 (SDK 3.x 호환)
sdk_version = version("sagemaker")
print("SageMaker SDK:", sdk_version)
print("boto3:", version("boto3"))
assert sdk_version.startswith("3."), "SageMaker SDK 3.x가 필요함"

aws = boto3.Session()
print("Region:", aws.region_name)
print("Account:", aws.client("sts").get_caller_identity()["Account"])
assert aws.region_name, "AWS 리전이 지정되지 않음"
```



## 5. Step 2 — 실습 스크립트 작성

### (1) 전처리 `preprocess.py`

```python
%%writefile preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "/opt/ml/processing"
df = pd.read_csv(f"{BASE}/input/iris.csv")
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"]
)
for folder, data, filename in [
    ("train", train, "train.csv"),
    ("test", test, "test.csv"),
]:
    os.makedirs(f"{BASE}/{folder}", exist_ok=True)
    data.to_csv(f"{BASE}/{folder}/{filename}", index=False)
print(f"train={len(train)}, test={len(test)}")
```

### (2) 학습 `train.py`

```python
%%writefile train.py
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SageMaker scikit-learn 추론 컨테이너가 사용하는 모델 로딩 함수
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

# CSV 추론 요청을 명시적으로 파싱함
def input_fn(request_body, request_content_type):
    import io
    if request_content_type != "text/csv":
        raise ValueError(f"지원하지 않는 ContentType: {request_content_type}")
    return pd.read_csv(io.StringIO(request_body), header=None).values

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept != "text/csv":
        raise ValueError(f"지원하지 않는 Accept: {accept}")
    return "\n".join(map(str, prediction.tolist())) + "\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.train, "train.csv"))
    X, y = df.drop(columns=["target"]), df["target"]
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    model.fit(X.values, y.values)
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("n_estimators:", args.n_estimators)
```

### (3) 평가 `evaluate.py`

```python
%%writefile evaluate.py
import json
import os
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

BASE = "/opt/ml/processing"
archive = f"{BASE}/model/model.tar.gz"
# 전체 아카이브를 추출하지 않고 필요한 파일만 안전하게 읽음
# model.joblib은 본 실습의 신뢰 가능한 학습 작업에서 생성된 아티팩트여야 함
with tarfile.open(archive, "r:gz") as tar:
    member = tar.getmember("model.joblib")
    if not member.isfile():
        raise ValueError("model.joblib이 일반 파일이 아님")
    with tar.extractfile(member) as f:
        model = joblib.load(f)
df = pd.read_csv(f"{BASE}/test/test.csv")
accuracy = float(accuracy_score(df["target"], model.predict(df.drop(columns=["target"]).values)))
report = {
    "metrics": {"accuracy": {"value": accuracy}},
    "multiclass_classification_metrics": {
        "accuracy": {"value": accuracy, "standard_deviation": "NaN"}
    }
}
os.makedirs(f"{BASE}/evaluation", exist_ok=True)
with open(f"{BASE}/evaluation/evaluation.json", "w") as f:
    json.dump(report, f)
print("accuracy:", accuracy)
```

실습 전 주의: `joblib.load()`는 신뢰하지 않는 모델 파일에 사용하면 안 됨. 본 실습에서 직접 생성한 SageMaker 학습 아티팩트에만 적용함.

## 6. Step 3 — 세션·리전·S3 준비

```python
import boto3
from sklearn.datasets import load_iris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core import image_uris

aws_session = Session()
pipeline_session = PipelineSession()
role = get_execution_role()
region = aws_session.boto_region_name
bucket = aws_session.default_bucket()
account = boto3.client("sts").get_caller_identity()["Account"]
print("region:", region, "bucket:", bucket, "role:", role)

load_iris(as_frame=True).frame.to_csv("iris.csv", index=False)
input_uri = aws_session.upload_data("iris.csv", bucket=bucket, key_prefix="week11-iris/raw")
print(input_uri)
```

## 7. Step 4 — 컨테이너 URI 확인

```python
FRAMEWORK = "1.2-1"   # 공식 v3 예제에서 사용된 이미지 버전; 최신 지원 여부 별도 확인
INSTANCE_TYPE = "ml.m5.large"  

sklearn_image = image_uris.retrieve(
    framework="sklearn", region=region, version=FRAMEWORK,
    py_version="py3", instance_type=INSTANCE_TYPE
)
print("scikit-learn image:", sklearn_image)
```

이미지 URI 조회 성공은 실행 성공을 보장하지 않음. 이미지의 Python·scikit-learn·joblib 버전 및 처리 작업 시작 여부까지 검증함.

## 8. Step 5 — 파라미터와 전처리 ProcessingStep

```python
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput, ProcessingS3Input,
    ProcessingOutput, ProcessingS3Output,
)
from sagemaker.core.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep

n_estimators = ParameterString(
    name="NEstimators",
    default_value="100"
)
accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.85
)


processor = ScriptProcessor(
    image_uri=sklearn_image,
    command=["python3"],
    role=role,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    sagemaker_session=pipeline_session,
)

prefix = "week11-iris/pipeline"
prep_args = processor.run(
    code="preprocess.py",
    inputs=[ProcessingInput(
        input_name="iris-input",
        s3_input=ProcessingS3Input(
            s3_uri=input_uri,
            local_path="/opt/ml/processing/input",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    )],
    outputs=[
        ProcessingOutput(
            output_name=name,
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/{prefix}/{name}",
                local_path=f"/opt/ml/processing/{name}",
                s3_upload_mode="EndOfJob",
            ),
        ) for name in ("train", "test")
    ],
)
step_prep = ProcessingStep(name="Preprocess", step_args=prep_args)
```

## 9. Step 6 — 학습 TrainingStep

```python

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, SourceCode
from sagemaker.mlops.workflow.steps import TrainingStep

import os

os.environ["SAGEMAKER_VALIDATE_ROLE"] = "false"

trainer = ModelTrainer(
    training_image=sklearn_image,
    source_code=SourceCode(
        source_dir=".",
        entry_script="train.py"
    ),
    role=role,
    compute=Compute(
        instance_type=INSTANCE_TYPE,
        instance_count=1
    ),
    hyperparameters={
        "n_estimators": n_estimators
    },
    input_data_config=[
        InputData(
            channel_name="train",
            data_source=step_prep.properties
                .ProcessingOutputConfig
                .Outputs["train"]
                .S3Output.S3Uri,
            content_type="text/csv"
        )
    ],
    sagemaker_session=pipeline_session
)

train_args = trainer.train()

step_train = TrainingStep(
    name="Train",
    step_args=train_args
)

```

`SourceCode(source_dir=".", entry_script="train.py")`는 SDK 3.x 공식 문서에 기재된 방식임. 현재 디렉터리에 `train.py`가 존재하는지 확인하고, 학습 컨테이너에서 사용되는 Python 및 scikit-learn 버전은 사전 테스트함.

```python
from pathlib import Path
assert Path("train.py").is_file(), "train.py 파일을 먼저 생성해야 함"
```

**주의:** 위 Step 6 코드에서 `train_args`와 `step_train`을 이미 생성하였으므로 같은 셀을 다시 실행하지 않습니다. Notebook 중간부터 재실행해 객체가 뒤섞이면 커널을 재시작하고 Step 3부터 순서대로 재구성하십시오.

## 10. Step 7 — 평가 ProcessingStep

```python
from sagemaker.core.workflow.properties import PropertyFile

eval_report = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")

eval_args = processor.run(
    code="evaluate.py",
    inputs=[
        ProcessingInput(
            input_name="model",
            s3_input=ProcessingS3Input(
                s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                local_path="/opt/ml/processing/model",
                s3_data_type="S3Prefix", s3_input_mode="File",
            ),
        ),
        ProcessingInput(
            input_name="test",
            s3_input=ProcessingS3Input(
                s3_uri=step_prep.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                local_path="/opt/ml/processing/test",
                s3_data_type="S3Prefix", s3_input_mode="File",
            ),
        ),
    ],
    outputs=[ProcessingOutput(
        output_name="evaluation",
        s3_output=ProcessingS3Output(
            s3_uri=f"s3://{bucket}/{prefix}/evaluation",
            local_path="/opt/ml/processing/evaluation",
            s3_upload_mode="EndOfJob",
        ),
    )],
)
step_eval = ProcessingStep(name="Evaluate", step_args=eval_args, property_files=[eval_report])
```

## 11. Step 8 — 모델 등록 및 정확도 조건 분기

```python

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet

MODEL_GROUP = "week11-iris-models"

# 모델 등록 설정
model_builder = ModelBuilder(
    s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    image_uri=sklearn_image,
    role_arn=role,
    sagemaker_session=pipeline_session,
)

register_args = model_builder.register(
    model_package_group_name=MODEL_GROUP,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=[INSTANCE_TYPE],
    approval_status="PendingManualApproval",
)

step_register = ModelStep(
    name="RegisterModel",
    step_args=register_args
)

# 정확도 조건 분기
step_condition = ConditionStep(
    name="CheckAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=step_eval.name,
                property_file=eval_report,
                json_path="metrics.accuracy.value",
            ),
            right=accuracy_threshold,
        )
    ],
    if_steps=[step_register],
    else_steps=[],
)

```


## 12. Step 9 — 파이프라인 정의 및 실행

```python
from sagemaker.mlops.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name="week11-iris-sdk3",
    parameters=[n_estimators, accuracy_threshold],
    steps=[step_prep, step_train, step_eval, step_condition],
    sagemaker_session=pipeline_session,
)

print(pipeline.definition()[:1500])
pipeline.upsert(role_arn=role)
execution1 = pipeline.start()
execution1.wait(delay=30, max_attempts=90)
print(execution1.list_steps())
```



## 13. Step 10 — 파라미터 변경 실행과 비교

```python
execution2 = pipeline.start(parameters={"NEstimators": "5"})
execution3 = pipeline.start(parameters={"AccuracyThreshold": 0.999})
for execution in (execution2, execution3):
    execution.wait(delay=30, max_attempts=90)
    print(execution.list_steps())
```

- 실행 1: 기본 트리 100개, 정확도 기준 0.85
- 실행 2: 트리 5개, 정확도 기준 0.85
- 실행 3: 트리 100개, 정확도 기준 0.999 (등록 생략 가능)

중요: 레지스트리에 등록된 모델만 조회하면 실행 3처럼 조건이 거짓인 결과는 비교할 수 없음. 각 Execution의 `Evaluate` 단계 S3 결과를 별도로 확인하고 표에 기재함.

| Execution | NEstimators | AccuracyThreshold | Accuracy | 등록 여부 |
|---|---:|---:|---:|---|
| 1 | 100 | 0.85 | 실행 후 기록 | 실행 후 기록 |
| 2 | 5 | 0.85 | 실행 후 기록 | 실행 후 기록 |
| 3 | 100 | 0.999 | 실행 후 기록 | 실행 후 기록 |

## 14. Step 11 — 모델 버전 확인 및 **GUI 수동 승인**

모델 등록은 자동으로 이루어지지만 `PendingManualApproval` 상태에서는 승인된 배포 후보가 아닙니다. **승인 대상은 최신 버전이 아니라 평가 결과를 확인한 뒤 명시적으로 선택한 버전**입니다. 이번 예시는 v1을 사용합니다. 파이프라인을 재실행한 환경에서는 버전 번호가 달라질 수 있습니다.

```python
import boto3
sm = boto3.client("sagemaker", region_name=region)
MODEL_GROUP = "week11-iris-models"

packages = []
for page in sm.get_paginator("list_model_packages").paginate(
    ModelPackageGroupName=MODEL_GROUP,
    ModelPackageType="Versioned",
    SortBy="CreationTime",
    SortOrder="Descending",
):
    packages.extend(page["ModelPackageSummaryList"])

for p in packages:
    print("v", p["ModelPackageVersion"], p["ModelApprovalStatus"], p["ModelPackageArn"])
```

**GUI 승인:** SageMaker AI → Model registry → Model package groups → `week11-iris-models` → 대상 버전(v1) → **Update status / Approve** (콘솔 메뉴명은 화면에 따라 다를 수 있음). Notebook에서는 승인 API를 자동 실행하지 않습니다.

```python
TARGET_VERSION = 1  # GUI에서 직접 승인한 실제 버전 번호
selected = next((p for p in packages if p["ModelPackageVersion"] == TARGET_VERSION), None)
if selected is None:
    raise ValueError(f"v{TARGET_VERSION} 모델을 찾지 못했습니다")
package_arn = selected["ModelPackageArn"]
package = sm.describe_model_package(ModelPackageName=package_arn)
if package["ModelApprovalStatus"] != "Approved":
    raise RuntimeError("GUI에서 해당 버전을 Approved로 변경한 후 다시 실행하십시오")
print("선택된 승인 버전:", TARGET_VERSION)
print("패키지 ARN:", package_arn)
```

> **모델 성능 비교 주의:** `list_model_packages()`에는 정확도가 자동 표시되지 않습니다. 본 실습의 `ModelBuilder.register()`에는 `ModelMetrics`가 연결되어 있지 않으므로 Step 10의 실행별 `evaluation.json`과 등록 결과를 비교해 승인 대상을 결정하십시오.

## 15. Step 12 — 모델 다운로드 없이 추론 스크립트를 연결하여 Real-time Endpoint 배포

### 12-1. 이번 실습의 배포 원리

**학습 모델 `model.tar.gz`는 S3에 그대로 둡니다.** Notebook에서 `inference.py`만 작성하고, 이 작은 스크립트를 `source.tar.gz`로 묶어 S3에 올립니다. SageMaker Scikit-learn 추론 컨테이너는 **기존 모델 S3 URI**와 **추론 코드 S3 URI**를 각각 참조합니다.

- 모델 파일 다운로드·압축 해제·재압축 **없음**
- 학습 Pipeline 재실행 **없음**
- GUI에서 수동 승인한 모델 버전의 **학습 아티팩트를 그대로 사용**
- 일반 Real-time Endpoint (`ml.m5.large`) 사용; Inference Component 불필요

> **정확한 구분:** 이 단계는 `ModelPackageName=package_arn`으로 기존 모델 패키지를 직접 배포하는 것이 아닙니다. 승인된 버전에서 S3 모델 경로와 이미지 정보를 읽어 **별도 SageMaker Model**을 만드는 방식입니다. 따라서 승인된 v1의 모델 파일은 재사용하지만, 생성한 SageMaker Model은 Registry v1에 직접 연결된 배포 리소스가 아닙니다. 모델 버전과 배포 모델의 대응관계는 이름/태그/실습 기록으로 관리하십시오. 모델 패키지 자체를 직접 배포하려면 **등록 시점에** 추론 환경 구성을 포함한 새 버전으로 등록해야 합니다.

### 12-2. `inference.py` 작성

다음 셀을 실행하면 Notebook 작업 폴더에 추론 스크립트가 만들어집니다.

```python
%%writefile inference.py
import io
import os
import joblib
import numpy as np
import pandas as pd

# SageMaker가 내려받아 /opt/ml/model에 배치한 모델 파일을 로드
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

# 요청 CSV: 한 행이 붓꽃 한 송이, 네 열이 측정값 4개
def input_fn(request_body, content_type):
    if content_type.split(";")[0].strip() != "text/csv":
        raise ValueError(f"Unsupported ContentType: {content_type}")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    x = pd.read_csv(io.StringIO(request_body), header=None).to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"Iris 입력은 4개 특성이 필요합니다: {x.shape}")
    return x

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept.split(";")[0].strip() != "text/csv":
        raise ValueError(f"Unsupported Accept: {accept}")
    return "\n".join(str(int(v)) for v in np.asarray(prediction).reshape(-1)) + "\n"
```

### 12-3. 승인된 v1의 S3 모델 경로와 이미지 확인

Step 11에서 `package_arn`, `package`를 만든 후 실행합니다. **모델 파일을 다운로드하지 않습니다.**

```python
import boto3
import json

sm = boto3.client("sagemaker", region_name=region)
s3 = boto3.client("s3", region_name=region)

package = sm.describe_model_package(ModelPackageName=package_arn)
assert package["ModelApprovalStatus"] == "Approved", "승인된 모델 버전만 선택하십시오"

containers = package["InferenceSpecification"]["Containers"]
if len(containers) != 1:
    raise ValueError("본 실습은 단일 추론 컨테이너만 지원합니다")

registered_container = containers[0]
image_uri = registered_container["Image"]
model_s3_uri = registered_container.get("ModelDataUrl")
if not model_s3_uri or not model_s3_uri.startswith("s3://"):
    raise ValueError("ModelDataUrl이 단일 S3 URI인지 확인하십시오")

print("승인된 모델 버전:", package["ModelPackageVersion"])
print("모델 S3 경로:", model_s3_uri)
print("추론 이미지:", image_uri)
print("기존 환경 설정:", json.dumps(registered_container.get("Environment", {}), indent=2))
```

**중요:** 이번 방식은 `model.joblib`이 아티팩트 최상위에 저장되었다는 기존 `train.py`의 동작을 전제로 합니다. 다른 이름으로 저장했다면 `model_fn()`의 파일명을 맞추십시오.

### 12-4. 추론 코드만 압축하여 S3에 업로드

`inference.py` **하나만** 포함한 코드 아카이브를 만듭니다. 모델 파일은 이 과정에 전혀 관여하지 않습니다.

```python
import tarfile
from pathlib import Path
from urllib.parse import urlparse

assert Path("inference.py").is_file(), "먼저 12-2 셀을 실행하십시오"

with tarfile.open("source.tar.gz", "w:gz") as tar:
    tar.add("inference.py", arcname="inference.py")

parsed = urlparse(model_s3_uri)
bucket = parsed.netloc
code_key = f"week11/serving-code/v{package['ModelPackageVersion']}/source.tar.gz"
s3.upload_file("source.tar.gz", bucket, code_key)
code_s3_uri = f"s3://{bucket}/{code_key}"

print("모델 아티팩트 (변경 없음):", model_s3_uri)
print("추론 코드 아카이브 (신규):", code_s3_uri)
```

### 12-5. 모델 S3 URI + 코드 S3 URI로 SageMaker Model 생성

Scikit-learn SageMaker 추론 이미지의 스크립트 모드에서 `SAGEMAKER_PROGRAM`은 실행할 모듈 파일명, `SAGEMAKER_SUBMIT_DIRECTORY`는 **S3에 업로드한 소스 코드 아카이브**를 나타냅니다. 원본 모델 아티팩트를 수정하지 않습니다.

```python
model_name = f"week11-iris-script-v{package['ModelPackageVersion']}"

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": model_s3_uri,
        "Environment": {
            **registered_container.get("Environment", {}),
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": code_s3_uri,
        },
    },
)
print("SageMaker Model 생성:", model_name)
```

> `SAGEMAKER_SUBMIT_DIRECTORY`를 `/opt/ml/model/code`로 지정하는 방식과 다릅니다. **이번 방식은 코드가 모델 아티팩트 안에 없으므로 S3의 `source.tar.gz` URI를 지정**합니다. 사용 중인 Scikit-learn 이미지가 이 스크립트 모드를 지원해야 하며, 실행 역할에 모델 및 코드 S3 객체의 읽기 권한이 있어야 합니다.

### 12-6. 일반 Real-time Endpoint 생성 (`ml.m5.large`)

```python
endpoint_config_name = f"week11-iris-script-config-v{package['ModelPackageVersion']}"
endpoint_name = f"week11-iris-script-endpoint-v{package['ModelPackageVersion']}"

sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
    }],
)
sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name,
)

try:
    sm.get_waiter("endpoint_in_service").wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    print("Endpoint InService:", endpoint_name)
except Exception:
    detail = sm.describe_endpoint(EndpointName=endpoint_name)
    print("Endpoint 상태:", detail["EndpointStatus"])
    print("실패 사유:", detail.get("FailureReason"))
    raise
```

**재실행 주의:** 같은 이름의 Model / Endpoint Configuration / Endpoint가 이미 있다면 생성 API가 실패합니다. 앞서 실패한 리소스가 있다면 삭제 상태를 확인하거나 새로운 이름을 사용하십시오. Endpoint가 `InService`가 되기 전에는 추론하지 않습니다.

### 12-7. 배포된 모델로 Iris 추론

```python
runtime = boto3.client("sagemaker-runtime", region_name=region)

iris_data = "5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3\n"
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Accept="text/csv",
    Body=iris_data.encode("utf-8"),
)

raw = response["Body"].read().decode("utf-8")
print("원본 예측 결과:\n", raw)

names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
for i, label in enumerate(raw.strip().splitlines(), start=1):
    cls = int(label.strip())
    print(f"붓꽃 {i}: {cls} ({names.get(cls, '알 수 없음')})")
```

기대 예측은 대체로 `0 / 2`이지만 실제 결과는 학습된 모델에 따라 달라질 수 있습니다. `InService`라도 추론 실패 시 CloudWatch 로그를 확인하십시오.

### 12-8. 실습 확인 포인트

1. `model_s3_uri`와 `code_s3_uri`가 **서로 다른 S3 객체**인지 확인합니다.
2. `model.tar.gz`를 Notebook으로 다운로드하거나 수정하지 않았는지 확인합니다.
3. `create_model()`의 `Environment`에 `SAGEMAKER_PROGRAM`과 `SAGEMAKER_SUBMIT_DIRECTORY`가 설정되었는지 확인합니다.
4. 일반 Real-time Endpoint이므로 `InferenceComponentName` 없이 호출하는지 확인합니다.
5. 모델 등록 시 추론 환경까지 완성해 두는 방식과, **이번처럼 배포 시 추론 스크립트를 별도로 연결하는 방식**의 차이를 설명합니다.

## 16. Step 13 — 과금 리소스 정리 (필수)

이번 개정 실습에서 생성한 Real-time Endpoint를 삭제합니다. **실행 중인 Real-time Endpoint는 요청이 없어도 인스턴스 비용이 발생**합니다.

```python
# 추론 종료 후 실행
sm.delete_endpoint(EndpointName=endpoint_name)
sm.get_waiter("endpoint_deleted").wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)
sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sm.delete_model(ModelName=model_name)
print("새 Real-time Endpoint / Config / Model 삭제 완료")
```

**이전에 GUI에서 생성한 Endpoint는 별도로 삭제해야 합니다.** 해당 Endpoint에 Inference Component가 연결되어 있다면 컴포넌트의 상태 및 삭제 가능 여부를 먼저 확인하고, GUI에서 컴포넌트 → Endpoint 순으로 정리하십시오. 이미 실패한 Serverless Endpoint도 잔존 여부를 확인하십시오. 기존 Model Registry v1, 파이프라인 정의, S3 학습 아티팩트는 이 코드로 삭제되지 않습니다.

- Studio JupyterLab 스페이스 중지
- S3에 추가 업로드한 `week11/serving-code/v.../source.tar.gz` 코드 아카이브 정리 여부 결정
- Model Registry 버전은 보관할 수 있으나 스토리지 비용 확인
- Billing / Cost Explorer에서 실습 비용 확인

## 17. 실제 발생 오류와 해결 방법

| 증상 | 원인 및 조치 |
|---|---|
| `NameError: package_arn is not defined` | Step 11에서 GUI 승인된 버전의 ARN을 선택하는 셀을 먼저 실행 |
| `Waiter ... EndpointStatus Failed` | `describe_endpoint()["FailureReason"]`과 CloudWatch 확인 |
| `Unable to successfully stand up ... 180 second timeout` | Serverless 추론 컨테이너 초기화 제한. 이번 개정에서는 일반 Real-time Endpoint 사용. 근본 원인은 로그로 진단 |
| `Inference Component Name header is required` | GUI가 Inference Component 방식으로 배포한 경우 컴포넌트 이름 필요. 이번 개정 방식은 해당 없음 |
| `Inference Component has no capacity` | 컴포넌트가 `Creating`이고 `Current copies=0`이면 준비될 때까지 대기 |
| `Cannot update ... CREATE_IN_PROGRESS` | 생성 중 복제본 수 변경 불가. 완료 또는 실패 상태 확인 |
| `AttributeError: 'NoneType' object has no attribute 'startswith'` 및 `/ping 500` | Scikit-learn 추론 모듈 엔트리포인트 누락 가능성. S3의 `source.tar.gz` + `SAGEMAKER_PROGRAM` + `SAGEMAKER_SUBMIT_DIRECTORY` 지정 |
| `ModelError`, CSV 파싱 오류 | `inference.py`의 `model_fn/input_fn/predict_fn/output_fn` 및 CloudWatch 로그 확인 |
| `ResourceLimitExceeded`, `AccessDenied` | 리전별 인스턴스 할당량, IAM 역할·S3·ECR·CloudWatch 권한 확인 |
| `Cannot assign property reference ... to argument of type String` | `NEstimators`를 `ParameterString`으로 선언하고 실행 시 문자열로 전달 |
| Pipeline 중복 Step | 커널 재시작 후 전처리→학습→평가→조건→Pipeline 순서대로 객체 재생성 |

**CloudWatch 로그 그룹:** `/aws/sagemaker/Endpoints/<EndpointName>`

```python
logs = boto3.client("logs", region_name=region)
log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
try:
    for event in logs.filter_log_events(logGroupName=log_group, limit=100).get("events", []):
        print(event["message"])
except Exception as e:
    print("로그 그룹 미생성 또는 로그 조회 권한 확인:", e)
```

## 18. 학습 정리 및 확인 문제

1. `train.py`에서 모델을 저장했는데도 추론 코드 `inference.py`를 별도로 연결하는 이유는 무엇인가요?
2. Model Registry의 `Approved`와 Endpoint의 `InService`는 각각 무엇을 보장하나요?
3. `EndpointStatus=InService`인데 Inference Component가 `Creating / Current copies=0`이라면 추론이 가능한가요?
4. 일반 Real-time Endpoint와 Serverless Endpoint의 비용·초기화 방식은 어떻게 다른가요?
5. 정확도 조건을 충족하지 못한 Pipeline 실행에서 `RegisterModel`이 생략되는 이유는 무엇인가요?

## 공식 참고자료

- [SageMaker SDK v3 파이프라인 예제](https://sagemaker.readthedocs.io/en/stable/v3-examples/ml-ops-examples/v3-pipeline-train-create-registry.html)
- [SageMaker SDK v3 MLOps API](https://sagemaker.readthedocs.io/en/v3.26.0/ml_ops/index.html)
- [SageMaker Model Registry 배포](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html)
- [SageMaker 실시간 엔드포인트](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [SageMaker Inference Components](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-component.html)
- [SageMaker 할당량](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)

---

**실습 환경:** SageMaker SDK 3.x, Boto3, SageMaker Scikit-learn 이미지 `1.2-1`, 리전별 사용 가능한 `ml.m5.large`. API와 이미지 동작은 수업 전 실제 계정에서 검증해야 합니다. 본 개정안은 기존 실습에서 확인된 오류를 근거로 배포 구성을 수정한 것이며, 새 Endpoint의 성공을 사전 보장하는 것은 아닙니다.
