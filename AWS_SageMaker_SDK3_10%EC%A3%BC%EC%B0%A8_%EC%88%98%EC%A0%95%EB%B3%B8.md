# 10주차 실습: SageMaker상에서 Iris 파이프라인 구축·모델 등록·배포


## 1. 실습 목표

- Kubeflow Pipelines와 SageMaker Pipelines의 컴포넌트/Step, 아티팩트/S3, Run/Execution을 비교함.
- Iris 데이터셋을 전처리·학습·평가하고 정확도가 기준 이상일 때만 모델 레지스트리에 등록함.
- 서로 다른 `NEstimators` 및 `AccuracyThreshold`로 3회 실행하고 실행별 평가 결과와 등록 여부를 비교함.
- 등록된 특정 모델 버전을 수동 승인한 뒤 서버리스 엔드포인트로 배포하고 삭제함.

## 2. 환경 및 사전 점검 (필수)

| 항목 | 권장 설정 / 검증 항목 |
|---|---|
| 계정 | SageMaker Studio, Processing, Training, Model Registry, Serverless Inference 생성 권한 확인 |
| Studio | JupyterLab 스페이스 `ml.t3.medium` (사용 가능할 때) |
| 작업 인스턴스 | Processing/Training `ml.m5.large` 예시. 리전별 지원 및 계정별 할당량 확인 후 변경 |
| SDK | `sagemaker==3.26.0`을 예시 검증 후보로 고정. 설치 가능성과 API 동작 사전 검증 필요 |
| 이미지 | SageMaker scikit-learn `1.2-1` (공식 v3 예제의 이미지 검색 패턴 기준). 오래된 이미지이므로 수업 전 이미지 URI 조회·실행 테스트 필수 |
| IAM | SageMaker 실행 역할의 S3, ECR, CloudWatch, SageMaker 작업 생성·모델 등록·배포 권한 확인 |


## 3. 비용 관리

- Processing 2회 + Training 1회가 파이프라인 실행마다 발생함. 기본 3회 실행 시 총 9개 작업임.
- `ml.m5.large`는 작업 시간만큼 과금되지만 Studio JupyterLab은 별도로 실행 시간 과금됨.
- 서버리스 추론은 사용량 기반이지만 엔드포인트 구성·로그·저장소 비용과 계정 할당량을 확인함.
- 실시간 인스턴스 엔드포인트는 사용하지 않음. 실습 종료 즉시 서버리스 엔드포인트를 삭제함.
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

> 설치 실패 시 다른 버전으로 임의 변경하지 않고 교수자가 검증한 버전을 전체 학생에게 동일하게 배포함. 설치 후 커널 재시작 필수.

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
input_uri = aws_session.upload_data("iris.csv", bucket=bucket, key_prefix="week10-iris/raw")
print(input_uri)
```

## 7. Step 4 — 컨테이너 URI 확인

```python
FRAMEWORK = "1.2-1"   # 공식 v3 예제에서 사용된 이미지 버전; 최신 지원 여부 별도 확인
INSTANCE_TYPE = "ml.m5.large"  # 교수자 사전 검증 후 변경 가능

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
from sagemaker.core.workflow.parameters import ParameterInteger, ParameterFloat
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep

n_estimators = ParameterInteger(name="NEstimators", default_value=100)
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.85)

processor = ScriptProcessor(
    image_uri=sklearn_image,
    command=["python3"],
    role=role,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    sagemaker_session=pipeline_session,
)

prefix = "week10-iris/pipeline"
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

# train.py가 위치한 현재 디렉터리를 소스 패키지로 업로드함
# ModelTrainer가 학습 컨테이너에서 train.py를 실행하도록 지정
trainer = ModelTrainer(
    training_image=sklearn_image,
    source_code=SourceCode(source_dir=".", entry_script="train.py"),
    role=role,
    compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1),
    hyperparameters={"n_estimators": n_estimators},
    input_data_config=[InputData(
        channel_name="train",
        data_source=step_prep.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        content_type="text/csv",
    )],
    sagemaker_session=pipeline_session,
)
```

`SourceCode(source_dir=".", entry_script="train.py")`는 SDK 3.x 공식 문서에 기재된 방식임. 현재 디렉터리에 `train.py`가 존재하는지 확인하고, 학습 컨테이너에서 사용되는 Python 및 scikit-learn 버전은 사전 테스트함.

```python
from pathlib import Path
assert Path("train.py").is_file(), "train.py 파일을 먼저 생성해야 함"
```

다음 셀에서 학습 작업을 파이프라인 단계로 정의함:

```python
train_args = trainer.train()
step_train = TrainingStep(name="Train", step_args=train_args)
```

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
from sagemaker.core.workflow.condition_step import ConditionStep
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet

MODEL_GROUP = "week10-iris-models"

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
step_register = ModelStep(name="RegisterModel", step_args=register_args)

step_condition = ConditionStep(
    name="CheckAccuracy",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=eval_report,
            json_path="metrics.accuracy.value",
        ),
        right=accuracy_threshold,
    )],
    if_steps=[step_register],
    else_steps=[],
)
```

차이점: 원본은 `ModelMetrics`를 모델 패키지에 저장했음. 이 개정안의 기본 등록 셀은 지표 등록이 빠져 있으므로, *모델 레지스트리 화면에 정확도 지표가 표시되는 것*까지 교육 목표라면 v3 `ModelBuilder.register()`의 `model_metrics` 지원 인자 및 `MetricsSource` 객체를 사전 확인하고 추가해야 함. 다만 평가 JSON은 S3에 별도 저장됨.

## 12. Step 9 — 파이프라인 정의 및 실행

```python
from sagemaker.mlops.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name="week10-iris-sdk3",
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

`train.py` 실행 설정을 완성하고 교수자가 정의 JSON을 검토한 뒤 실행함. `PropertyFile` 및 조건 분기는 해당 버전의 import 경로를 확인함.

## 13. Step 10 — 파라미터 변경 실행과 비교

```python
execution2 = pipeline.start(parameters={"NEstimators": 5})
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

## 14. Step 11 — 모델 버전 확인 및 수동 승인

```python
sm = boto3.client("sagemaker", region_name=region)
packages = sm.list_model_packages(
    ModelPackageGroupName=MODEL_GROUP,
    SortBy="CreationTime", SortOrder="Descending",
)["ModelPackageSummaryList"]
for p in packages:
    print(p["ModelPackageVersion"], p["ModelPackageArn"], p["ModelApprovalStatus"])
```

실습 안전장치: 가장 최근 버전을 자동 승인하지 않음. 학생이 출력 결과를 확인한 후 배포할 버전 번호를 명시함.

```python
TARGET_VERSION = 1  # 실제 등록된 버전으로 수정
selected = next(p for p in packages if p["ModelPackageVersion"] == TARGET_VERSION)
package_arn = selected["ModelPackageArn"]
sm.update_model_package(ModelPackageArn=package_arn, ModelApprovalStatus="Approved")
print("approved:", package_arn)
```

## 15. Step 12 — 승인된 모델 패키지로 서버리스 엔드포인트 생성

이 단계는 SDK 3.x에서 폐기된 `SKLearnModel`을 사용하지 않음. 등록된 모델 패키지 ARN을 SageMaker `CreateModel` API의 `Containers=[{"ModelPackageName": ARN}]` 형태로 전달함. `boto3`는 SDK 3.x와 병행 가능한 AWS 공식 서비스 API 클라이언트임.

> 실행 전: 해당 계정의 Serverless Inference 지원 여부 및 `MaxConcurrency` 할당량 확인. 승인된 모델 패키지의 이미지와 추론 코드가 실제 CSV 입력을 처리할 수 있는지도 검증해야 함. `ModelBuilder.register()`가 모델 아티팩트와 추론 코드를 올바르게 패키징했는지 확인하지 않고 배포하지 않음.

```python
from botocore.exceptions import ClientError

model_name = "week10-iris-model"
endpoint_config_name = "week10-iris-serverless-config"
endpoint_name = "week10-iris-serverless"

# 이미 같은 이름의 리소스가 있으면 덮어쓰지 않고 중단함
def assert_not_exists(describe, **kwargs):
    try:
        describe(**kwargs)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return
        raise
    raise RuntimeError(f"이미 존재하는 리소스: {kwargs}")

assert_not_exists(sm.describe_model, ModelName=model_name)
assert_not_exists(sm.describe_endpoint_config, EndpointConfigName=endpoint_config_name)
assert_not_exists(sm.describe_endpoint, EndpointName=endpoint_name)
```

다음 셀은 위 중복 확인을 통과한 경우에만 실행함. 중간 단계에서 오류가 발생하면 생성된 리소스를 확인하고 정리한 후 재실행함.

```python
sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    Containers=[{"ModelPackageName": package_arn}],
)
sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 1},
    }],
)
sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name,
)
sm.get_waiter("endpoint_in_service").wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)
print("endpoint:", sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"])
```

추론:

```python
runtime = boto3.client("sagemaker-runtime", region_name=region)
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Accept="text/csv",
    Body="5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3\n".encode("utf-8"),
)
print(response["Body"].read().decode("utf-8"))
```

주의: 서버리스 추론의 콜드 스타트로 초기 응답이 지연될 수 있음. 실패 시 CloudWatch 로그와 컨테이너 응답 형식을 확인함. `ModelBuilder.register()`에서 추론 스크립트 패키징이 누락되면 배포가 성공해도 호출이 실패할 수 있음.

## 16. Step 13 — 과금 리소스 정리 (필수)

```python
# 엔드포인트를 먼저 삭제하고 삭제 완료를 확인함.
sm.delete_endpoint(EndpointName=endpoint_name)
sm.get_waiter("endpoint_deleted").wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)
sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sm.delete_model(ModelName=model_name)
print("엔드포인트·설정·배포 모델 정리 완료")
```

- Studio에서 JupyterLab 스페이스 `week10` 중지
- 모델 레지스트리와 파이프라인 정의, S3 아티팩트는 수업 종료까지 보관 가능(스토리지 과금 확인)
- 전체 강의 종료 후 모델 패키지 버전 → 모델 패키지 그룹 → 파이프라인 → S3 데이터 순서로 삭제
- 다음 날 Billing 및 Cost Explorer에서 예상치 못한 과금 확인

## 17. 오류별 점검

| 오류 | 원인과 조치 |
|---|---|
| `ImportError` | 설치된 SDK v3와 코드의 import 경로 불일치. SDK 버전과 공식 문서 대조 |
| `AttributeError: sagemaker.__version__` | `importlib.metadata.version("sagemaker")`로 변경; 이 자료의 Step 1 참고 |
| `ValidationException` | 인스턴스 유형 미지원, 이미지 URI, 모델 패키지/서버리스 호환성 확인 |
| `ResourceLimitExceeded` | Training / Processing / Serverless 할당량 각각 확인 |
| `AccessDenied` | 실행 역할과 노트북 실행 주체의 IAM 권한 확인 |
| `train.py` 미실행 | `SourceCode(source_dir=".", entry_script="train.py")`와 스크립트 업로드 확인 |
| 평가에서 압축 해제 오류 | Python `tarfile` 버전과 아티팩트 경로 확인 |
| 모델 등록 생략 | 정확도가 임계값보다 낮아 조건 분기에서 제외된 것인지 확인 |
| 서버리스 추론 실패 | 추론 컨테이너 이미지, 모델 로딩 함수, 입력 `ContentType`, CloudWatch 로그 확인 |



## 공식 참고자료

- SDK v3 E2E 파이프라인 예제: https://sagemaker.readthedocs.io/en/stable/v3-examples/ml-ops-examples/v3-pipeline-train-create-registry.html
- SDK v3 MLOps API: https://sagemaker.readthedocs.io/en/v3.26.0/ml_ops/index.html
- SDK v3 ModelBuilder: https://sagemaker.readthedocs.io/en/stable/api/sagemaker_serve.html
- AWS 모델 레지스트리 배포: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html
- AWS 서버리스 엔드포인트: https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html
- AWS SageMaker 할당량: https://docs.aws.amazon.com/general/latest/gr/sagemaker.html


## 수정 이력 및 검증 범위

- 2026-10-29: SDK 3.x 버전 조회를 `importlib.metadata.version()`으로 수정함.
- `ModelTrainer`에 `SourceCode`를 연결하여 `train.py` 실행 설정 누락을 보완함.
- 추론 CSV 처리 함수를 명시하고, 평가 모델 파일의 안전한 읽기 방식 및 엔드포인트 중복 검사 코드를 수정함.
- **검증 범위:** Python 코드 블록의 정적 문법 검사 및 AWS SDK 3.x 공식 API 문서 대조. 실제 AWS 작업 생성·모델 패키지 등록·서버리스 배포는 실행 검증되지 않음. 특히 `ModelBuilder.register()`의 추론 코드 패키징, `SourceCode`와 파이프라인 파라미터의 조합, 동일 S3 출력 경로에서의 병렬 실행은 교수자 사전 검증이 필요함.
