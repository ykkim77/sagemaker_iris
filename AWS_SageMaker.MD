# 10주차 실습: SageMaker Pipelines로 Iris 파이프라인 구축·배포

 10. AWS 기반 AI플랫폼 실습① ·

## 1. 실습 목표

- 이전 시간 로컬에서 실습한 Iris 파이프라인(전처리 → 학습 → 평가)을 SageMaker Pipelines로 다시 구현한다.
- 평가 결과(정확도)가 기준 이상일 때만 모델을 모델 레지스트리에 등록하도록 조건 분기를 넣는다.
- 파라미터를 바꿔 여러 번 실행하고, 모델 버전별 정확도를 비교한다.
- 등록된 모델을 서버리스 추론으로 배포해 예측한 뒤, 과금 리소스를 모두 정리한다.

## 2. 실습 환경

| 항목 | 내용 |
|---|---|
| 리전 | 미국 동부(버지니아 북부) `us-east-1` — 11주차 Bedrock 실습을 위해 리전을 미국 버지니아에서 합니다. |
| 개발 환경 | SageMaker Studio의 JupyterLab 스페이스 (`ml.t3.medium`) |
| 작업 인스턴스 | 전처리·평가 `ml.t3.medium`, 학습 `ml.m5.large` |
| SDK | SageMaker Python SDK **2.x 버전 고정** |
| 실행 역할 | 도메인 빠른 설정 시 자동 생성되는 SageMaker 실행 역할 |
| 데이터 | scikit-learn 내장 Iris 데이터셋 (150행) |

> **SDK 버전 주의**: SageMaker Python SDK는 V3에서 `Estimator`, `Model`, `Predictor` 등의 인터페이스가 크게 바뀌었습니다. 이 실습 코드는 **2.x 버전 기준**이므로 반드시 Step 2에서 버전을 고정하세요. JupyterLab 기본 이미지에 V3가 설치되어 있어도 2.x로 덮어 설치하면 됩니다.

## 3. 비용 계획 (10·11주차 합산 $100 크레딧 기준)

| 주차 | 항목 | 예상 비용 |
|---|---|---|
| 10 | JupyterLab 스페이스 (`ml.t3.medium`, 3시간 내외) | $1 미만 |
| 10 | 전처리·학습·평가 작업 (파이프라인 3회 실행) | $1 미만 |
| 10 | 서버리스 추론 테스트 | $1 미만 |
| 11 | Bedrock 모델 호출·임베딩·Knowledge Bases (저비용 벡터 저장소 사용) | $5 미만 |
| | **목표**: 10주차 $5, 11주차 $10 이내 | 여유분 약 $85 |

- 위 금액은 공개 요금 기준의 추정치입니다. 실제 청구액은 리전과 시점에 따라 다를 수 있습니다.
- **가장 큰 위험은 정리하지 않은 리소스입니다.** 예를 들어 `ml.m5.large` 실시간 엔드포인트를 한 달 켜 두면 약 $80으로 크레딧 대부분이 사라집니다. 이 실습은 실시간 엔드포인트를 쓰지 않고 서버리스 추론만 사용합니다.

## 4. 사전 점검

`. **리전**: 콘솔 오른쪽 위 리전이 `미국 동부(버지니아 북부) us-east-1`인지 확인합니다.
1. **할당량**: 신규 계정은 일부 인스턴스 유형의 할당량이 작을 수 있습니다. 작업이 `ResourceLimitExceeded` 오류로 실패하면 6절 문제 해결표를 참고하세요.

## 5. 실습 단계

### Step 1. SageMaker AI 도메인 생성

1. AWS 콘솔에서 **Amazon SageMaker AI**를 엽니다.
2. 왼쪽 메뉴 **도메인(Domains)** → **도메인 생성(Create domain)**을 클릭합니다.
3. **단일 사용자용 설정(빠른 설정, Set up for single user)**을 선택합니다. 실행 역할과 사용자 프로필이 자동으로 만들어집니다.
4. 생성이 끝나면(수 분 소요) 사용자 프로필에서 **Studio 열기**를 클릭합니다.



### Step 2. JupyterLab 스페이스 생성과 SDK 버전 고정

1. Studio 왼쪽의 **JupyterLab** → **JupyterLab 스페이스 생성**을 클릭하고 이름을 `week10`으로 지정합니다.
2. 인스턴스를 `ml.t3.medium`으로 선택하고 **스페이스 실행**을 누른 뒤 **열기**를 클릭합니다.
3. 새 노트북(Python 3)을 만들고 첫 셀에서 SDK를 설치합니다.

```python
%pip install -q "sagemaker>=2.200,<3"
```

4. 메뉴 **Kernel → Restart Kernel**로 커널을 재시작한 뒤 버전을 확인합니다.

```python
import sagemaker
print(sagemaker.__version__)   # 2.x.x 가 출력되어야 함
```

"SDK v2는 지원 중단 예정(deprecation)"이라는 경고가 출력될 수 있습니다. 이 실습에서는 무시해도 됩니다.

### Step 3. 스크립트 3개 작성

6주차의 컴포넌트 함수 본문을 스크립트 파일로 옮깁니다. 아래 셀을 차례로 실행하면 노트북과 같은 폴더에 파일이 생깁니다.

**(1) 전처리: `preprocess.py`**

```python
%%writefile preprocess.py
import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "/opt/ml/processing"

df = pd.read_csv(f"{BASE}/input/iris.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

os.makedirs(f"{BASE}/train", exist_ok=True)
os.makedirs(f"{BASE}/test", exist_ok=True)
train.to_csv(f"{BASE}/train/train.csv", index=False)
test.to_csv(f"{BASE}/test/test.csv", index=False)
print(f"train={len(train)}, test={len(test)}")
```

**(2) 학습: `train.py`**

학습 작업에서 실행될 뿐 아니라, 서버리스 추론에서도 `model_fn`으로 모델을 불러오는 데 쓰입니다.

```python
%%writefile train.py
import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def model_fn(model_dir):
    """추론 컨테이너가 모델을 불러올 때 호출하는 함수"""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.train, "train.csv"))
    X, y = df.drop(columns=["target"]), df["target"]

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    clf.fit(X.values, y.values)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    print(f"n_estimators={args.n_estimators}, 학습 완료")
```

**(3) 평가: `evaluate.py`**

평가 결과를 `evaluation.json`에 씁니다. `metrics` 키는 조건 분기(ConditionStep)에서 읽고, `multiclass_classification_metrics` 키는 모델 레지스트리의 지표로 기록됩니다.

```python
%%writefile evaluate.py
import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

BASE = "/opt/ml/processing"

with tarfile.open(f"{BASE}/model/model.tar.gz") as tar:
    tar.extractall(path=f"{BASE}/model")
model = joblib.load(f"{BASE}/model/model.joblib")

df = pd.read_csv(f"{BASE}/test/test.csv")
X, y = df.drop(columns=["target"]), df["target"]
acc = float(accuracy_score(y, model.predict(X.values)))

report = {
    "metrics": {"accuracy": {"value": acc}},
    "multiclass_classification_metrics": {"accuracy": {"value": acc, "standard_deviation": "NaN"}},
}
os.makedirs(f"{BASE}/evaluation", exist_ok=True)
with open(f"{BASE}/evaluation/evaluation.json", "w") as f:
    json.dump(report, f)
print(f"accuracy={acc:.4f}")
```

### Step 4. 세션 준비와 데이터 업로드

```python
import boto3
import sagemaker
from sklearn.datasets import load_iris
from sagemaker.workflow.pipeline_context import PipelineSession

session = PipelineSession()                 # 파이프라인 정의용 세션
role = sagemaker.get_execution_role()       # Step 1에서 만들어진 실행 역할
bucket = session.default_bucket()           # sagemaker-us-east-1-<계정번호> 버킷
region = session.boto_region_name
print(region, bucket)

load_iris(as_frame=True).frame.to_csv("iris.csv", index=False)
input_uri = session.upload_data("iris.csv", bucket=bucket, key_prefix="iris/raw")
print(input_uri)
```

`region`이 `us-east-1`로 출력되는지 확인하세요.

### Step 5. 파라미터와 전처리 단계

```python
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep

FRAMEWORK = "1.2-1"   # 학습·전처리·추론에 같은 scikit-learn 버전 사용

n_estimators = ParameterInteger(name="NEstimators", default_value=100)
acc_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.85)

processor = SKLearnProcessor(
    framework_version=FRAMEWORK,
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=session,
)

step_prep = ProcessingStep(
    name="Preprocess",
    step_args=processor.run(
        code="preprocess.py",
        inputs=[ProcessingInput(source=input_uri, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
    ),
)
```

### Step 6. 학습 단계

전처리 단계의 출력 S3 경로를 `properties`로 참조합니다. 이 참조 때문에 학습 단계는 전처리가 끝난 뒤 자동으로 실행됩니다(6주차 자동 의존성 추론과 같은 원리).

```python
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep

estimator = SKLearn(
    entry_point="train.py",
    framework_version=FRAMEWORK,
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    hyperparameters={"n_estimators": n_estimators},
    sagemaker_session=session,
)

step_train = TrainingStep(
    name="Train",
    step_args=estimator.fit({
        "train": TrainingInput(
            s3_data=step_prep.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        )
    }),
)
```

### Step 7. 평가 단계와 PropertyFile

```python
from sagemaker.workflow.properties import PropertyFile

eval_report = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")

step_eval = ProcessingStep(
    name="Evaluate",
    step_args=processor.run(
        code="evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_prep.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
    ),
    property_files=[eval_report],
)
```

### Step 8. 모델 등록 단계와 조건 분기

정확도가 `AccuracyThreshold` 이상이면 모델을 `iris-models` 그룹에 **승인 대기(PendingManualApproval)** 상태로 등록합니다. 평가 결과 파일은 모델 지표로 함께 기록됩니다.

```python
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep

MODEL_GROUP = "iris-models"

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(on="/", values=[
            step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "evaluation.json",
        ]),
        content_type="application/json",
    )
)

model = SKLearnModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="train.py",
    framework_version=FRAMEWORK,
    sagemaker_session=session,
)

step_register = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    ),
)

step_cond = ConditionStep(
    name="CheckAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=step_eval.name, property_file=eval_report,
                         json_path="metrics.accuracy.value"),
            right=acc_threshold,
        )
    ],
    if_steps=[step_register],
    else_steps=[],
)
```

> 기준값을 0.85로 둔 이유: 테스트 데이터가 30개뿐이라 정확도가 0.9, 0.933처럼 1/30 단위로 움직입니다. 기준을 0.9로 두면 경계값에 걸려 실행마다 결과가 달라 보일 수 있습니다.

### Step 9. 파이프라인 생성과 실행

```python
from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name="iris-pipeline",
    parameters=[n_estimators, acc_threshold],
    steps=[step_prep, step_train, step_eval, step_cond],
    sagemaker_session=session,
)

pipeline.upsert(role_arn=role)   # 파이프라인 정의(JSON)를 SageMaker에 등록
execution = pipeline.start()     # 기본값: NEstimators=100, AccuracyThreshold=0.85
execution.wait(delay=30, max_attempts=60)
execution.list_steps()
```

실행에는 10~15분 정도 걸립니다. 각 작업이 인스턴스를 띄우고 컨테이너를 내려받는 시간이 포함되기 때문입니다.

### Step 10. 실행 결과 확인

1. Studio 왼쪽 메뉴 **Pipelines** → `iris-pipeline` → 실행 목록에서 방금 실행을 선택합니다.
2. 그래프에서 `Preprocess → Train → Evaluate → CheckAccuracy → RegisterModel-RegisterModel` 단계가 모두 성공(초록색)인지 확인합니다. (`ModelStep`은 내부적으로 `단계이름-RegisterModel`이라는 이름으로 표시됩니다.)
3. `Evaluate` 단계를 클릭해 출력 S3 경로와 로그를 확인합니다.
4. **Models → Model registry**(모델 레지스트리)에서 `iris-models` 그룹에 버전 1이 승인 대기 상태로 등록되었는지 확인합니다.

> 화면 메뉴 이름과 배치는 콘솔 업데이트에 따라 다를 수 있습니다.

### Step 11. 파라미터를 바꿔 재실행하고 비교

두 번 더 실행합니다. 두 실행은 동시에 진행해도 됩니다.

```python
exec2 = pipeline.start(parameters={"NEstimators": 5})
exec3 = pipeline.start(parameters={"AccuracyThreshold": 0.999})
exec2.wait(delay=30, max_attempts=60)
exec3.wait(delay=30, max_attempts=60)
```

실행이 끝나면 모델 버전별 정확도를 표로 출력합니다.

```python
import json

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

pkgs = sm.list_model_packages(ModelPackageGroupName=MODEL_GROUP, SortBy="CreationTime",
                              SortOrder="Ascending")["ModelPackageSummaryList"]
for p in pkgs:
    desc = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"])
    uri = desc["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
    b, k = uri.replace("s3://", "").split("/", 1)
    acc = json.loads(s3.get_object(Bucket=b, Key=k)["Body"].read())["metrics"]["accuracy"]["value"]
    print(f"버전 {desc['ModelPackageVersion']}: accuracy={acc:.4f}, 상태={desc['ModelApprovalStatus']}")
```

생각해 볼 점:

1. `NEstimators=5`로 실행한 결과는 등록되었습니까? 정확도는 기본 실행과 비교해 어떻습니까? 트리 수가 적다고 항상 정확도가 낮아지는지, 테스트 데이터가 30개뿐이라는 점과 함께 생각해 보세요.
2. `AccuracyThreshold=0.999`로 실행했을 때 그래프에서 어떤 단계가 실행되지 않았습니까? 모델 버전 수는 늘어났습니까?
3. 6주차에는 사람이 Run 비교 화면을 보고 판단했습니다. 이번 파이프라인에서는 그 판단 중 무엇이 자동화되었습니까?

### Step 12. 모델 승인과 서버리스 추론 배포

가장 최근에 등록된 모델 버전을 **승인(Approved)**한 뒤 서버리스 추론으로 배포합니다. "승인된 모델만 배포한다"는 운영 절차를 체험하는 단계입니다.

```python
latest = sm.list_model_packages(ModelPackageGroupName=MODEL_GROUP, SortBy="CreationTime",
                                SortOrder="Descending")["ModelPackageSummaryList"][0]
arn = latest["ModelPackageArn"]
sm.update_model_package(ModelPackageArn=arn, ModelApprovalStatus="Approved")

desc = sm.describe_model_package(ModelPackageName=arn)
model_s3_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
print(desc["ModelPackageVersion"], model_s3_uri)
```

```python
from sagemaker.serverless import ServerlessInferenceConfig

deploy_session = sagemaker.Session()   # 배포는 일반 세션으로 수행

serve_model = SKLearnModel(
    model_data=model_s3_uri,
    role=role,
    entry_point="train.py",
    framework_version=FRAMEWORK,
    sagemaker_session=deploy_session,
)

predictor = serve_model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=1),
    endpoint_name="iris-serverless",
)

# 0=setosa, 1=versicolor, 2=virginica
print(predictor.predict([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]))
```

- 배포에는 수 분이 걸리고, 첫 예측은 콜드 스타트로 수십 초가 걸릴 수 있습니다.
- `instance_type`을 지정하는 일반 배포(실시간 엔드포인트)는 **사용하지 마세요.** 요청이 없어도 시간 단위로 과금됩니다.

### Step 13. 리소스 정리 (반드시 수행)

**(1) 엔드포인트와 모델 삭제**

```python
predictor.delete_endpoint()   # 엔드포인트와 엔드포인트 설정 삭제
predictor.delete_model()      # 배포용으로 만든 모델 삭제
```

**(2) 남은 엔드포인트가 없는지 확인**

```python
print(sm.list_endpoints()["Endpoints"])   # [] 이 출력되어야 함
```

**(3) JupyterLab 스페이스 중지**: 노트북을 저장한 뒤 Studio의 JupyterLab 스페이스 목록에서 `week10`의 **중지(Stop)**를 클릭합니다. 스페이스를 켜 두면 사용하지 않아도 시간 단위로 과금됩니다.

**(4) 유지하는 것과 다음 날 확인할 것**

| 항목 | 처리 | 이유 |
|---|---|---|
| 서버리스 엔드포인트·배포 모델 | 삭제 | 위 (1)에서 완료 |
| JupyterLab 스페이스 | 중지 | 위 (3)에서 완료 |
| 도메인·실행 역할 | 유지 | 11주차에 재사용, 과금 없음 |
| S3 데이터·파이프라인 정의·모델 레지스트리 | 유지 | 저장 용량이 작아 비용이 거의 없음 |
| 다음 날 | Billing 콘솔의 과금 내역 확인 | 예상 밖 항목이 있으면 즉시 확인 |

## 6. 문제 해결

| 증상 | 확인·해결 |
|---|---|
| `ImportError` 또는 `Estimator`·`SKLearn` 관련 오류 | SDK V3가 설치된 상태. Step 2의 설치 명령 실행 후 **커널을 재시작**했는지 확인 |
| 작업이 `ResourceLimitExceeded`로 실패 | 신규 계정의 인스턴스 할당량 부족. Service Quotas → Amazon SageMaker에서 해당 인스턴스 유형(예: `ml.m5.large for training job usage`)의 할당량 증가를 요청하거나, 교수 시연으로 대체 |
| `AccessDenied` 오류 | 실행 역할 권한 부족. 노트북에서 `role`이 빠른 설정으로 만든 SageMaker 실행 역할인지 확인 |
| 단계가 실패(빨간색) | 그래프에서 실패한 단계를 클릭해 로그(CloudWatch) 확인. 스크립트의 파일 경로 오타가 가장 흔한 원인 |
| `RegisterModel`이 실행되지 않음 | 정확도가 `AccuracyThreshold`보다 낮아 조건 분기에서 등록이 생략된 것. 오류가 아님 |
| 서버리스 배포 시 할당량 오류 | 서버리스 엔드포인트의 동시성 할당량 부족. `max_concurrency=1`인지 확인 |
| 첫 예측이 타임아웃 | 콜드 스타트. 잠시 후 다시 예측 |
| 리전이 `us-east-1`이 아님 | Studio를 다른 리전에서 연 경우. 콘솔 리전을 바꾸고 해당 리전의 도메인에서 다시 시작 |

## 7. 실습 정리

- 6주차 KFP의 컴포넌트·아티팩트·Run은 SageMaker Pipelines의 Step·S3 경로·Execution에 대응한다.
- PropertyFile과 ConditionStep으로 "정확도 기준을 넘은 모델만 등록"하는 판단을 파이프라인 안에서 자동화했다.
- 모델 레지스트리에서 버전·지표·승인 상태를 관리하고, 승인된 모델만 서버리스 추론으로 배포했다.
- 실습이 끝나면 엔드포인트 삭제와 스페이스 중지로 과금 리소스를 정리했다.

## 8. 강의 종료 후 전체 삭제 (11주차 이후, 선택)

11주차 실습까지 마친 뒤 AWS를 더 사용하지 않는다면 아래 순서로 모두 삭제합니다.

1. 남은 엔드포인트가 없는지 확인 (`sm.list_endpoints()`)
2. 파이프라인 삭제: `pipeline.delete()`
3. 모델 레지스트리의 모델 버전과 `iris-models` 그룹 삭제
4. JupyterLab 스페이스 삭제 → 사용자 프로필 삭제 → 도메인 삭제
5. S3 버킷 `sagemaker-us-east-1-<계정번호>` 비우기 및 삭제
