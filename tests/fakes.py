"""Shared test doubles (no MLflow server needed)."""
from types import SimpleNamespace

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


class FakeClient:
    def __init__(self):
        self.calls = []
        self.versions = []            # list of SimpleNamespace(name, version, tags)
        self.experiments = {}
        self.registered = set()
        self.fail = {}                # method name -> exception to raise
        self.version_status = "READY"
        self.run_tags = {}

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))
        if name in self.fail:
            raise self.fail[name]

    def names(self):
        return [c[0] for c in self.calls]

    def arg(self, name):
        return [c for c in self.calls if c[0] == name][-1]

    # --- API used by the service
    def search_model_versions(self, filter_string=None, max_results=1000, page_token=None, **kw):
        self._rec("search_model_versions")
        out = [v for v in self.versions if filter_string is None or v.name == filter_string.split("'")[1]]
        return out

    def get_experiment_by_name(self, name):
        self._rec("get_experiment_by_name", name)
        return self.experiments.get(name)

    def create_experiment(self, name, artifact_location=None):
        self._rec("create_experiment", name, artifact_location=artifact_location)
        self.experiments[name] = SimpleNamespace(experiment_id="7", lifecycle_stage="active")
        return "7"

    def create_run(self, experiment_id, tags=None, run_name=None):
        self._rec("create_run", experiment_id, tags=tags, run_name=run_name)
        self.run_tags = dict(tags or {})
        return SimpleNamespace(info=SimpleNamespace(run_id="run123"))

    def log_batch(self, run_id, metrics=(), params=(), tags=()):
        self._rec("log_batch", run_id, metrics=list(metrics), params=list(params))

    def log_dict(self, run_id, d, path):
        self._rec("log_dict", run_id, d, path)

    def log_artifacts(self, run_id, local_dir, artifact_path=None):
        self._rec("log_artifacts", run_id, local_dir, artifact_path=artifact_path)

    def get_registered_model(self, name):
        self._rec("get_registered_model", name)
        if name not in self.registered:
            raise MlflowException("nope", error_code=RESOURCE_DOES_NOT_EXIST)

    def create_registered_model(self, name, description=None):
        self._rec("create_registered_model", name, description=description)
        self.registered.add(name)

    def create_model_version(self, name, source, run_id, tags, description):
        self._rec("create_model_version", name, source=source, run_id=run_id, tags=tags)
        return SimpleNamespace(version="3")

    def get_model_version(self, name, version):
        self._rec("get_model_version", name, version)
        return SimpleNamespace(status=self.version_status)

    def set_model_version_tag(self, name, version, key, value):
        self._rec("set_model_version_tag", name, version, key, value)

    def set_tag(self, run_id, key, value):
        self._rec("set_tag", run_id, key, value)

    def set_terminated(self, run_id, status):
        self._rec("set_terminated", run_id, status)
