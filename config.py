"""Configuration helpers for the spike CAM experiment platform.

The project is intentionally configuration-driven:

- experiment settings live in JSON files under ``configs/``
- runtime code converts them into typed dataclasses
- experiment variants are expressed as config overrides rather than
  ad-hoc code edits

This module centralizes:

- dataclass definitions
- config loading / validation
- deep-merge logic for experiment variants
- path resolution relative to the project root

中文说明
--------
这个文件是整个实验平台的“配置中枢”。

你以后做实验时，原则上不应该去主逻辑代码里改参数，而是：

1. 在 ``configs/*.json`` 里写实验设置
2. 用本文件把 JSON 转成 dataclass
3. 通过 variant 机制批量展开实验

也就是说，这个文件主要解决三个问题：

- 怎么把实验参数组织清楚
- 怎么把不同实验配置统一加载
- 怎么保证以后“只改配置，不改主代码”
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    """Return a path inside the project root.

    中文：把相对路径统一挂到项目根目录下，避免硬编码绝对路径。
    """

    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_str: str) -> Path:
    """Resolve a config path relative to the project root.

    中文：配置文件里的路径默认按项目根目录解释。
    """

    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def json_ready(value: Any) -> Any:
    """Convert nested objects into JSON-serializable values."""

    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def strip_comment_fields(value: Any) -> Any:
    """Recursively remove comment-style keys from config payloads.

    Supported keys:
    - keys that start with ``_``
    - keys that end with ``_comment``
    - keys that end with ``_note``

    中文：
    JSON 原生不支持注释，所以这里允许在配置里写：

    - ``_comment``
    - ``_note``
    - ``_section_comment``

    读取配置时会自动忽略这些说明字段。
    """

    if isinstance(value, list):
        return [strip_comment_fields(item) for item in value]
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str.startswith("_") or key_str.endswith("_comment") or key_str.endswith("_note"):
                continue
            cleaned[key_str] = strip_comment_fields(item)
        return cleaned
    return value


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep-merge ``override`` into ``base`` and return ``base``."""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = deepcopy(value)
    return base


@dataclass
class WaveformConfig:
    """Waveform extraction settings.

    中文：控制怎样从连续 raw voltage 中截取单个 spike waveform。
    """

    waveform_length: int = 79
    center_index: int = 39
    align_mode: str = "min"
    alignment_reference: str = "global_activity"
    channel_selection: str = "topk_max_abs"
    fixed_channel: int = 0
    topk_channels: int = 8
    selection_radius: int = 8
    channel_order: str = "by_strength"
    flatten_order: str = "channel_major"


@dataclass
class PreprocessConfig:
    """Raw recording preprocessing settings.

    中文：
    这部分配置的是“在 waveform 截取之前”对连续电压做什么处理。

    默认值尽量贴近常见 extracellular spike pipeline：

    - bandpass: 开
    - CMR: 开
    - whitening: 关（因为 whitening 更激进，先作为可选项）
    """

    bandpass_enable: bool = True
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 6000.0
    bandpass_order: int = 3
    common_reference_enable: bool = True
    common_reference_mode: str = "median"
    whitening_enable: bool = False
    whitening_num_samples: int = 20000
    whitening_chunk_size: int = 50000
    whitening_eps: float = 1e-5


@dataclass
class SubsetConfig:
    """Label subset rule for long-tail control.

    中文：用于控制实验只看哪些 unit，解决真实数据长尾很严重的问题。
    """

    mode: str = "all"
    topk: Optional[int] = None
    min_count: Optional[int] = None


@dataclass
class MemorySubsetConfig:
    """Rule for deciding which labels are actually loaded into CAM memory.

    ``mode``:
    - ``same_as_stream``: keep the legacy behavior, i.e. every label seen in
      warmup can become an initial template
    - ``all`` / ``topk`` / ``min_count``: explicitly choose which labels the
      CAM is allowed to remember

    ``selection_source``:
    - ``pre_sampling``: rank labels using counts before waveform downsampling
    - ``encoded``: rank labels using the final encoded dataset
    - ``warmup``: rank labels using the warmup segment only

    中文：
    这个配置专门回答：
    “测试流里可以有很多类，但 CAM 最终允许记住哪些类？”

    也就是说：

    - ``dataset.subset`` 控制哪些 spike 会进入整条编码/测试流
    - ``cam.memory_subset`` 控制其中哪些类会真正被加载进 CAM 模板
    """

    mode: str = "same_as_stream"
    topk: Optional[int] = None
    min_count: Optional[int] = None
    selection_source: str = "pre_sampling"


@dataclass
class SamplingConfig:
    """Optional spike downsampling for faster experimentation.

    中文：可选的下采样配置，主要用于加快实验和做 sanity check。
    """

    max_total_spikes: Optional[int] = None
    max_spikes_per_unit: Optional[int] = None
    selection_mode: str = "uniform_time"


@dataclass
class DatasetConfig:
    """Raw dataset settings.

    中文：原始数据层配置，包括数据路径、waveform 提取和 subset 规则。
    """

    npz_path: str = "dataset/my_validation_subset_810000samples_27.00s.npz"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    sort_by_time: bool = True


@dataclass
class EncoderConfig:
    """Encoder settings.

    ``method``:
    - ``ae``: autoencoder-style encoder
    - ``pca``: PCA baseline encoder

    ``backend`` for ``ae``:
    - ``auto``: prefer external AE repo, fall back to lightweight numpy AE
    - ``external``: require ``external/Autoencoders-in-Spike-Sorting``
    - ``numpy``: use lightweight numpy implementation shipped in this repo

    中文：
    这里描述的是“前端编码器”怎么把 waveform 压成 bits。
    毕设主角不是 encoder，但它决定了后面 CAM 能看到什么输入。
    """

    method: str = "ae"
    backend: str = "auto"
    code_size: int = 16
    cache_dir: str = "results/cache/encoded_cache"
    reuse_cache: bool = True
    force_reencode: bool = False
    artifact_path: Optional[str] = None
    use_artifact: bool = True
    save_artifact: bool = True
    force_retrain_artifact: bool = False
    fit_scope: str = "all"
    ae_type: str = "normal"
    epochs: int = 8
    layers: List[int] = field(default_factory=lambda: [64, 32])
    scale: str = "robust"
    learning_rate: float = 0.01
    batch_size: int = 128
    binarize_mode: str = "median"
    binarize_threshold: float = 0.0
    verbose: int = 1


@dataclass
class TemplateInitConfig:
    """Initial template construction settings.

    中文：控制 warmup 阶段如何生成 CAM 初始模板。
    """

    method: str = "majority_vote"
    stable_mask_threshold: float = 0.85
    multi_template_per_unit: int = 2


@dataclass
class MatcherConfig:
    """Matching strategy settings.

    中文：控制 CAM 如何把一条输入 bits 和当前模板做匹配。
    """

    method: str = "hamming_nearest"
    threshold: float = 4.0
    min_margin: float = 2.0
    min_accept_margin: float = 1.0


@dataclass
class UpdateConfig:
    """Dynamic template update settings.

    中文：控制 CAM 在匹配之后是否更新模板，以及怎样更新。
    """

    method: str = "none"
    max_confidence: int = 12
    alpha: float = 0.05
    lr: float = 0.15
    max_conf: float = 5.0
    min_weight: float = 0.25
    flip_threshold: float = 0.9
    split_threshold: float = 6.0
    margin_band: float = 1.0
    cooldown_steps: int = 50
    min_margin: float = 2.0
    allow_evict: bool = False


@dataclass
class CamConfig:
    """CAM capacity and row-management settings.

    中文：控制 CAM 能存多少行模板、满了以后如何管理行。
    """

    capacity: Optional[int] = None
    capacity_factor: float = 1.0
    extra_rows: int = 0
    eviction_policy: str = "least_used"
    memory_subset: MemorySubsetConfig = field(default_factory=MemorySubsetConfig)


@dataclass
class EvaluationConfig:
    """Evaluation protocol settings.

    中文：控制实验协议，尤其是 chronological online evaluation。
    """

    mode: str = "chronological"
    warmup_ratio: float = 0.25
    random_train_frac: float = 0.7
    window_size: int = 500
    store_curves: bool = True


@dataclass
class ResultConfig:
    """Result saving settings.

    中文：控制实验结果保存目录和保存哪些文件。
    """

    results_dir: str = "results"
    save_predictions: bool = True
    save_curves: bool = True
    save_confusion: bool = True
    copy_encoded_dataset: bool = False


@dataclass
class VariantSpec:
    """One named experiment variant inside a config suite.

    中文：一个 config 里可以放多个变体，便于一次性比较多种方法。
    """

    name: str
    description: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VariantSpec":
        payload = dict(data)
        name = str(payload.pop("name"))
        description = str(payload.pop("description", ""))
        return cls(name=name, description=description, overrides=payload)


@dataclass
class ExperimentConfig:
    """Typed configuration for one concrete experiment run.

    中文：这是“单个具体实验运行实例”的完整配置对象。
    """

    experiment_name: str
    description: str = ""
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    template_init: TemplateInitConfig = field(default_factory=TemplateInitConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    cam: CamConfig = field(default_factory=CamConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    results: ResultConfig = field(default_factory=ResultConfig)
    variant_name: str = "default"
    variant_description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        payload = dict(data)
        return cls(
            experiment_name=str(payload["experiment_name"]),
            description=str(payload.get("description", "")),
            seed=int(payload.get("seed", 42)),
            dataset=DatasetConfig(
                **{
                    **payload.get("dataset", {}),
                    "preprocess": PreprocessConfig(**payload.get("dataset", {}).get("preprocess", {})),
                    "waveform": WaveformConfig(**payload.get("dataset", {}).get("waveform", {})),
                    "subset": SubsetConfig(**payload.get("dataset", {}).get("subset", {})),
                    "sampling": SamplingConfig(**payload.get("dataset", {}).get("sampling", {})),
                }
            ),
            encoder=EncoderConfig(**payload.get("encoder", {})),
            template_init=TemplateInitConfig(**payload.get("template_init", {})),
            matcher=MatcherConfig(**payload.get("matcher", {})),
            update=UpdateConfig(**payload.get("update", {})),
            cam=CamConfig(
                **{
                    **payload.get("cam", {}),
                    "memory_subset": MemorySubsetConfig(**payload.get("cam", {}).get("memory_subset", {})),
                }
            ),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
            results=ResultConfig(**payload.get("results", {})),
            variant_name=str(payload.get("variant_name", "default")),
            variant_description=str(payload.get("variant_description", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config into a nested dictionary.

        中文：把 dataclass 再转回普通字典，便于保存到结果目录。
        """

        return json_ready(asdict(self))


@dataclass
class ExperimentSuiteConfig:
    """Top-level config loaded from a JSON file.

    中文：表示一个 JSON 配置文件整体，里面可能包含多个 variant。
    """

    experiment_name: str
    description: str = ""
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    template_init: TemplateInitConfig = field(default_factory=TemplateInitConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    cam: CamConfig = field(default_factory=CamConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    results: ResultConfig = field(default_factory=ResultConfig)
    variants: List[VariantSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSuiteConfig":
        payload = dict(data)
        variants = [VariantSpec.from_dict(v) for v in payload.get("variants", [])]
        base = ExperimentConfig.from_dict(payload)
        return cls(
            experiment_name=base.experiment_name,
            description=base.description,
            seed=base.seed,
            dataset=base.dataset,
            encoder=base.encoder,
            template_init=base.template_init,
            matcher=base.matcher,
            update=base.update,
            cam=base.cam,
            evaluation=base.evaluation,
            results=base.results,
            variants=variants,
        )

    def to_base_dict(self) -> Dict[str, Any]:
        """Return the suite without variant-specific overrides."""

        payload = ExperimentConfig(
            experiment_name=self.experiment_name,
            description=self.description,
            seed=self.seed,
            dataset=self.dataset,
            encoder=self.encoder,
            template_init=self.template_init,
            matcher=self.matcher,
            update=self.update,
            cam=self.cam,
            evaluation=self.evaluation,
            results=self.results,
            variant_name="default",
            variant_description="",
        ).to_dict()
        payload.pop("variant_name", None)
        payload.pop("variant_description", None)
        return payload


def load_config(config_path: str) -> ExperimentSuiteConfig:
    """Load a JSON config file into an :class:`ExperimentSuiteConfig`.

    中文：读取一个实验配置文件。
    """

    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = strip_comment_fields(json.load(handle))
    return ExperimentSuiteConfig.from_dict(payload)


def expand_variants(suite: ExperimentSuiteConfig) -> List[ExperimentConfig]:
    """Expand a config suite into one or more concrete experiment configs.

    中文：把 base config 和各个 variant 合并，展开成真正要跑的实验列表。
    """

    base = suite.to_base_dict()
    if not suite.variants:
        merged = deepcopy(base)
        merged["variant_name"] = "default"
        merged["variant_description"] = ""
        return [ExperimentConfig.from_dict(merged)]

    configs: List[ExperimentConfig] = []
    for variant in suite.variants:
        merged = deepcopy(base)
        deep_update(merged, variant.overrides)
        merged["variant_name"] = variant.name
        merged["variant_description"] = variant.description
        configs.append(ExperimentConfig.from_dict(merged))
    return configs


def ensure_parent(path: Path) -> None:
    """Create the parent directory of ``path`` if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Save a JSON file with consistent formatting."""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, indent=2, sort_keys=True)


__all__ = [
    "PROJECT_ROOT",
    "CamConfig",
    "DatasetConfig",
    "EncoderConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExperimentSuiteConfig",
    "MatcherConfig",
    "MemorySubsetConfig",
    "ResultConfig",
    "SamplingConfig",
    "SubsetConfig",
    "TemplateInitConfig",
    "UpdateConfig",
    "VariantSpec",
    "WaveformConfig",
    "deep_update",
    "expand_variants",
    "json_ready",
    "load_config",
    "project_path",
    "resolve_path",
    "save_json",
    "strip_comment_fields",
]
