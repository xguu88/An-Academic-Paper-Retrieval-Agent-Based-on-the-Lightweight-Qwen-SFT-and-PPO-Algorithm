"""集中实现 Qwen3 tokenizer/model 的严格本地加载与结构信息记录。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class QwenModelInfo:
    """保存本地 config 中与 SFT 工程相关的模型结构信息。"""

    model_type: str | None
    architectures: list[str]
    vocab_size: int | None
    hidden_size: int | None
    num_hidden_layers: int | None
    max_position_embeddings: int | None

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def _require_local_model(path: str | Path) -> Path:
    """在调用 Transformers 前验证本地目录，禁止 silent Hub fallback。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"QWEN MODEL NOT FOUND: {resolved}")
    return resolved


def _model_info(config: Any) -> QwenModelInfo:
    """从 AutoConfig 提取稳定结构字段。"""

    return QwenModelInfo(
        model_type=getattr(config, "model_type", None),
        architectures=list(getattr(config, "architectures", None) or []),
        vocab_size=getattr(config, "vocab_size", None),
        hidden_size=getattr(config, "hidden_size", None),
        num_hidden_layers=getattr(config, "num_hidden_layers", None),
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
    )


def load_qwen_tokenizer(#将文章数字化，不涉及语义，但是融合padding等信息
    model_path: str | Path,
) -> tuple[Any, QwenModelInfo]:
    """以 local_files_only=True 加载 tokenizer 和 config。"""

    local_path = _require_local_model(model_path)
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(#加载模型配置文件
        str(local_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(#加载分词器
        str(local_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer 同时缺少 pad_token_id 和 eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, _model_info(config)


def load_qwen_causal_lm(
    model_path: str | Path,
    *,
    dtype: Any | None = None,
    **model_kwargs: Any,
) -> tuple[Any, QwenModelInfo]:
    """严格本地加载模型，并兼容新旧 Transformers 的 dtype 参数名。
model_path
   ↓
检查必须是本地目录
   ↓
AutoConfig
读取模型配置
   ↓
确定 dtype
   ↓
AutoModelForCausalLM.from_pretrained
构建模型 + 加载权重
   ↓
如果 dtype 参数版本不兼容
改用 torch_dtype 再试
   ↓
返回 model + model_info

    """

    local_path = _require_local_model(model_path)
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(#返回配置对象，没有网络，没有权重
        str(local_path), local_files_only=True, trust_remote_code=False
    )
    legacy_dtype = model_kwargs.pop("torch_dtype", None)
    selected_dtype = dtype if dtype is not None else legacy_dtype
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        **model_kwargs,
    }
    if selected_dtype is not None:
        load_kwargs["dtype"] = selected_dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(str(local_path), **load_kwargs)#加入权重
    except TypeError as exc:
        # Transformers 旧版本尚不接受 dtype；仅针对该参数回退到 torch_dtype。
        if "dtype" not in load_kwargs or "dtype" not in str(exc):
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(str(local_path), **load_kwargs)
    return model, _model_info(config)
