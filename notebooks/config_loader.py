import yaml
from pathlib import Path
from dataclasses import dataclass, asdict


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    temperature: float
    max_tokens: int
    num_ctx: int = 4096
    reasoning: bool = True
    keep_alive: str = "5m"


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llm_section = raw.get("llm", {})
    llm = LLMConfig(
        backend=llm_section["backend"],
        model=llm_section["model"],
        temperature=float(llm_section["temperature"]),
        max_tokens=int(llm_section["max_tokens"]),
        num_ctx=int(llm_section.get("num_ctx", 4096)),
        reasoning=bool(llm_section.get("reasoning", True)),
        keep_alive=llm_section.get("keep_alive", "5m"),
    )
    return AppConfig(llm=llm)

def load_config_as_dict(path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    return config_as_dict(load_config(path))


def config_as_dict(config: AppConfig) -> dict:
    return asdict(config)


if __name__ == "__main__":
    cfg = load_config()
    print(config_as_dict(cfg))
    print(load_config_as_dict())
