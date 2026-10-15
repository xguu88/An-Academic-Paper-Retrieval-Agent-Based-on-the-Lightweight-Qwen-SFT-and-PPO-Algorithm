"""从论文 JSONL 的 references 构建引用图并输出离线统计报告。"""

import argparse
import json
from pathlib import Path
import sys


# 允许从项目根目录直接执行脚本而无需安装本包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.citation_graph import build_citation_graph  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析语料输入与 JSON 报告输出路径。"""

    parser = argparse.ArgumentParser(description="构建离线论文引用图统计")
    parser.add_argument("--corpus", type=Path, required=True, help="论文 JSONL 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/citation_graph_report.json"),
        help="统计报告输出路径",
    )
    return parser.parse_args()


def main() -> None:
    """加载语料、构图并保存包含缺失目标的统计。"""

    args = parse_args()
    corpus = PaperCorpus.from_jsonl(args.corpus)
    _, stats = build_citation_graph(list(corpus))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print(f"Citation graph report saved: {args.output}")


if __name__ == "__main__":
    main()

