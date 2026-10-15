"""无需模型、GPU 或网络的 Action 协议离线冒烟测试。"""

from pathlib import Path
import sys


# 直接运行脚本时将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.exceptions import InvalidActionError  # noqa: E402
from src.agent.parser import parse_action  # noqa: E402
from src.agent.state import AgentState  # noqa: E402
from src.agent.trajectory import AgentTrajectory, TrajectoryStep  # noqa: E402
from src.retrieval.citation_graph import CitationGraph  # noqa: E402


VALID_ACTIONS = {
    "RewriteQuery": '{"action":"RewriteQuery","query":"new query"}',
    "KeywordSearch": '{"action":"KeywordSearch","query":"keywords","top_k":10}',
    "SemanticSearch": '{"action":"SemanticSearch","query":"semantic","top_k":10}',
    "HybridSearch": '{"action":"HybridSearch","query":"hybrid","top_k":20}',
    "Rerank": '{"action":"Rerank","candidate_ids":["p1","p2"],"top_k":1}',
    "ExpandCitation": (
        '{"action":"ExpandCitation","paper_id":"p1","direction":"references"}'
    ),
    "ReadPaper": '{"action":"ReadPaper","paper_id":"p1"}',
    "Stop": '{"action":"Stop","candidate_ids":["p1"]}',
}


def main() -> None:
    """执行协议、引用方向与序列化的最小离线验证。"""

    print("ACTION PROTOCOL SMOKE TEST")
    print()
    parsed = {}
    for name, text in VALID_ACTIONS.items():
        parsed[name] = parse_action(text)
        print(f"[PASS] {name}")

    try:
        parse_action('{"action":"Unknown"}')
    except InvalidActionError:
        print("[PASS] Invalid action rejected")
    else:
        raise AssertionError("非法动作未被拒绝")

    graph = CitationGraph()
    graph.add_edge("source", "target")
    assert graph.get_references("source") == ["target"]
    assert graph.get_citations("target") == ["source"]
    print("[PASS] Citation direction")

    before = AgentState("smoke", "query", "query", remaining_budget=1.0)
    after = AgentState("smoke", "query", "query", step=1, remaining_budget=0.9)
    trajectory = AgentTrajectory(
        task_id="smoke",
        steps=[
            TrajectoryStep(
                step=1,
                state_before=before,
                action=parsed["SemanticSearch"],
                observation={"paper_ids": ["p1"]},
                state_after=after,
            )
        ],
    )
    import json

    json.dumps(trajectory.to_dict(), ensure_ascii=False)
    print("[PASS] Serialization")
    print()
    print("ACTION PROTOCOL SMOKE TEST PASS")


if __name__ == "__main__":
    main()
