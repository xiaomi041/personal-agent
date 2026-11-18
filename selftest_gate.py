"""检索门控的纯函数自检（不联网、不需要向量服务）。

`knowledge._gate_scores` 是「这段检索结果算不算有答案」的唯一决策点，
所以用纯分数列表把它钉住，避免以后调参把行为改坏。

运行：python selftest_gate.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DATA_DIR", os.path.join(HERE, ".tmp-gate-selftest"))
os.environ.pop("DOC_MIN_SCORE", None)
os.environ.pop("DOC_GATE_FLOOR", None)
os.environ.pop("DOC_GATE_MARGIN", None)
sys.path.insert(0, HERE)

import knowledge  # noqa: E402

PASS = 0
FAIL = 0


def ok(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def mask_of(scores, min_score=None):
    return knowledge._gate_scores(scores, min_score)


def main() -> int:
    print("=" * 70)
    print("检索门控自检")
    print("=" * 70)

    # 1) 高置信：全部高于 0.42 → strict，全部放行
    m, mode = mask_of([0.80, 0.62, 0.55, 0.44])
    ok("高置信 → strict", mode == "strict" and all(m), f"{mode} {m}")

    # 2) 高置信时低于阈值的尾部要被截掉
    m, mode = mask_of([0.80, 0.41, 0.30])
    ok("高置信仍截掉低分尾部", mode == "strict" and m == [True, False, False], f"{m}")

    # 3) 改写题的典型情形：0.41 不够绝对线，但比同批明显高 → relaxed 放行
    m, mode = mask_of([0.41, 0.33, 0.32, 0.31])
    ok("0.41 脱颖而出 → relaxed 放行", mode == "relaxed" and m[0], f"{mode} {m}")

    # 4) 分数都挤在一起 → 没有真正匹配的片段，拒答
    m, mode = mask_of([0.41, 0.40, 0.395, 0.39])
    ok("分数挤在一起 → reject", mode == "reject" and not any(m), f"{mode} {m}")

    # 5) 最高分低于地板 → 拒答（哪怕相对差距大）
    m, mode = mask_of([0.22, 0.10, 0.09])
    ok("低于地板 → reject", mode == "reject" and not any(m), f"{mode} {m}")

    # 6) 空候选 → 空掩码
    m, mode = mask_of([])
    ok("空候选 → 空", m == [] and mode == "empty", f"{mode} {m}")

    # 7) relaxed 只保留贴近顶部的，不把 0.31 这种也带上
    #    注意别用 0.50——那已经过了绝对线，走的是 strict 分支
    m, mode = mask_of([0.40, 0.31, 0.30, 0.29])
    ok("relaxed 只留顶部附近", mode == "relaxed" and m[0] and not m[1], f"{mode} {m}")

    # 8) 绝对阈值可被参数覆盖（测试用）
    m, mode = mask_of([0.30, 0.29], min_score=0.28)
    ok("min_score 可覆盖", mode == "strict" and m == [True, True], f"{m}")

    # 9) 环境变量生效（同样要落在绝对线以下，否则走 strict）
    os.environ["DOC_GATE_FLOOR"] = "0.55"
    m, mode = mask_of([0.40, 0.30, 0.29])
    ok("地板调高后拒答", mode == "reject" and not any(m), f"{mode} {m}")
    os.environ.pop("DOC_GATE_FLOOR")

    print("-" * 70)
    print(f"通过 {PASS} / 失败 {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
