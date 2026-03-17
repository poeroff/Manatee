import random
import numpy as np
import torch
import sys
import json
import os

sys.path.insert(0, 'Manatee/src')
from vae import VAE

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ── VAE 모델 로드 ──────────────────────────────────────────
genes = open('Manatee/GSE72857/processed/genes.txt').read().splitlines()
tfs   = open('Manatee/GSE72857/processed/tfs.txt').read().splitlines()
mask  = np.isin(np.array(genes), np.array(tfs))

vae = VAE(mask_tf=mask, depth=3)
vae.load_state_dict(torch.load('Manatee/data/model/embryo_fold_2.pt', map_location='cpu'))
vae.eval()

# ── 데이터 및 셀 이름 로드 ─────────────────────────────────
x = np.genfromtxt('Manatee/data/combined_data_x.csv.gz', delimiter=' ')
stage_sizes = {'E3.5': 90, 'E4.5': 67, 'E5.25': 331, 'E5.5': 464, 'E6.25': 321}
labels = np.concatenate([np.full(n, s) for s, n in stage_sizes.items()])

e35_cellnames    = open('Manatee/GSE72857/processed/seurat_E3.5_cellnames.txt').read().splitlines()
cell_name_to_idx = {name: i for i, name in enumerate(e35_cellnames)}
tf_name_to_idx   = {tf: i for i, tf in enumerate(tfs)}

with torch.no_grad():
    _, z_all, _, _ = vae.forward(torch.FloatTensor(x))
z_all_np  = z_all.numpy()
centroids = {s: z_all_np[labels == s].mean(axis=0) for s in stage_sizes}

os.makedirs('Manatee/output', exist_ok=True)


def predict_celltype(z_vec):
    dists = {s: float(np.linalg.norm(z_vec - centroids[s])) for s in centroids}
    return min(dists, key=dists.get), dists


def _get_z(cell_name: str):
    idx  = cell_name_to_idx[cell_name]
    cell = x[idx].reshape(1, -1)
    with torch.no_grad():
        x_rec, _, mu, _ = vae.forward(torch.FloatTensor(cell))
        mu = torch.nn.functional.relu(mu)  # z와 동일하게 non-negative 적용
    return mu, x_rec


# ── 출력 헬퍼 ──────────────────────────────────────────────
def _print_activation(data: dict):
    print(f"\n{'='*55}")
    print(f"  세포: {data['cell_name']}")
    print(f"  예측 발생 단계: {data['predicted_stage']}")
    dists = "  ".join(f"{s}:{d:.4f}" for s, d in data["stage_distances"].items())
    print(f"  단계별 거리: {dists}")
    print(f"  {'─'*50}")
    print(f"  {'순위':<4} {'TF':<12} {'활성도':>8}")
    print(f"  {'─'*50}")
    for t in data["activated_tfs"]:
        print(f"  {t['rank']:<4} {t['tf']:<12} {t['activation']:>8.4f}")
    print(f"{'='*55}\n")


def _print_perturb(data: dict):
    print(f"\n{'='*60}")
    print(f"  세포: {data['cell_name']}")
    print(f"  TF: {data['tf_name']}  |  "
          f"조작 전: {data['tf_activation_orig']:.4f}  →  조작 후: {data['tf_activation_after']:.4f}")
    stage_change = (f"{data['stage_before']} → {data['stage_after']}  "
                    f"({'변화 있음' if data['stage_changed'] else '변화 없음'})")
    print(f"  발생 단계: {stage_change}")
    print(f"\n  [변동 유전자 상위 10개]")
    print(f"  {'유전자':<12} {'원본':>8} {'교란후':>8} {'delta':>8}")
    print(f"  {'─'*42}")
    for g in data["top10_changed_genes"]:
        print(f"  {g['gene']:<12} {g['orig_expr']:>8.4f} {g['perturb_expr']:>8.4f} {g['delta']:>+8.4f}")
    print(f"\n  [교란 후 TF 활성도 상위 10개]")
    print(f"  {'순위':<4} {'TF':<12} {'활성도':>8}")
    print(f"  {'─'*28}")
    for t in data["top10_tfs_after"]:
        print(f"  {t['rank']:<4} {t['tf']:<12} {t['activation']:>8.4f}")
    print(f"\n  저장된 파일: {', '.join(data['saved_csv'])}")
    print(f"{'='*60}\n")


# ── Tool 0: 랜덤 세포 선택 ────────────────────────────────
@tool
def get_random_cell_activation() -> str:
    """사용자가 무작위 세포를 요청할 때 사용합니다. 랜덤으로 세포를 하나 선택해 TF 활성도와 발생 단계를 반환합니다."""
    cell_name = random.choice(e35_cellnames)
    print(f"  [랜덤 선택] {cell_name}")
    return get_cell_activation.invoke(cell_name)


# ── Tool 1: 세포 TF 활성도 전체 조회 ──────────────────────
@tool
def get_cell_activation(cell_name: str) -> str:
    """세포 이름으로 전사인자(TF) 활성도 전체와 예측 발생 단계를 반환합니다.
    세포의 TF 활성도 조회, 발생 단계 예측이 필요할 때 사용하세요."""
    if cell_name not in cell_name_to_idx:
        return f"오류: '{cell_name}' 세포를 찾을 수 없습니다. 예시: {e35_cellnames[0]}"

    z, _ = _get_z(cell_name)
    z_np = z.numpy()[0]
    sorted_idx = np.argsort(z_np)[::-1]
    ct, dists = predict_celltype(z_np)

    all_tfs = [
        {"rank": i + 1, "tf": tfs[i_], "activation": round(float(z_np[i_]), 4)}
        for i, i_ in enumerate(sorted_idx)
    ]
    _print_activation({"cell_name": cell_name, "predicted_stage": ct,
                        "stage_distances": dists, "activated_tfs": all_tfs})

    # LLM에는 상위 20개만 전달 (전체는 터미널에 출력됨)
    summary = {
        "cell_name": cell_name,
        "predicted_stage": ct,
        "stage_distances": {s: round(d, 4) for s, d in dists.items()},
        "top20_activated_tfs": all_tfs[:20],
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ── Tool 2: TF 교란 실험 (knockout / 값 조절) ─────────────
@tool
def perturb_tf(cell_name: str, tf_name: str, new_value: float) -> str:
    """특정 세포에서 전사인자(TF)를 지정 값으로 조절한 결과를 반환합니다.
    new_value=0.0 이면 knockout(비활성화), 현재보다 큰 값은 과발현, 작은 값은 억제입니다.
    TF 교란/knockout/과발현/억제 실험이 필요할 때 사용하세요.
    결과 CSV는 Manatee/output/ 에 저장됩니다."""
    if cell_name not in cell_name_to_idx:
        return f"오류: '{cell_name}' 세포를 찾을 수 없습니다."
    if tf_name not in tf_name_to_idx:
        return f"오류: '{tf_name}' TF를 찾을 수 없습니다."

    tf_idx = tf_name_to_idx[tf_name]

    z, x_rec_orig = _get_z(cell_name)
    z_np_orig     = z.numpy()[0]
    orig_value    = float(z_np_orig[tf_idx])
    ct_before, dists_before = predict_celltype(z_np_orig)

    z_perturb = z.clone()
    z_perturb[0, tf_idx] = new_value

    with torch.no_grad():
        x_rec_perturb = vae.decode(z_perturb)

    with torch.no_grad():
        _, z_after, _, _ = vae.forward(x_rec_perturb)
    z_np_after = z_after.numpy()[0]
    ct_after, dists_after = predict_celltype(z_np_after)

    diff        = x_rec_perturb.numpy()[0] - x_rec_orig.numpy()[0]
    top_changed = np.argsort(np.abs(diff))[::-1][:10]
    top10_after = np.argsort(z_np_after)[::-1][:10]

    tag = f"{cell_name}_{tf_name}_{new_value}"
    np.savetxt(f'Manatee/output/{tag}_orig.csv',
               x_rec_orig.numpy(), delimiter=',',
               header=','.join(genes), comments='')
    np.savetxt(f'Manatee/output/{tag}_perturb.csv',
               x_rec_perturb.numpy(), delimiter=',',
               header=','.join(genes), comments='')
    np.savetxt(f'Manatee/output/{tag}_diff.csv',
               diff.reshape(1, -1), delimiter=',',
               header=','.join(genes), comments='')

    orig_arr    = x_rec_orig.numpy()[0]
    perturb_arr = x_rec_perturb.numpy()[0]

    data = {
        "cell_name":   cell_name,
        "tf_name":     tf_name,
        "tf_activation_orig":  round(orig_value, 4),
        "tf_activation_after": round(new_value, 4),
        "stage_before":  ct_before,
        "stage_after":   ct_after,
        "stage_changed": ct_before != ct_after,
        "top10_changed_genes": [
            {
                "gene":        genes[i],
                "orig_expr":   round(float(orig_arr[i]),    4),
                "perturb_expr":round(float(perturb_arr[i]), 4),
                "delta":       round(float(diff[i]),        4),
            }
            for i in top_changed
        ],
        "top10_tfs_after": [
            {"rank": r + 1, "tf": tfs[i_], "activation": round(float(z_np_after[i_]), 4)}
            for r, i_ in enumerate(top10_after)
        ],
        "saved_csv": [f"{tag}_orig.csv", f"{tag}_perturb.csv", f"{tag}_diff.csv"],
    }
    _print_perturb(data)
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── LLM 로드 ──────────────────────────────────────────────
chat_model = ChatOllama(model="qwen3:8b", temperature=0, think=False, num_ctx=8192)


# ── 에이전트 설정 ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a mouse embryo single-cell RNA-seq analysis expert.

LANGUAGE RULE — ABSOLUTE PRIORITY:
- You MUST respond ONLY in Korean (한국어).
- NEVER use Chinese (中文) under any circumstances. Not even a single Chinese character.
- Scientific terms may appear in English only when necessary.

CRITICAL RULES — you MUST follow these without exception:
1. NEVER answer from memory or make up information about cells or TFs.
2. You MUST call a tool for EVERY user request. No exceptions.
3. Cell activation / stage query → call get_cell_activation(cell_name=...)
4. TF perturbation (knockout / overexpression / suppression) → call perturb_tf(cell_name=..., tf_name=..., new_value=...)
5. After receiving the tool result, interpret the biological meaning in Korean. Do not repeat numbers."""

agent = create_agent(
    model=chat_model,
    tools=[get_random_cell_activation, get_cell_activation, perturb_tf],
    system_prompt=SYSTEM_PROMPT,
)

# ── 대화 히스토리 저장소 ───────────────────────────────────
_session_store: dict[str, ChatMessageHistory] = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent,
    _get_session_history,
    input_messages_key="messages",
)


def run_agent(query: str, session_id: str = "default") -> str:
    result = agent_with_history.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"session_id": session_id}, "recursion_limit": 10},
    )
    return result["messages"][-1].content


def clear_history(session_id: str = "default"):
    """대화 히스토리 초기화"""
    if session_id in _session_store:
        _session_store[session_id].clear()


# ── 메인 루프 ──────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        query = input("질문: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "종료"):
            break

        answer = run_agent(query)
        print(f"\n[답변]\n{answer}\n")
