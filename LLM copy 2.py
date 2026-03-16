import numpy as np
import torch
import sys
import json
import os

sys.path.insert(0, 'Manatee/src')
from vae import VAE

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

e35_cellnames   = open('Manatee/GSE72857/processed/seurat_E3.5_cellnames.txt').read().splitlines()
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
    """세포 이름으로 VAE 잠재 벡터와 원본 재구성 발현을 반환합니다."""
    idx  = cell_name_to_idx[cell_name]
    cell = x[idx].reshape(1, -1)
    with torch.no_grad():
        x_rec, z, _, _ = vae.forward(torch.FloatTensor(cell))
    return z, x_rec


# ── Tool 1: 세포 TF 활성도 조회 ───────────────────────────
@tool
def get_cell_activation(cell_name: str) -> str:
    """세포 이름으로 전사인자(TF) 활성도 상위 10개와 예측 발생 단계를 반환합니다."""
    if cell_name not in cell_name_to_idx:
        return f"오류: '{cell_name}' 세포를 찾을 수 없습니다. 예시: {e35_cellnames[0]}"

    z, _ = _get_z(cell_name)
    z_np = z.numpy()[0]
    top10_idx = np.argsort(z_np)[::-1][:10]
    ct, dists = predict_celltype(z_np)

    return json.dumps({
        "cell_name": cell_name,
        "predicted_stage": ct,
        "stage_distances": {s: round(d, 4) for s, d in dists.items()},
        "top10_activated_tfs": [
            {"rank": i + 1, "tf": tfs[i_], "activation": round(float(z_np[i_]), 4)}
            for i, i_ in enumerate(top10_idx)
        ],
    }, ensure_ascii=False, indent=2)


# ── Tool 2: TF 교란 실험 (knockout / 값 조절) ─────────────
@tool
def perturb_tf(cell_name: str, tf_name: str, new_value: float) -> str:
    """특정 세포에서 전사인자(TF)를 지정 값으로 조절한 결과를 반환합니다.
    new_value=0.0 이면 knockout, 양수는 과발현, 음수는 억제입니다.
    결과 CSV는 Manatee/output/ 에 저장됩니다."""
    if cell_name not in cell_name_to_idx:
        return f"오류: '{cell_name}' 세포를 찾을 수 없습니다."
    if tf_name not in tf_name_to_idx:
        return f"오류: '{tf_name}' TF를 찾을 수 없습니다."

    tf_idx = tf_name_to_idx[tf_name]

    # 원본
    z, x_rec_orig = _get_z(cell_name)
    z_np_orig     = z.numpy()[0]
    orig_value    = float(z_np_orig[tf_idx])
    ct_before, dists_before = predict_celltype(z_np_orig)

    # 교란
    z_perturb = z.clone()
    z_perturb[0, tf_idx] = new_value

    with torch.no_grad():
        x_rec_perturb = vae.decode(z_perturb)

    # 교란 후 VAE 재통과 → 새 TF 활성도
    with torch.no_grad():
        _, z_after, _, _ = vae.forward(x_rec_perturb)
    z_np_after = z_after.numpy()[0]
    ct_after, dists_after = predict_celltype(z_np_after)

    # 발현 변화 상위 10개
    diff        = x_rec_perturb.numpy()[0] - x_rec_orig.numpy()[0]
    top_changed = np.argsort(np.abs(diff))[::-1][:10]
    top10_after = np.argsort(z_np_after)[::-1][:10]

    # CSV 저장
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

    return json.dumps({
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
    }, ensure_ascii=False, indent=2)


# ── LLM 로드 ──────────────────────────────────────────────
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"LLM 로딩 중: {LLM_MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
llm_model.eval()

pipe = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=False,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
)
chat_model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
print("LLM 로딩 완료\n")


# ── LLM 쿼리 파싱 ─────────────────────────────────────────
def _extract_json(text: str):
    """중첩 JSON도 처리하는 괄호 깊이 기반 파서."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PARSE_SYSTEM = """\
사용자 쿼리에서 정보를 추출해 JSON만 출력하세요. 다른 텍스트는 절대 출력하지 마세요.

출력 스키마:
{{"cell_name": string|null, "tf_name": string|null, "op": "activation"|"knockout"|"modulate", "value": number|null, "direction": "up"|"down"|null}}

op 및 direction 규칙:
- 활성도/분석 조회만 → op: "activation", value: null, direction: null
- TF를 끄기/제거/knockout/비활성화 → op: "knockout", value: 0.0, direction: null
- TF를 특정 숫자로 설정 → op: "modulate", value: 그 숫자, direction: null
- TF를 올리기/높이기/증가/과발현/up → op: "modulate", value: null, direction: "up"
- TF를 낮추기/줄이기/억제/감소/down → op: "modulate", value: null, direction: "down"

예시:
입력: "E3.5_P8_Cell3_embryo2_single 세포의 활성도 알려줘"
출력: {{"cell_name": "E3.5_P8_Cell3_embryo2_single", "tf_name": null, "op": "activation", "value": null, "direction": null}}

입력: "E3.5_P8_Cell3_embryo2_single 세포에서 Hdgf 꺼줘"
출력: {{"cell_name": "E3.5_P8_Cell3_embryo2_single", "tf_name": "Hdgf", "op": "knockout", "value": 0.0, "direction": null}}

입력: "E3.5_P8_Cell3_embryo2_single 세포에서 Hdgf 0.2로 조작해줘"
출력: {{"cell_name": "E3.5_P8_Cell3_embryo2_single", "tf_name": "Hdgf", "op": "modulate", "value": 0.2, "direction": null}}

입력: "E3.5_P8_Cell5_embryo3_single 세포에서 Sox2 발현 올려줘"
출력: {{"cell_name": "E3.5_P8_Cell5_embryo3_single", "tf_name": "Sox2", "op": "modulate", "value": null, "direction": "up"}}

입력: "E3.5_P8_Cell7_embryo4_single에서 Nanog 억제해줘"
출력: {{"cell_name": "E3.5_P8_Cell7_embryo4_single", "tf_name": "Nanog", "op": "modulate", "value": null, "direction": "down"}}

입력: "E3.5_P8_Cell22_embryo5_single 세포에서 Ctcf를 완전히 제거해줘"
출력: {{"cell_name": "E3.5_P8_Cell22_embryo5_single", "tf_name": "Ctcf", "op": "knockout", "value": 0.0, "direction": null}}
"""

parse_prompt = ChatPromptTemplate.from_messages([
    ("system", PARSE_SYSTEM),
    ("human", "{query}"),
])


def parse_query(query: str) -> dict:
    """LLM few-shot으로 쿼리를 파싱합니다."""
    raw  = (parse_prompt | chat_model | StrOutputParser()).invoke({"query": query}).strip()
    print(f"[파서 출력] {raw}")

    data = _extract_json(raw)
    if not data:
        print("[파서 경고] JSON 파싱 실패, activation으로 처리")
        return {"cell_name": None, "tf_name": None, "op": "activation", "value": None}

    cell_name = data.get("cell_name")
    tf_name   = data.get("tf_name")
    op        = data.get("op", "activation")
    value     = data.get("value")
    direction = data.get("direction")

    # direction이 있고 value가 null이면 현재 활성도 기반으로 계산
    if op == "modulate" and value is None and direction and cell_name and tf_name:
        if cell_name in cell_name_to_idx and tf_name in tf_name_to_idx:
            z, _ = _get_z(cell_name)
            cur  = float(z.numpy()[0][tf_name_to_idx[tf_name]])
            value = cur * 2.0 if direction == "up" else cur * 0.5

    return {"cell_name": cell_name, "tf_name": tf_name, "op": op, "value": value}


# ── LangChain LCEL 체인 ────────────────────────────────────
activation_explain_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 마우스 배아 단세포 RNA-seq 데이터 분석 전문가입니다. "
     "수치 데이터는 이미 사용자에게 출력됐습니다. "
     "아래 결과를 보고 생물학적 의미를  한국어로 간략히 해석해주세요. "
     "숫자를 다시 반복하지 마세요."),
    ("human", "{tool_result}"),
])

perturb_explain_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 마우스 배아 단세포 RNA-seq 데이터 분석 전문가입니다. "
     "수치 데이터는 이미 사용자에게 출력됐습니다. "
     "아래 교란 실험 결과를 보고 생물학적 의미를 3~5문장으로 한국어로 간략히 해석해주세요. "
     "숫자를 다시 반복하지 마세요."),
    ("human", "{tool_result}"),
])
general_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 마우스 배아 단세포 RNA-seq 데이터 분석 전문가입니다. 한국어로 답하세요."),
    ("human", "{query}"),
])

def _print_activation(data: dict):
    print(f"\n{'='*55}")
    print(f"  세포: {data['cell_name']}")
    print(f"  예측 발생 단계: {data['predicted_stage']}")
    dists = "  ".join(f"{s}:{d:.4f}" for s, d in data["stage_distances"].items())
    print(f"  단계별 거리: {dists}")
    print(f"  {'─'*50}")
    print(f"  {'순위':<4} {'TF':<12} {'활성도':>8}")
    print(f"  {'─'*50}")
    for t in data["top10_activated_tfs"]:
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


activation_explain_chain = activation_explain_prompt | chat_model | StrOutputParser()
perturb_explain_chain    = perturb_explain_prompt    | chat_model | StrOutputParser()
general_chain            = general_prompt            | chat_model | StrOutputParser()


def run_agent(query: str) -> str:
    parsed = parse_query(query)
    cell   = parsed["cell_name"]
    tf     = parsed["tf_name"]
    op     = parsed["op"]
    val    = parsed["value"]

    print(f"[파싱] 세포={cell}, TF={tf}, 조작={op}, 값={val}")

    if not cell:
        return general_chain.invoke({"query": query})

    if op == "activation" or tf is None:
        print(f"[툴] get_cell_activation('{cell}')")
        result_str = get_cell_activation.invoke(cell)
        data = json.loads(result_str)
        _print_activation(data)
        return "\n[해석]\n" + activation_explain_chain.invoke({"tool_result": result_str})
    else:
        print(f"[툴] perturb_tf('{cell}', '{tf}', {val})")
        result_str = perturb_tf.invoke({"cell_name": cell, "tf_name": tf, "new_value": val})
        data = json.loads(result_str)
        _print_perturb(data)
        return "\n[해석]\n" + perturb_explain_chain.invoke({"tool_result": result_str})


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
