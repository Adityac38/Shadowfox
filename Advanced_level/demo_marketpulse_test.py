import os
import re
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
FINETUNED_DIR = "./marketpulse_model"
MODEL_NAME = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1) 
MASK = tokenizer.mask_token
_num_re = re.compile(r"^\d+(\.\d+)?$") 
_alpha_re = re.compile(r"^[A-Za-z][A-Za-z\-']*$")
def _unique_keep(seq):
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out
def _looks_like_price_query(q: str) -> bool:
    q_low = q.lower()
    return ("rs" in q_low or "₹" in q_low or "price" in q_low) and MASK in q
def _clean_join(template: str, token: str) -> str:
    """Replace MASK with token and tidy spaces/punctuation."""
    s = template.replace(MASK, token.strip())
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"(?i)\brs\.\s*rs\b", "Rs.", s)
    return s[0].upper() + s[1:] if s else s
def _pick_tokens_for_single_mask(query: str, results, top_k_show=3):
    """
    Build predictions for single-mask queries.
    For price prompts, prefer numeric tokens and filter out 'rs'.
    """
    candidates = [r["token_str"].strip() for r in results]
    candidates = _unique_keep(candidates)
    candidates = [c for c in candidates if c not in {"rs", "##rs", "##s", ".", ",", "##."}]
    if _looks_like_price_query(query):
        numeric = [c for c in candidates if _num_re.fullmatch(c)]
        if not numeric:
            numeric_seq = []
            for r in results:
                seq_nums = re.findall(r"\d+(?:\.\d+)?", r.get("sequence", ""))
                numeric_seq.extend(seq_nums)
            numeric = _unique_keep(numeric_seq)
        tokens_to_use = numeric[:top_k_show] if numeric else candidates[:top_k_show]
    else:
        words = [c for c in candidates if _alpha_re.fullmatch(c)]
        tokens_to_use = words[:top_k_show] if words else candidates[:top_k_show]
    outs = [_clean_join(query, tok) for tok in tokens_to_use]
    return outs
def _pick_tokens_for_multi_mask(query: str, results_per_mask):
    """
    For multiple masks, take the best token per mask with light filtering.
    """
    chosen = []
    for mask_alternatives in results_per_mask:
        wordy = [alt["token_str"].strip() for alt in mask_alternatives
                 if alt["token_str"].strip() not in {"rs", ".", ",", "##."}
                 and (_alpha_re.fullmatch(alt["token_str"].strip()) or _num_re.fullmatch(alt["token_str"].strip()))]
        tok = wordy[0] if wordy else mask_alternatives[0]["token_str"].strip()
        chosen.append(tok)
    out = query
    for tok in chosen:
        out = out.replace(MASK, tok, 1)
    out = _clean_join(out, "")
    return [out]
def predict_masked_text(query: str):
    query = (query or "").strip()
    if not query:
        return "⚠️ Please enter a sentence with at least one [MASK] token."
    if MASK not in query:
        return f"⚠️ No {MASK} found. Example: `Street food item Vada Pav is sold at Rs. {MASK}.`"
    raw = fill_mask(query, top_k=15)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        preds = _pick_tokens_for_single_mask(query, raw, top_k_show=3)
    else:
        preds = _pick_tokens_for_multi_mask(query, raw)
    lines = "\n".join([f"- {p}" for p in preds if p])
    return f"### Predictions:\n{lines}" if lines else "No clean prediction could be generated."
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;'>Market Pulse BERT — Masked LM Demo</h1>")
    gr.Markdown("Enter a sentence with one or more `[MASK]` tokens. "
                "For price prompts, prefer *`Rs. [MASK]`* or *`₹ [MASK]`* (no extra period after the number).")
    with gr.Row():
        inp = gr.Textbox(
            label="Input",
            value=f"Street food item Vada Pav is sold at Rs. {MASK}.",
            lines=2,
        )
    with gr.Row():
        btn = gr.Button("Predict", variant="primary")
    out = gr.Markdown()
    gr.Examples(
        examples=[
            f"Street food item Vada Pav is sold at Rs. {MASK}.",
            f"Waste generated: {MASK} kg daily.",
            f"Hygiene tip: {MASK} {MASK} {MASK}.",
            f"Sustainability advice: {MASK} {MASK} {MASK}.",
        ],
        inputs=inp,
        label="Examples"
    )
    btn.click(predict_masked_text, inputs=inp, outputs=out)
if __name__ == "__main__":
    demo.launch()