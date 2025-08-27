'''
# ----- flash_attn stub begin -----
import os, sys, tempfile, textwrap
stub_root = os.path.join(tempfile.gettempdir(), "stub_flash_attn")
pkg_dir = os.path.join(stub_root, "flash_attn")
os.makedirs(pkg_dir, exist_ok=True)
with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
    f.write(textwrap.dedent("""
    def flash_attn_func(*args, **kwargs):
        raise RuntimeError("flash_attn is stubbed. Use sdpa.")
    def flash_attn_varlen_func(*args, **kwargs):
        raise RuntimeError("flash_attn is stubbed. Use sdpa.")
    """))
with open(os.path.join(pkg_dir, "bert_padding.py"), "w") as f:
    f.write(textwrap.dedent("""
    def index_first_axis(x, idx):
        return x[idx]
    def pad_input(*args, **kwargs):
        raise RuntimeError("flash_attn pad_input stubbed")
    def unpad_input(*args, **kwargs):
        raise RuntimeError("flash_attn unpad_input stubbed")
    """))
sys.path.insert(0, stub_root)
# ----- flash_attn stub end -----
'''
from transformers import AutoConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
# Load the tokenizer and model
repo_name = "nvidia/Hymba-1.5B-Base"
cfg = AutoConfig.from_pretrained(repo_name, trust_remote_code=True)
# 둘 중 하나 사용
print(cfg._attn_implementation)
cfg.attn_implementation = "sdpa"
cfg._attn_implementation = "sdpa"   # 권장
# cfg.attn_implementation = "eager"
model = AutoModelForCausalLM.from_pretrained(
    repo_name,
    config=cfg,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)
model = model.cuda().to(torch.float16)
attn = model.model.layers[0].mamba.self_attn  # 첫 레이어 어텐션
print('mamba_d_state : ', model.config.mamba_d_state)
print('intermediate_size : ',model.config.intermediate_size)
print('layer_type : ',model.config.layer_type)
print("W (sliding_window)      =", model.config.sliding_window)   # None이면 SWA 없음
print("global_attn_idx (no SWA) =", model.config.global_attn_idx) # 여기에 들어간 레이어는 창 제한 안 받음
print("num_attention_heads      =", model.config.num_attention_heads)
print("num_key_value_heads      =", model.config.num_key_value_heads)
print("head_dim (Q/K default)   =", model.config.hidden_size // model.config.num_attention_heads)
print("kq_head_dim (actual)     =", getattr(attn, "kq_head_dim", None))
print("v_head_dim (actual)      =", getattr(attn, "v_head_dim", None))  # 보통 = (mamba_expand*hidden_size)/num_heads
print("num_memory_tokens (m)    =", model.config.num_memory_tokens)
print("interspersed_every       =", model.config.memory_tokens_interspersed_every)
prompt = 'What is the chatGPT? '
# Chat with Hymba
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=10).to("cuda")
outputs = model.generate(**inputs, max_length=64, do_sample=False, use_cache=True)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Model response: {response}")
