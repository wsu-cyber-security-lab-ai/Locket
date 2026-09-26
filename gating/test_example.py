base_model_name = "meta-llama/Llama-3.2-1B"

scratch = '../local_models'
adapter_masked_name = "lora_masked"
adapter_unprotected_name = "lora_unprotected"
adapter_masked_path = f"{scratch}/{adapter_masked_name}"
adapter_unprotected_path = f"{scratch}/{adapter_unprotected_name}"
device = "cuda"


target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

gc.collect()
torch.cuda.empty_cache()

# Ensure the HF API key is set
hf_api_key = os.getenv("HF_API_KEY")
if not hf_api_key:
    exit(1)


tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_auth_token=hf_api_key,
    device_map=device,
    # subfolder="tokenizer",
    # revision=args.revision,
    use_fast=False,
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    # special_tokens_dict = {'additional_special_tokens': ['foo']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    # quantization_config=bnb_config,
    device_map="cuda",  # or "cuda" if you want to force GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.resize_token_embeddings(len(tokenizer))
model.to(device)

lora_list = [adapter_masked_path, adapter_unprotected_path]
LoRA_root_path = '/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/cv_multi_lora/dreambooth/cpkt'

number_of_lora = len(lora_list)

def set_and_load_loras(unet, text_encoder, dir, adapter_name):
    load_adapter(unet, text_encoder, dir, adapter_name=adapter_name)
    set_adapter(unet, adapter_name=adapter_name)
    set_adapter(text_encoder, adapter_name=adapter_name)

def load_adapter(unet, text_encoder, ckpt_dir, adapter_name):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)

def set_adapter(net, adapter_name):
    net.set_adapter(adapter_name)

for lora in lora_list:
    set_and_load_loras(unet, text_encoder, os.path.join(LoRA_root_path, lora), adapter_name=lora)


unet.delete_adapter('default')

if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    # below fails when using lora so commenting it out
    if args.train_text_encoder and not args.use_lora:
        text_encoder.gradient_checkpointing_enable()

if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# Optimizer creation
params_to_optimize = (
    itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
)
optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)