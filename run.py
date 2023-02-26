import lm_eval

model = lm_eval.get_model("hf-causal", pretrained="distilgpt2", device="cpu")
tasks = lm_eval.get_task_list(
    "boolq",
    template_names=['yes_no_question'])
results = lm_eval.evaluate(model=model, tasks=tasks)
