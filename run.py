import lm_eval
import json

max_checks = 10000000

tasks = {
    "boolq": lm_eval.get_task_list(
        "boolq",
        template_names=['yes_no_question'])
}

for task in tasks.values():
    task[0].dataset = task[0].dataset.filter(lambda _, i: i < max_checks, with_indices=True)

def bench(model_name, task="boolq"):
    print(f"Running {task} benchmark on {model_name}")

    if 't5' in model_name or 'bart' in model_name:
        model = lm_eval.get_model("hf-seq2seq", pretrained=model_name, device="cpu")
    else:
        model = lm_eval.get_model("hf-causal", pretrained=model_name, device="cpu")

    results = lm_eval.evaluate(model=model, tasks=tasks[task])

    with open(f"results/{model_name.replace('/','-')}-{task}.json", 'w') as out:
        out.write(json.dumps(results))

if __name__ == '__main__':
    bench('google/mt5-small')
    bench('facebook/bart-base')
    bench('t5-small')
    bench('t5-base')
    bench('t5-large')
    bench('google/flan-t5-small')
    bench('google/flan-t5-base')
    bench('google/flan-t5-large')
    bench('distilgpt2')
    bench('gpt2')
    bench('gpt2-medium')
    bench('gpt2-large')
    bench('EleutherAI/gpt-neo-125M')
    bench('gpt2-xl')
    bench('EleutherAI/gpt-neo-1.3B')
