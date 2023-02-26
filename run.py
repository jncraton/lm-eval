import lm_eval
import json

tasks = lm_eval.get_task_list(
    "boolq",
    template_names=['yes_no_question'])

tasks[0].dataset = tasks[0].dataset.filter(lambda _, i: i < 32, with_indices=True)
def bench(model_name):
    if 't5' in model_name:
        model = lm_eval.get_model("hf-seq2seq", pretrained=model_name, device="cpu")
    else:
        model = lm_eval.get_model("hf-causal", pretrained=model_name, device="cpu")

    results = lm_eval.evaluate(model=model, tasks=tasks)
    print(results)

    with open(f"results/{model_name.replace('/','-')}.json", 'w') as out:
        out.write(json.dumps(results))

if __name__ == '__main__':
    bench('google/flan-t5-small')
    bench('google/flan-t5-base')
    bench('distilgpt2')
    bench('gpt2')
