import lm_eval
import json
import os
import csv

max_checks = 10000000

tasks = {
    "boolq": lm_eval.get_task_list(
        "boolq",
        template_names=['yes_no_question'])
}

for task in tasks.values():
    task[0].dataset = task[0].dataset.filter(lambda _, i: i < max_checks, with_indices=True)

def bench(model_name, task="boolq"):
    outfile = f"results/{model_name.replace('/','-')}-{task}.json"

    if (os.path.isfile(outfile)):
        print(f"{task} for {model_name} already complete ({outfile})")
        return
    else:
        print(f"Running {task} benchmark on {model_name}")

    if 't5' in model_name or 'bart' in model_name:
        model = lm_eval.get_model("hf-seq2seq", pretrained=model_name, device="cpu")
    else:
        model = lm_eval.get_model("hf-causal", pretrained=model_name, device="cpu")

    results = lm_eval.evaluate(model=model, tasks=tasks[task])

    with open(outfile, 'w') as out:
        out.write(json.dumps(results))

if __name__ == '__main__':
    for model in csv.DictReader(open('open-models.csv')):
        if int(model['params']) <= 1000:
            bench(model['model'])
