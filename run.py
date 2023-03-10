import lm_eval
import json
import os
import csv
import torch

max_checks = 10000000

task_config = {
    "cb": lm_eval.get_task_list(
        "cb",
        template_names=['based on the previous passage']),
    "copa": lm_eval.get_task_list(
        "copa",
        template_names=['more likely']),
    "boolq": lm_eval.get_task_list(
        "boolq",
        template_names=['yes_no_question']),
}

for task in task_config.values():
    task[0].dataset = task[0].dataset.filter(lambda _, i: i < max_checks, with_indices=True)

def bench(model_name, tasks=["cb", "copa", "boolq"], shotlist=[0]):
    for task in tasks:
        for shots in shotlist:
            if shots > 0:
                outfile = f"results/{model_name.replace('/','-')}-{task}-{shots}.json"
            else:
                outfile = f"results/{model_name.replace('/','-')}-{task}.json"

            if (os.path.isfile(outfile)):
                print(f"{task} for {model_name} ({shots}-shot) already complete ({outfile})")
                continue

            model_type = 'hf-seq2seq' if 't5' in model_name or 'bart' in model_name or 'tk' in model_name or 't0' in model_name else 'hf-causal'

            print(f"Running {task} benchmark on {model_name} ({model_type})")

            model = lm_eval.get_model(model_type, pretrained=model_name, device='cpu', dtype=torch.float32)

            results = lm_eval.evaluate(model=model, tasks=task_config[task], num_fewshot=shots)

            with open(outfile, 'w') as out:
                out.write(json.dumps(results))

if __name__ == '__main__':
    models = list(csv.DictReader(open('open-models.csv')))
    models.sort(key = lambda m: int(m['params']))
    for model in models:
        if int(model['params']) <= 3000:
            bench(model['model'])
