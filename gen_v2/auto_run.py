import subprocess
from concurrent.futures import ThreadPoolExecutor


model_parameters = ['GPT4o','Gemini']

commands = [
    [  # Reasoning
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-none',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-none',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '0',
        '--max_rows', '200',
        '--output_structure', 'index'
    ],
[  # Reasoning
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-none',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '0',
        '--max_rows', '200',
        '--output_structure', 'index'
    ],


]


def run_command_for_model(model):
    for command in commands:
        full_command = ['python', 'systematic_evaluation.py', '--models', model] + command
        print(f"Running command for model {model}: {' '.join(full_command)}")
        result = subprocess.run(full_command, capture_output=True, text=True)
        print(f"Standard Output for {model}:\n{result.stdout}")
        print(f"Standard Error for {model}:\n{result.stderr}")

with ThreadPoolExecutor(max_workers=len(model_parameters)) as executor:
    executor.map(run_command_for_model, model_parameters)
