# KAIST IC Lab. Internship summer (2024)

```bash
This is the official implementation of Prompt Modularization & Emotion Recognition with LLM, conducted during the KAIST IC Lab Internship Summer 2024.
```

## Before running the project, ensure you have the following setup:

1. Create account on [huggingface](https://huggingface.co/) and register organization "KAIST-IC-LAB721" or Creating an new organization on [huggingface](https://huggingface.co/) for data upload
2. Issuing API keys for closed-source LLMs ([OpenAI](https://platform.openai.com/docs/overview), [Gemini](https://ai.google.dev/gemini-api/docs/api-key))
3. Installing [Ollama](https://github.com/ollama/ollama) with Docker

## Setup environment
if you use Window and cannot install transformers, Install Rust.
Python version have to be 3.11.x~3.10.x. Pytorch does not support 3.13.0. You can install veriosn 3.10.14 or 3.11.6
If using Visual Studio Code instead of Pycharm, in file Efficient_auto_run change line
        full_command = ['python', 'systematic_evaluation.py', '--models', model] + command
-->     full_command = [sys.executable, 'systematic_evaluation.py', '--models', model] + command

## Data Preparation
Please execute each .py file individually in the preprocess_data folder. preprocessed data will be uploaded to HuggingFace. Initially, it will show error HuggingFace acc authentication, do following step.
Create account on HunggingFace and get Access Token.
Login with Access Token in your local.
  - pip install huggingface_hub
  - huggingface-cli login
  - paste your token.
If it keep showing your token error, try another way
  - Set the Environment Variable: $env:HUGGINGFACE_HUB_TOKEN = "your_api_token_here".
  - or run this script
    from huggingface_hub import HfApi, HfFolder
    HfFolder.save_token("your_api_token_here")
  - close and open Terminal and run preprocess file again.

The data folder includes the following datasets:

SDCNL: A dataset for classifying between depression and suicidal tendencies using web-scraped data. [SDCNL](https://github.com/ayaanzhaque/SDCNL)

CSSRS: Data based on the Columbia-Suicide Severity Rating Scale for assessing suicidal ideation and behavior. [CSSRS](https://paperswithcode.com/dataset/reddit-c-ssrs), [download](https://www.kaggle.com/datasets/thedevastator/c-ssrs-labeled-suicidality-in-500-anonymized-red) 

Dreaddit: A Reddit dataset for stress analysis in social media. [Dreaddit](https://arxiv.org/abs/1911.00133), [download](https://www.kaggle.com/datasets/monishakant/dataset-for-stress-analysis-in-social-media)

EmoBench: An emotion recognition benchmark dataset evaluating the emotional intelligence of large language models. [EmoBench](https://github.com/Sahandfer/EmoBench)

EmoryNLP: A dataset for emotion detection in multiparty dialogue, developed by Emory University's NLP research group. [EmoryNLP](https://github.com/emorynlp/emotion-detection)

IEMOCAP: The Interactive Emotional Dyadic Motion Capture Database for emotion and sentiment analysis.[IEMOCAP](https://paperswithcode.com/dataset/iemocap)

Manage these datasets on Hugging Face. Please upload them as private for copyright protection.

## Classification with GPT and Prompt Templates

To execute the `gen_v2/systematic_evaluation.py` file with the same parameters as `gen_v2/Efficient_auto_run.py`, follow these steps to initiate inference since there are multiple experiment runs:

Files like `gen_v2/Efficient_auto_run.py` can be executed quickly due to parallel processing.

For `Efficient_auto_run_seq.py`, use a list format to input model names and set parameters according to the JSON path in `prompt_template_v2`.

To add API keys for the models, create an `api_keys.json` file in the `gen_v2/` directory with the following dictionary format:

```python
{
  "Gemini": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "Sonnet": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "GPT4o": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```
The output files generated from these executions will be created in the same directory level as the `gen_v2` folder.

Efficient_auto_run_GPT.py: do classification by GPT API
Efficient_auto_run_Gemini.py: do classification by Gemini API
Efficient_auto_run_seq.py: do classification by Ollama (local model)
## Evaluation

Use the `gen_v2/eval/eval_classification.py` file to perform evaluations. In the main function, select the models you wish to evaluate by modifying the models list. 

Also, within the if __name__ == '__main__' block, adjust the base_folder_path to match the path of your results folder to proceed with the evaluation.

The evaluated results will be generated in the same location as the results folder.

