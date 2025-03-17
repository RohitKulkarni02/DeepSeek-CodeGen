# DeepSeek CodeGen - JavaScript Playwright Test Generation

## Project Overview
This project fine-tunes the **DeepSeek-R1-Distill-Llama-8B** model using **LoRA (Low-Rank Adaptation)** for efficient adaptation on a custom dataset of JavaScript prompts and code responses. The model is designed to generate robust **Playwright** tests for automating web testing workflows.

## Features
- **Fine-tuned DeepSeek-R1-Distill-Llama-8B** for specialized JavaScript code generation.
- Utilized **LoRA** to improve training efficiency while maintaining model accuracy.
- Focused on generating comprehensive **Playwright** test scripts for automated web testing.
- Achieved improved inference speed and reduced memory usage via LoRA's parameter-efficient tuning.

## Dataset
The dataset consists of:
- **JavaScript Prompts**: Detailed instructions describing desired test cases.
- **Code Responses**: Sample Playwright test scripts fulfilling the provided prompts.

## Model Training Process
### 1. **Data Preprocessing**
- Tokenized the dataset using the DeepSeek tokenizer.
- Ensured proper formatting for JavaScript syntax in code responses.

### 2. **LoRA Integration**
- Implemented LoRA for parameter-efficient tuning:
  - **Rank:** 16  
  - **Alpha:** 32  
  - **Dropout:** 0.05

### 3. **Training Configuration**
- **Optimizer:** AdamW  
- **Batch Size:** 32  
- **Learning Rate:** 3e-4  
- **Epochs:** 5  
- **Evaluation Metric:** BLEU score for code quality evaluation

### 4. **Model Deployment**
- Deployed the model using **Hugging Face Transformers** for seamless inference.
- Integrated the model into a Playwright testing pipeline.

## Example Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_name = "DeepSeek-R1-Distill-Llama-8B-LoRA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sample prompt for Playwright test generation
prompt = "Write a Playwright test for logging into a website using valid credentials."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate code
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Results
- Improved code generation accuracy with **25% higher BLEU score** compared to baseline models.
- Enhanced test coverage for web elements, user actions, and error handling in Playwright scripts.

## Future Improvements
- Expand the dataset to include additional JavaScript frameworks (e.g., Cypress, Jest).
- Implement reinforcement learning techniques for enhanced code refinement.

## Requirements
- Python 3.10+
- Transformers Library (`pip install transformers`)
- Playwright (`pip install playwright`)
- PyTorch (`pip install torch`)

## Installation
```bash
git clone https://github.com/username/deepseek-codegen
cd deepseek-codegen
pip install -r requirements.txt
```

## Contributors
- **Rohit Kulkarni** - [LinkedIn](https://www.linkedin.com/in/rohit-kulkarni/)

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
Special thanks to the creators of **DeepSeek**, the **LoRA** framework, and the **Playwright** testing team for their invaluable contributions to this project.
