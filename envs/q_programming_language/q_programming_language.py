import verifiers as vf
from datasets import load_dataset
import subprocess
import tempfile
import os
import shutil
from pathlib import Path


def run_q_code(code: str, timeout: float = 1.0) -> tuple[bool, str, str]:
    """Execute Q code and return (success, stdout, stderr)."""
    # Get Q executable path from environment variable
    q_path = os.getenv("Q_EXECUTABLE_PATH")
    
    # Ensure Q code ends with exit 0;
    if not code.strip().endswith("exit 0;"):
        code = code.strip() + "\nexit 0;"
    
    temp_dir = Path(f"./temp_q_{os.getpid()}_{os.getppid()}")
    temp_dir.mkdir(exist_ok=True)
    temp_script = temp_dir / "script.q"

    try:
        with open(temp_script, 'w') as f:
            f.write(code)

        process = subprocess.run(
            f"{q_path} {temp_script.name}".split(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(temp_dir)
        )
        
        success = process.returncode == 0 and "error" not in process.stderr.lower()
        return success, process.stdout.strip(), process.stderr.strip()
    
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", str(e)
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def extract_q_code(completion: str) -> str:
    """Extract Q code from completion, handling reasoning format and code blocks."""
    # Look for <answer> tags first
    if "<answer>" in completion and "</answer>" in completion:
        start = completion.find("<answer>") + 8
        end = completion.find("</answer>")
        code = completion[start:end].strip()
    else:
        # If no tags, use the whole completion
        code = completion.strip()
    
    # Look for ```q code blocks
    if "```q" in code and "```" in code:
        start = code.find("```q") + 4
        end = code.find("```", start)
        if end != -1:
            return code[start:end].strip()
    
    # Look for generic ``` code blocks
    if "```" in code:
        start = code.find("```") + 3
        end = code.find("```", start)
        if end != -1:
            return code[start:end].strip()
    
    # Return the code as is
    return code


def load_environment(
    use_think: bool = True,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    perfect_bonus: float = 1.0,
):
    """Load Q programming environment with test case evaluation."""
    
    # Load dataset from Hugging Face
    raw_dataset = load_dataset("morganstanley/sft-python-q-problems", split="train")
    if num_train_examples != -1:
        raw_dataset = raw_dataset.select(range(num_train_examples))
    
    raw_eval_dataset = load_dataset("morganstanley/sft-python-q-problems", split="test")
    if num_eval_examples != -1:
        raw_eval_dataset = raw_eval_dataset.select(range(num_eval_examples))

    # Convert to verifiers format: prompt -> problem description, answer -> q_solution
    def convert_dataset(raw_data):
        converted = []
        for item in raw_data:
            converted.append({
                "prompt": [{"role": "user", "content": item["problem_description"]}],
                "answer": item["q_solution"],
                "test_cases": item["test_cases"],  # Keep for reward function
                "problem_id": item["problem_id"]
            })
        return converted

    # Convert to Dataset objects (verifiers expects HuggingFace Dataset)
    from datasets import Dataset
    dataset = Dataset.from_list(convert_dataset(raw_dataset))
    eval_dataset = Dataset.from_list(convert_dataset(raw_eval_dataset))

    # Create parser that extracts Q code
    if use_think:
        parser = vf.ThinkParser(extract_fn=extract_q_code)
    else:
        parser = vf.Parser(extract_fn=extract_q_code)

    # Create a mapping from prompt to test cases for the reward function
    prompt_to_tests = {}
    for item in dataset:
        # Use the prompt content as the key
        prompt_content = item['prompt'][0]['content'] if isinstance(item['prompt'], list) else item['prompt']
        prompt_to_tests[prompt_content] = item['test_cases']
    
    for item in eval_dataset:
        prompt_content = item['prompt'][0]['content'] if isinstance(item['prompt'], list) else item['prompt']
        prompt_to_tests[prompt_content] = item['test_cases']

    def q_reward_func(parser, completion, answer, **kwargs):
        """Reward function based on test case performance."""
        q_code = parser.parse_answer(completion) or ""
        
        if not q_code.strip():
            return -0.2
        
        # Get the prompt from kwargs to find the test cases
        prompt = kwargs.get('prompt', '')
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt_content = prompt[0].get('content', '')
        else:
            prompt_content = str(prompt)
        
        # Get test cases from our mapping
        test_cases = prompt_to_tests.get(prompt_content, [])
        if not test_cases:
            # Fallback: if no test cases, give basic reward for having code
            return 0.5 if q_code.strip() else 0.0
        
        # Run Q code against test cases
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            test_code = test_case.get('q_test_code', '')
            expected_output = test_case.get('q_expected_output', '')
            
            if not test_code:
                continue
            
            # Remove exit 0; from solution if present
            clean_q_code = q_code.strip()
            if clean_q_code.endswith("exit 0;"):
                clean_q_code = clean_q_code[:-7].strip()
            
            # Combine solution with test
            full_code = f"{clean_q_code}\n\n{test_code}"
            
            # Execute with timeout
            success, stdout, stderr = run_q_code(full_code, timeout=1.0)
            
            # Check if test passed
            if success and stdout is not None:
                actual_stripped = stdout.strip()
                expected_stripped = expected_output.strip()
                
                # Handle empty string cases
                if ((actual_stripped == "" and expected_stripped == '""') or 
                    (actual_stripped == '""' and expected_stripped == "") or
                    (actual_stripped == expected_stripped)):
                    passed_tests += 1
        
        # Calculate reward: percentage passed + bonus for perfect
        if total_tests > 0:
            base_reward = passed_tests / total_tests
            perfect_reward = perfect_bonus if passed_tests == total_tests else 0.0
            return base_reward + perfect_reward
        else:
            return 0.0

    rubric = vf.Rubric(
        funcs=[q_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt="You are an expert Q language programmer. Write correct Q code that solves the given programming problem.",
        parser=parser,
        rubric=rubric,
    )
    return vf_env