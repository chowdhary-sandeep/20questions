import verifiers as vf

from transformers import pipeline

from verifiers.utils.data_utils import (
    load_example_dataset,
)

SYSTEM_PROMPT = """Thou art a scholar born of the seventeenth age. Bethink thee by degrees within the <ponder>...</ponder> tokens; thereafter, render thy final answer within <andswaru>...</andswaru>."""


def load_environment(
    num_train_examples=-1,
    num_eval_examples=-1,
):
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))
    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    parser = vf.XMLParser(fields=["ponder", "andswaru"], answer_field="andswaru")
    classifier = pipeline("text-classification", model="notaphoenix/shakespeare_classifier_model", top_k=None)

    def correct_answer_reward_func(parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0
    
    def old_speak_reward_func(parser, completion, **kwargs):
        score = classifier(parser.get_assistant_messages(completion)[-1]['content'])
        return next(
                item['score'] 
                for item in score[0] 
                if item['label'] == 'shakespearean'
            )

    rubric = vf.Rubric(
        parser=parser,
        funcs=[correct_answer_reward_func, parser.get_format_reward_func(), old_speak_reward_func],
        weights=[1.0, 0.5, 1.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
