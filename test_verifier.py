import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer

import verifier as capa


CORRECT_ANSWER = "1"

parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
rubric = capa.CapaRubric(parser=parser, feature_mode="id")


def check_response(response):
    return rubric.correct_features_reward_func(
        parser, [{"content": response, "role": "assistant"}], CORRECT_ANSWER
    )


print(f"Score for correct answer: {check_response("<think></think>\\boxed{1}")}")
print(f"Score for incorrect answer: {check_response("<think></think>\\boxed{2}")}")
print(f"Score for no answer: {check_response("<think></think>idk lol")}")
print(f"Score for partially correct answer (1/2): {check_response("<think></think>\\boxed{1,2}")}")
print(f"Score for partially correct answer (1/3): {check_response("<think></think>\\boxed{1,2,3}")}")
