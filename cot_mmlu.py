import sys
import datasets
import os
import re
from openai import OpenAI
import json
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_response_with_system(system_prompt, prompt):
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=prompt,
        temperature=0.0
    )
    return response.output_text

def get_response(prompt):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.0
    )
    return response.output_text


def extract_final_answer(text):
    """대화에서 최종 답변을 추출합니다."""
    # MATH 형식: 250
    # MMLU-Pro/ExploreToM 형식: 250 minutes
    prompt=f"""You are an assistant that is helping an user to identify the intention of certain responses in a conversation. 
You should extract the exact option number that the response is submiting as the final answer, or say "not sure yet" if it seems like there is no explict answer included in the response. On [Extraction], you should only return the option number, not any other text.
[Response]
{text}
[Extraction]"""
    response = get_response(prompt)

    return response

dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
categories=["chemistry", "law", "engineering", "economics", "health", "psychology", "business", "biology", "computer_science", "philosophy"]

for target_category in categories:
    result_arr=[]
    for sample in dataset["test"]:
        category=sample["category"]
        if category!=target_category:
            continue
        problem=sample["question"]
        options=sample["options"]
        answer_index=sample["answer_index"]
        
        options_str = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        problem_str = f"{problem} Think step by step.\n\n{options_str}"
        response=get_response(problem_str)
        chosen_option=extract_final_answer(response)
        try:
            is_correct=int(chosen_option) == int(answer_index)+1
        except:
            is_correct="-"

        results = {
            "question": problem,
            "options": options,
            "response": response,
            "chosen_option": chosen_option,
            "is_correct": is_correct,
        }
        result_arr.append(results)
        with open(f"./result_mmlu/cot_results_{target_category}.json", "w", encoding="utf-8") as f:
            json.dump(result_arr, f, ensure_ascii=False, indent=2)
        if len(result_arr) > 10:
            break
    
    with open(f"./result_mmlu/cot_results_{target_category}.json", "w", encoding="utf-8") as f:
        json.dump(result_arr, f, ensure_ascii=False, indent=2)
        