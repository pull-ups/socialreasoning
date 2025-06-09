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

def get_response_llama_with_system(system_prompt, prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        eos_token_id=terminators,
        temperature=0,
    )
    return outputs[0]["generated_text"][-1]




def get_response_llama(prompt):
    messages = [     
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        eos_token_id=terminators,
        temperature=0,
    )
    return outputs[0]["generated_text"][-1]

    



class CoralAgent:
    """
    Coral 방법론을 따르는 LLM 에이전트를 나타냅니다.
    """
    def __init__(self, agent_name, llm_model, task_name, task_specific_instruction):
        self.agent_name = agent_name
        self.llm_model = llm_model # 사용할 LLM 모델
        self.system_prompt = self._create_system_prompt(task_name, task_specific_instruction)

    def _create_system_prompt(self, task_name, task_specific_instruction):
        general_instruction = f"""
You are working with an advanced user to solve some complex {task_name} problems.

Here is how you should proceed:
- Starting on the problem, first lay out a plan and ask for confirmation on the plan;
- When the user proposes a plan, an actual solution, or a partial solution, look carefully at each of the step, and ask clarification questions if you are unsure about the correctness of a certain step;
- When you notice an error, be precise and direct, over-politenss will not help anyone;
- When the user asks you questions about your solution, try to unravel certain steps and explain how they work, correct your mistake if you think you've made one, but stand your ground if you think it's actually correct;
- Always stay on topic and work towards a solution to the original problem;
- {task_specific_instruction}"""
        return general_instruction

    def get_llm_response(self, prompt, agent_name):
        """
        가상의 LLM API 호출 함수입니다.
        실제 사용 시에는 이 부분을 사용하는 LLM의 API로 교체해야 합니다.
        """
        prompt+= f"\n{agent_name}:"
        print(f"\033[91m--- Prompt to {self.agent_name} ({self.llm_model}) ---\033[0m")
        print(f"\033[91m{prompt}\033[0m")
        
        
        response = get_response_with_system(self.system_prompt, prompt)
        print(f"\033[94m--- Response ---\033[0m")
        print(f"\033[94m{response}\033[0m")
        print("-" * 20)
        return response

def extract_final_answer(text):
    """대화에서 최종 답변을 추출합니다."""
    # MATH 형식: 250
    # MMLU-Pro/ExploreToM 형식: 250 minutes
    prompt=f"""You are an assistant that is helping an user to identify the final answer of a math problem. 
You should extract the exact the answer(number) of the math problem, or say "not sure yet" if it seems like there is no explict answer included in the response. On [Extraction], you should only return the answer, not any other text.
[Response]
{text}
[Extraction]"""
    response = get_response(prompt)

    return response

def run_coral_collaboration(problem, task_name, task_specific_instruction, llm_model_a, llm_model_b):
    """
    Coral 협력 대화를 실행하고 최종 합의된 답변을 반환합니다.
    """
    agent_a = CoralAgent("Agent_A", llm_model_a, task_name, task_specific_instruction)
    agent_b = CoralAgent("Agent_B", llm_model_b, task_name, task_specific_instruction)

    # 대화 시작 (Rule from Section 2.1) 
    conversation_history = f"Agent A:I'm trying to solve this {task_name} problem: \"{problem}\""
    conversation_log = []

    max_turns = 5 # 최대 대화 턴 수 제한
    answer_a=""
    answer_b=""
    for turn in range(max_turns):
        # Agent B의 턴
        response_b = agent_b.get_llm_response(conversation_history, "Agent B")
        conversation_history += f"\n\nAgent B: {response_b}"
        answer_b = extract_final_answer(response_b)
        conversation_log.append({"agent": "Agent B", "turn": turn + 1, "response": response_b, "extracted_answer": answer_b})
        print("EXTRACTED answer_b: ", answer_b)


        if answer_a and answer_b and answer_a == answer_b and "not" not in answer_a and "not" not in answer_b:
            print("Agreement reached!")
            return answer_a, conversation_log

        # Agent A의 턴
        response_a = agent_a.get_llm_response(conversation_history, "Agent A")
        conversation_history += f"\n\nAgent A: {response_a}"
        answer_a = extract_final_answer(response_a)
        conversation_log.append({"agent": "Agent A", "turn": turn + 1, "response": response_a, "extracted_answer": answer_a})
        print("EXTRACTED answer_a: ", answer_a)

        # 종료 조건: 두 에이전트의 답변이 일치하여 합의에 도달했는지 확인 (Rule from Section 2.2) 
        if answer_a and answer_b and answer_a == answer_b and "not" not in answer_a and "not" not in answer_b:
            print("Agreement reached!")
            return answer_a, conversation_log

    print("Agreement not reached within max turns.")
    return None, conversation_log

def solve(problem, category):
    final_answer, conversation_log = run_coral_collaboration(
        problem=problem,
        task_name=category,
        task_specific_instruction='To give a final answer to the question (e.g., "\\sqrt{3}"), put your answer in an LaTex box like $\\boxed{\\sqrt{3}}$', # MATH 문제용 지침
        # task_specific_instruction='To give a final answer, do it in the format of "The correct answer is (insert answer here)", such as "The correct answer is (B)"',
        llm_model_a="virtual_llm_a", # 사용할 LLM 모델 지정
        llm_model_b="virtual_llm_b"
    )


    if final_answer:
        print(f"\nFinal agreed answer: {final_answer}")
        return (final_answer, True, conversation_log)
    else:
        print("\nCould not get a final answer.")
        return (None, False, conversation_log)


if __name__ == '__main__':
    # run_mode=sys.argv[1]
    # if run_mode=="llama":
    #     model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #     pipeline = transformers.pipeline(
    #         "text-generation",
    #         model=model_id,
    #         model_kwargs={"torch_dtype": torch.bfloat16},
    #         device_map="auto",
    #     )
    #         terminators = [
    #         pipeline.tokenizer.eos_token_id,
    #         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]


    dataset = datasets.load_dataset("openai/gsm8k", "main")
    
    result_arr=[]
    for sample in dataset["train"]:
        problem=sample["question"]
        answer_index=sample["answer"].split("\n####")[1]
        category="math"
        problem_str = f"I'm trying to solve this problem: {problem}"

        final_answer, is_agreed, conversation_log = solve(problem_str, category)
        try:
            is_correct=int(final_answer) == int(answer_index)
        except:
            is_correct="-"


        results = {
            "final_answer": final_answer,
            "is_agreed": is_agreed,
            "gt_answer": answer_index,
            "is_correct": is_correct,
            "conversation_log": conversation_log
        }
        result_arr.append(results)

        print("Final answer: ", final_answer)
        print("Is agreed: ", is_agreed)
        print("GT answer: ", answer_index)
        print("Conversation log: ", conversation_log)
        

        with open("results_math_50.json", "w", encoding="utf-8") as f:
            json.dump(result_arr, f, ensure_ascii=False, indent=2)
        if len(result_arr) > 50:
            break
    
    
    with open("results_math_50.json", "w", encoding="utf-8") as f:
        json.dump(result_arr, f, ensure_ascii=False, indent=2)
        