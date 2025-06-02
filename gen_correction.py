import openai
import re

openai.api_key = "你的key"
openai.api_base = "你的base"  # 如果有自定义代理，填入，否则可注释

def gen_corrective_action(prompt, environment_prompt, model="gpt-4o", generation_action=None):
    """
    输入原始prompt、环境描述、openai模型名和当前action，返回corrective action。
    """
    if generation_action is None:
        # 先生成一个action
        generation_action = Gen_Action(prompt, model)
    
    # 生成反馈
    feedback_prompt = "You are a world simulator. You need to generate the feedback according to plans. " + prompt[100:] + generation_action + "\nFeedback: "
    feedback_model = Gen_Action(feedback_prompt, model)
    
    # 如果有Error，生成corrective action
    if 'Error' in feedback_model:
        corrective_prompt = (
            "You are a replanner. Given the original plan, you are expected to generate the new plan according to the feedback. "
            + environment_prompt + prompt[100:] + generation_action + "\n A corrective step would be"
        )
        corrective_action = Gen_Action(corrective_prompt, model)
        # 再次用反馈模型检查corrective action
        feedback_check = Gen_Action(
            "You are a world simulator. You need to generate the feedback according to plans. "
            + prompt[100:] + corrective_action + "\nFeedback: ", model)
        if 'Error' not in feedback_check:
            return corrective_action
        else:
            return None
    else:
        # 没有Error，直接返回原action
        return generation_action

# 用openai接口生成action

def Gen_Action(input, model="gpt-4o"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": input}]
    )
    gen_action = completion.choices[0].message.content
    if '\n' in gen_action:
        gen_action = gen_action.split('\n')[0]
    return gen_action

# 使用示例：
# action = gen_corrective_action(prompt, environment_prompt, model="gpt-4o") 