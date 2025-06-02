import re
import json
from tqdm import tqdm

def rewrite_failR(example):
    '''
    输入需要这个数据还有SuccessOrFail, FailReasons, GenAction
    输出时新的fail_reason
    '''
    
    def translate_program_nlp(sentence):

        def remove_parentheses(sentence):
            pattern = r'\([^)]*\)'
            return re.sub(pattern, '', sentence)

        def remove_angle_brackets(sentence):
            pattern = r'<|>'
            return re.sub(pattern, '', sentence)

        def remove_extra_spaces(sentence):
            pattern = r'\s+'
            return re.sub(pattern, ' ', sentence)

        def replace_word(sentence, word, replacement):
            return sentence.replace(word, replacement)

        idx = sentence.find('when executing')
        sentence = sentence[:idx]
        sentence = remove_parentheses(sentence)
        sentence = remove_angle_brackets(sentence)
        sentence = remove_extra_spaces(sentence)
        sentence = replace_word(sentence, "character", "Robot agent")

        return sentence

    example_SorF = example['SuccessOrFail']
    example_failR = example['FailReasons']
    example_action = example['GenAction']

    vh_error = ''
    for i in range(len(example_failR)):
        if 'when executing' in example_failR[len(example_failR)-i-1]:
            vh_error = example_failR[len(example_failR)-i-1]
            break

    vh_error_list = vh_error.split(',')
    index= 0

    example_new_failR = []
    for j in range(len(example_failR)):
        if example_SorF[j] == True:
            example_new_failR.append('')
        elif 'when executing' in example_failR[j]:
            example_new_failR.append(translate_program_nlp(vh_error_list[index]))
            index = index + 1
        else:
            example_new_failR.append(example_failR[j])

    return example_new_failR

def gen_feedback_trainingData(evaluation_path, model_training_data_path):
    
    evaluation_data = []
    with open(evaluation_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            evaluation_data.append(json_object)

    training_data = []

    for  exp_data in tqdm(evaluation_data):

        task_name = exp_data['task']
        task_desc = exp_data['description']
        task_gen_action = exp_data['GenAction']
        task_ObserANDState = exp_data['ObserANDState']
        task_SuccessOrFail = exp_data['SuccessOrFail']
        task_path = exp_data['path']

        task_FailReasons = rewrite_failR(exp_data)
        prompt = "You are a world simulator. You need to generate the feedback according to plans. "
        eachData_prompt = prompt + '\n\nTask: ' + task_name + '\nDescription: ' + task_desc + '\n'

        for i in range(len(task_gen_action)):
            
            one_data = dict()

            iaction = task_gen_action[:i]
            iobservation = task_ObserANDState[:i]

            result = ""
            for j in range(len(iaction)):
                step_text = f"step{j+1}: {iaction[j]}"
                observation_text = f"observation{j+1}: {iobservation[j]}"
                result += step_text + "\n" + observation_text + "\n"
            
            if len(iaction) == 0:
                result = result + f"step{1}: {task_gen_action[i]}" + "\n" + "Feedback: "
            else:
                next_step_text = f"step{j+2}: {task_gen_action[i]}"
                result = result + next_step_text + "\n" + "Feedback: "

            if i ==  len(task_gen_action)-1:
                one_data['output'] = "Task finished"
            else:
                if task_SuccessOrFail[i] == True:
                    one_data['output'] = 'True'
                else:
                    one_data['output'] = "Error, " + task_FailReasons[i]

            one_data['input'] = eachData_prompt + result
            one_data['path'] = task_path
            one_data['task'] = task_name
            one_data['description'] = task_desc
            
            # 进行长度过长剪切
            if len(one_data['input'].split(' '))<1024:
                training_data.append(one_data)

            # print(one_data['input'])
            # print(one_data['output'])

    # 进行保存数据
    with open(model_training_data_path, 'w') as f:
        for item in training_data:
            json.dump(item, f)
            f.write('\n')  

gen_feedback_trainingData('', '')