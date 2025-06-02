# 将生成数据转化为可执行数据
from enum import Enum
from collections import OrderedDict
import parse
import json

def jsonl_add_data(file_path, add_data):
    '''file_path是一个jsonl文件 add_data是一个列表'''
    ori_data = []
    with open(file_path, 'r') as file:
        for line in file:
            ori_data.append(json.loads(line))

    ori_data.extend(add_data)

    with open(file_path, 'w') as file:
        for item in ori_data:
            file.write(json.dumps(item) + '\n')

class EvolveGraphAction(Enum):
    """
    All supported actions, value of each enum is a pair (humanized name, required_number of parameters)
    """
    CLOSE = ("Close", 1, 'close {}')
    DRINK = ("Drink", 1, 'drink {}')
    FIND = ("Find", 1, 'find {}')
    WALK = ("Walk", 1, 'walk to {}')
    GRAB = ("Grab", 1, 'grab {}')
    LOOKAT = ("Look at", 1, 'look at {}')
    # LOOKAT_SHORT = ("Look at short", 1, 'look at {}')
    # LOOKAT_MEDIUM = LOOKAT
    # LOOKAT_LONG = ("Look at long", 1, 'look at {}')
    OPEN = ("Open", 1, 'open {}')
    POINTAT = ("Point at", 1, 'point at {}')
    PUTBACK = ("Put", 2, 'put {} on {}')
    #PUT = ("Put", 2, '')
    #PUTBACK = PU, ''T
    PUTIN = ("Put in", 2, 'put {} in {}')
    PUTOBJBACK = ("Put back", 1, 'put back {}')
    RUN = ("Run", 1, 'run to {}')
    SIT = ("Sit", 1, 'sit on {}')
    STANDUP = ("Stand up", 0, 'stand up')
    SWITCHOFF = ("Switch off", 1, 'switch off {}')
    SWITCHON = ("Switch on", 1, 'switch on {}')
    TOUCH = ("Touch", 1, 'touch {}')
    TURNTO = ("Turn to", 1, 'turn to {}')
    WATCH = ("Watch", 1, 'watch {}')
    WIPE = ("Wipe", 1, 'wipe {}')
    PUTON = ("PutOn", 1, 'put on {}')
    PUTOFF = ("PutOff", 1, 'take off {}')
    GREET = ("Greet", 1, 'greet {}')
    DROP = ("Drop", 1, 'drop {}')
    READ = ("Read", 1, 'read {}')
    LIE = ("Lie", 1, 'lie on {}')
    POUR = ("Pour", 2, 'pour {} into {}')
    TYPE = ("Type", 1, 'type on {}')
    PUSH = ("Push", 1, 'push {}')
    PULL = ("Pull", 1, 'pull {}')
    MOVE = ("Move", 1, 'move {}')
    WASH = ("Wash", 1, 'wash {}')
    RINSE = ("Rinse", 1, 'rinse {}')
    SCRUB = ("Scrub", 1, 'scrub {}')
    SQUEEZE = ("Squeeze", 1, 'squeeze {}')
    PLUGIN = ("PlugIn", 1, 'plug in {}')
    PLUGOUT = ("PlugOut", 1, 'plug out {}')
    CUT = ("Cut", 1, 'cut {}')
    EAT = ("Eat", 1, 'eat {}') 
    SLEEP = ("Sleep", 0, 'sleep')
    WAKEUP = ("WakeUp", 0, 'wake up')
    RELEASE = ("Release", 1, 'release')

def merge_add(d, k, v):
    if k == v:
        return
    # print(f'adding {k} --> {v}')
    if k in d:
        prev_v = d[k]
        # print(f'existing: {k} --> {prev_v}')
        merge_add(d, v, prev_v)
    else:
        d[k] = v

with open("virtualhome_master/class_name_equivalence.json", 'r') as f:
    abstract2detail = json.load(f)

detail2abstract = dict()
for abstract, details in abstract2detail.items():
    for detail in details:
        merge_add(detail2abstract, detail, abstract)

def process_format(arg):
    # don't use any underscore in args
    arg = arg.replace(' ', '_')
    return arg

def str2program_list(program_lines):
    '''
    这个代码的功能是将allowed action里面的动作给转化为virtualhome中的标准格式
    比如：
    input: ["close address_book","close address_book","close address_book"]
    output: ["[CLOSE] <address_book> (1)"]
    '''
    def _format_arg(arg):
        arg = arg.lower().strip().replace(' ', '_')
        if arg in detail2abstract:
            return detail2abstract[arg]
        return arg

    # start parsing ==============================
    # pl = program_str[program_str.index('Step 1:'):].split('\n')
    info = dict()
    info['parsing_error'] = []
    pl = program_lines
    parsed_lines = []
    success_count = 0
    for i, line in enumerate(pl):
        line = line.lower().strip()
        if len(line) == 0:
            continue
        if ':' in line:
            line = line[line.index(':') + 1:].strip()
        try:
            # try matching each possible action
            possible_parsed = OrderedDict()
            for action in EvolveGraphAction:
                action_template = action.value[2]
                expected_num_args = action.value[1]
                parsed = parse.parse(action_template, line)
                if parsed is not None:
                    assert action.name not in possible_parsed
                    if len(parsed.fixed) == expected_num_args:
                        # print(action_template, parsed, expected_num_args)
                        possible_parsed[action.name] = parsed
                    else:
                        # skip if number of parsed args does not match expected
                        pass
            assert len(possible_parsed) == 1, f'possible_parsed: {possible_parsed} does not equal to 1'
            parsed_action = list(possible_parsed.keys())[0]
            parsed_args = possible_parsed[parsed_action]
            if len(parsed_args.fixed) == 0:
                pl_str = '[{}]'
                pl_str = pl_str.format(parsed_action)
            elif len(parsed_args.fixed) == 1:
                pl_str = '[{}] <{}> (1)'
                # pl_str = pl_str.format(parsed_action, _format_arg(parsed_args[0]))                                  # 考虑了名字的问题
                pl_str = pl_str.format(parsed_action, process_format(parsed_args[0]))
            elif len(parsed_args.fixed) == 2:
                pl_str = '[{}] <{}> (1) <{}> (1)'
                # pl_str = pl_str.format(parsed_action, _format_arg(parsed_args[0]), _format_arg(parsed_args[1]))     # 考虑了名字的问题
                pl_str = pl_str.format(parsed_action, process_format(parsed_args[0]), process_format(parsed_args[1]))
            else:
                raise NotImplementedError
            parsed_lines.append(pl_str)
            success_count += 1
        except AssertionError as e:
            message = "| {} | {} | '{}'".format(e.__class__.__name__, e, line)
            info['parsing_error'].append(message)
            line = pl[i]
            if ':' in line:
                line = line[line.index(':') + 1:].strip()
            # none of these is likely going to work, but parse it this way to obey vh format
            if len(line) > 0:
                words = line.split(' ')
                if len(words) == 1:
                    pl_str = '[{}]'.format(words[0].upper())
                elif len(words) == 2:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), words[1])
                elif len(words) == 3:
                    pl_str = '[{}] <{}> (1) <{}> (1)'.format(words[0].upper(), words[1], words[2])
                else:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), '_'.join(words[1:]))
            else:
                pl_str = '[EMPTY]'
            parsed_lines.append(pl_str)
    info['num_parsed_lines'] = len(parsed_lines)
    info['num_total_lines'] = len(pl)
    if len(pl) != 0:
        info['parsibility'] = success_count / len(pl)
    else:
        info['parsibility'] = 0
    return parsed_lines, info

print(str2program_list(['Turn to face the DVD player','Walk to home office', 'Walk to shoes', 'Find shoes', 'Grab shoes', 'Put on shoes', 'pour out lotion into right hand', 'Find the computer desk']))

'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


# 单条数据进行验证
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
import json

path = '/data/hanlin/different_size/t5_small_epoch/epoch_0'
# tokenizer = AutoTokenizer.from_pretrained(path)
# model = AutoModelForCausalLM.from_pretrained(path, device_map='cuda:0', trust_remote_code=True)
tokenizer = T5Tokenizer.from_pretrained(path)
model = T5ForConditionalGeneration.from_pretrained(path, device_map='cuda:6', trust_remote_code=True)

device = 'cuda:6' if cuda.is_available() else 'cpu'

# 采用t5的方式进行推理
def Gen_Action(input, tokenizer, model):

    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    output_action = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_action


'''----------------------------------------------------------------结束-----------------------------------------------------------------------------'''

'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
import sys
from tqdm import tqdm
import re
import random

from virtualhome_master.virtualhome.simulation.evolving_graph import utils
from virtualhome_master.virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome_master.virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome_master.virtualhome.simulation.evolving_graph.environment import EnvironmentGraph, EnvironmentState


def remove_duplicate_edge(input_dict):
    Edges = input_dict['edges']
    for edge in Edges:
        fgledge = {'from_id':edge['to_id'], 'relation_type': 'INSIDE', 'to_id': edge['from_id']}
        if fgledge in Edges:
            if edge == fgledge:
                Edges.remove(edge)
            else:
                Edges.remove(fgledge)
                Edges.remove(edge)
    input_dict['edges'] = Edges
    return input_dict

def change_obj_index(graph, program, id, specific_objects, last_obj_id):

    graph_dict = graph.to_dict()
    agent_has_objid = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == id and "HOLD" in n["relation_type"]]

    obj_id_dict = {}
    obj_ids_close = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == id and  n["relation_type"]=="CLOSE"]  # 离agent close的物品id
    obj_ids_close_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == id and  n["relation_type"]=="CLOSE"]
    obj_ids_close.extend(obj_ids_close_two)
    obj_ids_close = list(set(obj_ids_close))
    # obj = [node['class_name'] for node in graph_dict['nodes'] if node["id"] in obj_ids_close]  # 离agent close的物品名称
    obj = []
    for i in range(len(obj_ids_close)):
        obj.append([node['class_name'] for node in graph_dict['nodes'] if node['id']==obj_ids_close[i]][0])

    print('agent close to:', obj_ids_close)

    if last_obj_id != -1:
        last_obj_ids_close = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == last_obj_id and  n["relation_type"]=="CLOSE"]  # 离agent close的物品id
        last_obj_ids_close_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == last_obj_id and  n["relation_type"]=="CLOSE"]
        last_obj_ids_close.extend(last_obj_ids_close_two)
        last_obj_ids_close = list(set(last_obj_ids_close))
        # last_obj = [node['class_name'] for node in graph_dict['nodes'] if node["id"] in last_obj_ids_close]  # 离agent close的物品名称

        # 再加一个限制，不仅是在上一个物体的附近，也可能是在上一个物体的里面也是可以的
        last_obj_ids_inside = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == last_obj_id and  n["relation_type"]=="INSIDE"]  # 离agent close的物品id
        last_obj_ids_inside_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == last_obj_id and  n["relation_type"]=="INSIDE"]
        last_obj_ids_inside.extend(last_obj_ids_inside_two)
        last_obj_ids_inside = list(set(last_obj_ids_inside))   

        last_obj_ids_close.extend(last_obj_ids_inside)     

        last_obj = []
        for i in range(len(last_obj_ids_close)):
            last_obj.append([node['class_name'] for node in graph_dict['nodes'] if node['id']==last_obj_ids_close[i]][0])

        print('last obj id close:', last_obj_ids_close)
        print('last obj:', last_obj)

    else:
        last_obj_ids_close = []
        last_obj = []

    # 第一种格式 [ ]
    if program.count('<') == 0:
        return program, specific_objects, last_obj_id
    
    # 第二种格式 [ ] < > ( )
    if program.count('<') == 1:
        
        def extract_text(input_string):
            pattern = r'\[([^]]+)\]|\<([^>]+)\>|\(([^)]+)\)'  # 正则表达式模式，匹配方括号、尖括号和圆括号中的内容
            matches = re.findall(pattern, input_string)  # 查找所有匹配的内容
            extracted_text = [match[0] or match[1] or match[2] for match in matches]  # 提取匹配结果
            return extracted_text
        
        extracted_text = extract_text(program)

        for i in range(len(obj_ids_close)):
            if obj[i] == extracted_text[1]:
                obj_id_dict[obj[i]] = obj_ids_close[i]

        for i in range(len(last_obj_ids_close)):
            if last_obj[i] == extracted_text[1]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]

        if extracted_text[0] not in ['FIND', 'WALK']:
            obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == extracted_text[1]]  # 环境中所有的名称相同的node

            # print('extracted text:', extracted_text[1])
            # print('obj_ids:', obj_id1)
            if extracted_text[1] in list(specific_objects.keys()):
                id1 = specific_objects[extracted_text[1]]
                print('specific objs')
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
                print('close objects')
            elif len(obj_id1) == 0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
                print('random objects')
            pattern = r'\d+'  # 正则表达式模式，匹配数字
            replaced_string = re.sub(pattern, str(id1), program)     
            return replaced_string, specific_objects, id1   
        else:
            obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == extracted_text[1]]
            if len(obj_id1)==0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id
            
            # print('extracted text:', extracted_text[1])
            # print('obj_ids:', obj_id1)

            if extracted_text[1] in list(specific_objects.keys()):
                id1 = specific_objects[extracted_text[1]]
                print('specific objs')
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
                print('close objs')
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
                print('random objs')
            
            pattern = r'\d+'  # 正则表达式模式，匹配数字
            replaced_string = re.sub(pattern, str(id1), program)
            return replaced_string, specific_objects, id1  
            
    # 第三种格式 [ ] < > ( ) < > ( )
    if program.count('<') == 2:
        
        ori_specific_objects = specific_objects

        def parse_content(input_string):
            pattern = r'\[(.*?)\]|\<(.*?)\>|\((.*?)\)'  # 正则表达式模式，匹配方括号、尖括号和圆括号中的内容
            matches = re.findall(pattern, input_string)  # 查找所有匹配的内容
            parsed_content = [group for match in matches for group in match if group]  # 解析匹配结果
            return parsed_content
        
        content = parse_content(program)
        obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == content[1] and node['id'] in agent_has_objid]
        obj_id2 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == content[3]]
        
        for i in range(len(obj_ids_close)):
            if obj[i] == content[1]:
                obj_id_dict[obj[i]] = obj_ids_close[i]
            if obj[i] == content[3]:
                obj_id_dict[obj[i]] = obj_ids_close[i]

        for i in range(len(last_obj_ids_close)):
            if last_obj[i] == content[1]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]
            if last_obj[i] == content[3]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]

        if len(obj_id1) == 0:
            return content[1] + " not in hand. Robot agent should hold " + content[1] + " firstly.", specific_objects, last_obj_id

        id1 = random.choice(obj_id1)
        specific_objects[content[1]] = id1

        if len(obj_id2) == 0:
            return content[3] + " isn't available in the environment.", specific_objects, last_obj_id
        elif content[3] in list(specific_objects.keys()):
            id2 = specific_objects[content[3]]
        elif content[3] in list(obj_id_dict.keys()):
            id2 = obj_id_dict[content[3]]
            specific_objects[content[3]] = id2
        else:
            id2 = random.choice(obj_id2)
            specific_objects[content[3]] = id2

        # 防止两个物体时相同的，导致循环
        if id1 == id2:
            return content[1] + " can't be put or pour into itself.", ori_specific_objects, last_obj_id

        program_list = list(program)
        positions = [index for index, element in enumerate(program_list) if element == ')']
        qian_program = program[:positions[0]+1]
        hou_program = program[positions[0]+1:]
        qian_program = re.sub(r'\((\d+)\)', '('+str(id1)+')', qian_program, count=1)
        hou_program = re.sub(r'\((\d+)\)', '('+str(id2)+')', hou_program, count=1)
        program = qian_program + hou_program

        return program, specific_objects, id2

def check_action_format(program_text):
    action = re.findall(r'\[(.*?)\]', program_text)[0]
    num_para = EvolveGraphAction[action].value[1]
    action_para = program_text.count('<')
    if num_para == action_para:
        return program_text
    else:
        return action + " needs " + str(num_para) + " parameters. But there are " + str(action_para) + " parameters."

data = []
# 逐行读取 JSONL 文件
with open('data/test.jsonl', "r") as file:
    for line in file:
        # 解析 JSON 对象
        json_object = json.loads(line)
        
        # 在此处进行处理
        data.append(json_object)

print('number of data:', len(data))

# path = "/data/hlwang/my_exp_DATA/all_data_evaluation_my_ckt.jsonl"

def evaluation(input_data, output_path):
    num_cor = 0
    print('number of correction:', num_cor)
    New_Data = []
    for eachline in tqdm(input_data):
        print('--------------------------------------------New task------------------------------------------')
        print('----------------------------------------------------------------------------------------------')

        scene_path = "virtualhome_master/init_and_final_graphs/" + eachline['path'][151:-4] + ".json"
        with open(scene_path) as f:
            Tdata = json.load(f)

        Tdata = Tdata['init_graph']
        Tdata = remove_duplicate_edge(Tdata)

        # 为了输出environment prompt
        object_Tdata = Tdata['nodes']    # 所有的物体名称
        all_nodes = list(set([each_node['class_name'] for each_node in object_Tdata]))
        all_allowed_actions = EvolveGraphAction._member_names_  # 所有可执行的动作
        object_Tdata_text = ', '.join(all_nodes)
        all_allowed_actions_text = ', '.join(all_allowed_actions)
        environment_prompt = "In the environment, all available objects are " + object_Tdata_text + ". And all executable action are " + all_allowed_actions_text + '. When generating the plans. Robot agent only can execute these actions and interact with available objects.'

        env_graph = EnvironmentGraph(Tdata)

        assert len([n['id'] for n in Tdata["nodes"] if n['class_name'] == 'character']) == 1
        agent_id = [n['id'] for n in Tdata["nodes"] if n['class_name'] == 'character'][0]
        print('agent id:', agent_id)

        # # 导入script文件
        # script = eachline['new_program_lines']

        name_equivalence = utils.load_name_equivalence()
        executor = ScriptExecutor(env_graph, name_equivalence)

        final_state = EnvironmentState(env_graph, name_equivalence, instance_selection=True)


        OberNState = []
        SuccessOrFail = []
        FailReasons = [] 
        GenAction = []

        prompt = "You are a robot agent. You need to complete some housework activities under the given instructions. " + '\n\nTask: ' + eachline['task'] + '\nDescription: ' + eachline['description'] + '\n'
        output_action = None
        step_index = 1
        
        # 防止一直没有end，人工设置阈值
        threshold = 0

        # 为了保证物体的一致性
        specific_objects = {}
        last_obj_id = -1

        while(output_action!='END'):
            print('-----------------------------generate one step-----------------------------------')
            print('---------------------------------------------------------------------------------')

            # 人工设置阈值
            if threshold>25:
                break
            threshold = threshold + 1

            print('prompt:', prompt)

            step_text = f"step{step_index}: "
            prompt = prompt + step_text
            # generation_action = Gen_newAction(prompt, tokenizer, model)
            generation_action = Gen_Action(prompt, tokenizer, model)

            # 进行预测反馈并且查看这个动作是否合理，如果不合理进行重新规划
            feeback_prompt = "You are a world simulator. You need to generate the feedback according to plans. " + prompt[100:] + generation_action + "\nFeedback: "
            print('****************feedback prompt:*****************', feeback_prompt)
            feedback_model = Gen_Action(feeback_prompt, tokenizer, model)
            if 'Error' in feedback_model:
                corrective_prompt = "You are a replanner. Given the original plan, you are expected to generate the new plan according to the feedback. " + prompt[100:] + generation_action + "\n A corrective step would be"
                print('******************corrective prompt**************', corrective_prompt)
                # corrective_action = Gen_newAction(corrective_prompt, tokenizer, model)
                corrective_action = Gen_Action(corrective_prompt, tokenizer, model)
                if 'Error' not in Gen_Action("You are a world simulator. You need to generate the feedback according to plans. " + prompt[100:] + corrective_action + "\nFeedback: ", tokenizer, model):
                    generation_action = corrective_action
                    
                    num_cor = num_cor + 1
                    print('************************+1***************************')

            GenAction.append(generation_action)
            if generation_action == 'END':
                break

            print('generation action:', generation_action)
            parsed_action = str2program_list([generation_action])[0][0]

            # 将index从1转换为对应的数字
            parsed_action, specific_objects, last_obj_id = change_obj_index(final_state, parsed_action, agent_id, specific_objects, last_obj_id)

            print('parsed_action:', parsed_action)
            
            prompt = prompt + generation_action + "\n"
            
            # 检查物体是否存在
            if '[' in parsed_action:
                
                # 检验是否有这个动作
                matches_action = re.findall(r'\[(.*?)\]', parsed_action)[0]
                if matches_action in dir(EvolveGraphAction):
                    
                    # 检验参数数量是否一致
                    parsed_action = check_action_format(parsed_action)
                    if '[' in parsed_action:
                        script = parse_script_line(parsed_action, 0)
                        success, final_state = executor.execute_one_step(Script([script]), final_state) 
                        print('success or not:', success)
                        print('failrue infomation:', executor.info.get_error_string())
                        SuccessOrFail.append(success)
                        FailReasons.append(executor.info.get_error_string())
                    else:
                        print('success or not:', False)
                        print('failrue infomation:', parsed_action)
                        SuccessOrFail.append(False)
                        FailReasons.append(parsed_action)
                else:
                    print('success or not:', False)
                    print('failrue infomation:', "agent doesn't have ability to " + matches_action)
                    SuccessOrFail.append(False)
                    FailReasons.append("agent doesn't have ability to " + matches_action)               
            else:
                print('success or not:', False)
                print('failrue infomation:', parsed_action)
                SuccessOrFail.append(False)
                FailReasons.append(parsed_action)

            temp_total_graph = final_state.to_dict()

            partial_graph = utils.get_visible_nodes(temp_total_graph, agent_id=agent_id)

            # agent拿着什么东西
            agent_has_objid = [n['to_id'] for n in temp_total_graph["edges"] if n['from_id'] == agent_id and "HOLD" in n["relation_type"]]
            agent_has_obj = [n['class_name'] for n in temp_total_graph["nodes"] if n['id'] in agent_has_objid]
            # agent看到什么东西
            obj_ids_close = [n['to_id'] for n in temp_total_graph["edges"] if n['from_id'] == agent_id and  n["relation_type"]=="CLOSE"]
            obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
            obj_ids = dict([(node['id'], node['class_name']) for node in temp_total_graph['nodes'] if node["id"] in obj_ids_close and node['class_name'] in obj])
            relations = list(set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in temp_total_graph["edges"] if n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"] not in ["CLOSE","FACING", "INSIDE", "HOLDS_LH", "HOLDS_RH"]]))    
            obj_states = [(node['class_name'], node['states']) for node in temp_total_graph['nodes'] if node['class_name'] in obj]
            objs = ""
            
            for ob_states in obj_states:
                if len(ob_states[1])>0:
                    objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
                else:
                    objs = objs + ob_states[0] + ', '
            objs = list(set(objs.split(', ')))
            objs = [ob for ob in objs if len(ob)>0]

            # objs = ', '.join(objs) + ', ' + ', '.join(relations)  + '. '
            if len(objs) == 0:
                if len(relations) != 0:
                    objs = ', '.join(relations) + '. '
                else:
                    objs = ""
            else:
                if len(relations) == 0:
                    objs = ', '.join(objs) + '. '
                else:
                    objs = ', '.join(objs) + ', ' + ', '.join(relations)  + '. '

            if len(agent_has_obj)>0:
                agent_has_obj = ', '.join(agent_has_obj)
                objs += f"You have {agent_has_obj}. "
            print('objs:', objs)
            observation_text = f"observation{step_index}: {objs}"
            prompt = prompt + observation_text + "\n"
            step_index = step_index + 1

            OberNState.append(objs)
        
        eachline['ObserANDState'] = OberNState
        eachline['SuccessOrFail'] = SuccessOrFail
        eachline['FailReasons'] = FailReasons
        eachline['GenAction'] = GenAction

        print(eachline)

        jsonl_add_data(output_path, [eachline])


evaluation(data, "outputs/")