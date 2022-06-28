import os, sys
import json

sys.path.append("/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason/options") # Adds higher directory to python modules path.
sys.path.append("/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason")
from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
import utils.utils as utils
import torch

import warnings

def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type


def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i+1]:
            return False
        if pred[i] == 2:
            break
    return True

if torch.cuda.is_available():
  print("cuda available..")  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
warnings.filterwarnings("ignore")

opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
executor = get_executor(opt)
model = Seq2seqParser(opt).to(device)

print('| running test')
stats = {
    #'count': 0,
    #'count_tot': 0,
    #'exist': 0,
    #'exist_tot': 0,
    #'compare_num': 0,
    #'compare_num_tot': 0,
    #'compare_attr': 0,
    #'compare_attr_tot': 0,
    #'query': 0,
    #'query_tot': 0,
    'correct_ans': 0,
    'correct_ans_R': 0,
    'correct_ans_NR': 0,
    'correct_prog': 0,
    'guess':0,
    'total': 0
}
for x, y, ans, idx, complete_image_idx, constraint_type in loader:
    x = x.to(device = device)
    y = y.to(device = device)
    ans = ans.to(device = device)
    idx = idx.to(device = device)
    #complete_image_idx = complete_image_idx.to(device = device)
    #constraint_type =  constraint_type.to(device = device)
    ct_np = constraint_type.cpu().detach().numpy()
    cs_np = complete_image_idx.cpu().detach().numpy()
    model.set_input(x, y)
    pred_program = model.parse()
    y_np, pg_np, idx_np, ans_np = y.cpu().detach().numpy(), pred_program.cpu().detach().numpy(), idx.cpu().detach().numpy(), ans.cpu().detach().numpy()
    
    for i in range(pg_np.shape[0]): 
        pred_ans, flag_reason, flag_error, flag_re, flag_dir, g  = executor.run(pg_np[i], idx_np[i], cs_np[i], ct_np[i], 'val', guess=True)
        gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

        q_type = find_clevr_question_type(executor.vocab['program_idx_to_token'][y_np[i][1]])
        if pred_ans == gt_ans:
            stats['correct_ans']+=1
            if g == 1:
              stats['guess'] += 1 
            #stats[q_type] += 1
            elif flag_reason and g == 0:
              stats['correct_ans_R'] += 1
            elif flag_dir ==1 and g==0:
              stats['correct_ans_NR'] += 1
        if check_program(pg_np[i], y_np[i]):
            stats['correct_prog'] += 1

        #stats['%s_tot' % q_type] += 1
        stats['total'] += 1
    print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))

result = {
    #'count_acc': stats['count'] / stats['count_tot'],
    #'exist_acc': stats['exist'] / stats['exist_tot'],
    #'compare_num_acc': stats['compare_num'] / stats ['compare_num_tot'],
    #'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
    #'query_acc': stats['query'] / stats['query_tot'],
    'program_acc': stats['correct_prog'] / stats['total'],
    'overall_acc': stats['correct_ans'] / stats['total'],
    'correct-ans':stats['correct_ans'],
    'correct-ans-guess': stats['guess'],
    'correct-ans-NR': stats['correct_ans_NR'],
    'correct-ans-R': stats['correct_ans_R']
    
}
print(result)

utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    
