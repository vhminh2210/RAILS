import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prj_ids', type=str, default='', help= 'Project id-name list')
parser.add_argument('--tpl_ids', type=str, default='', help= 'TPL id-name list')
parser.add_argument('--rec_txt', type=str, default='', help= 'Recommended list')
parser.add_argument('--root', type=str, default='msr', help= 'Save dir')
args = parser.parse_args()

PRJ_IDS = args.prj_ids
TPL_IDS = args.tpl_ids
REC_TXT = args.rec_txt
ROOT = args.root

def convertPrj(prj_link):
    '''
    git://github.com/3mtee/jforum.git -> 3mtee__jforum
    '''
    words = prj_link.replace('.git', '').split('/')
    return '__'.join(words[-2:])

prj_list = []

with open(PRJ_IDS, 'r') as file:
    lines = file.readlines()
    file.close()

for line in lines:
    prj_list.append(convertPrj(line.split()[1]))

tpl_list = []

with open(TPL_IDS, 'r') as file:
    lines = file.readlines()
    file.close()

for line in lines:
    tpl = '#DEP#' + line.split()[1]
    tpl_list.append(tpl)

if not os.path.exists(ROOT):
    os.makedirs(ROOT)

with open(REC_TXT, 'r') as file:
    lines = file.readlines()
    file.close()

for line in lines:
    prj_id = int(line.split()[0])
    rec_ids = [int(x) for x in line.split()[1 : ]]
    cnt = len(rec_ids)
    with open(os.path.join(ROOT, prj_list[prj_id]), 'w') as file:
        print(prj_list[prj_id])
        for idx in rec_ids:
            file.write(tpl_list[int(idx)] + ' ' + str(cnt) + '\n')
            cnt -= 1
        file.close()