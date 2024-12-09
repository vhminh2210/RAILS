import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prj_ids', type=str, default='', help= 'Project id-name list')
parser.add_argument('--tpl_ids', type=str, default='', help= 'TPL id-name list')
parser.add_argument('--rec_txt', type=str, default='', help= 'Recommended list')
parser.add_argument('--test_txt', type=str, default='', help= 'GroundTruth')
parser.add_argument('--corpus_txt', type=str, default='', help= 'Full corpus: query + test for cold-start scheme')
parser.add_argument('--projects_txt', type=str, default='', help= 'Testing projects')

parser.add_argument('--root', type=str, default='msr', help= 'Save dir')

args = parser.parse_args()

PRJ_IDS = args.prj_ids
TPL_IDS = args.tpl_ids
REC_TXT = args.rec_txt
TEST_TXT = args.test_txt
CORPUS_TXT = args.corpus_txt

ROOT = args.root
REC_ROOT = os.path.join(ROOT, "Recommendations")
TEST_ROOT = os.path.join(ROOT, "GroundTruth")
QUERY_ROOT = os.path.join(ROOT, "Queries")
PROJECTS_TXT = os.path.join(ROOT, "projects.txt")

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

if not os.path.exists(REC_ROOT):
    os.makedirs(REC_ROOT)

if not os.path.exists(TEST_ROOT):
    os.makedirs(TEST_ROOT)

if not os.path.exists(QUERY_ROOT):
    os.makedirs(QUERY_ROOT)

# Write recommendations
with open(REC_TXT, 'r') as file:
    lines = file.readlines()
    file.close()

for line in lines:
    prj_id = int(line.split()[0])
    rec_ids = list(set([int(x) for x in line.split()[1 : ]]))
    cnt = len(rec_ids)
    with open(os.path.join(REC_ROOT, prj_list[prj_id]), 'w') as file:
        print(prj_list[prj_id])
        for idx in rec_ids:
            file.write(tpl_list[int(idx)] + ' ' + str(cnt) + '\n')
            cnt -= 1
        file.close()

# Write groundtruths
with open(TEST_TXT, 'r') as file:
    lines = file.readlines()
    file.close()

for line in lines:
    prj_id = int(line.split()[0])
    test_ids = list(set([int(x) for x in line.split()[1 : ]]))
    cnt = len(test_ids)
    with open(os.path.join(TEST_ROOT, prj_list[prj_id]), 'w') as file:
        print(prj_list[prj_id])
        for idx in test_ids:
            file.write(tpl_list[int(idx)] + ' ' + str(cnt) + '\n')
            cnt -= 1
        file.close()

# Write queries
corpus_dict = {}
test_dict = {}

with open(CORPUS_TXT, 'r') as file:
    corpuslines = file.readlines()
    file.close()

with open(TEST_TXT, 'r') as file:
    testlines = file.readlines()
    file.close()

for line in corpuslines:
    words = line.split()
    corpus_dict[words[0]] = list(set([int(x) for x in words[1 : ]]))

for line in testlines:
    words = line.split()
    test_dict[words[0]] = list(set([int(x) for x in words[1 : ]]))

for user in test_dict.keys():
    corpus = corpus_dict[user]
    testlist = test_dict[user]

    query_ids = list(set(corpus) - set(testlist))
    cnt = len(query_ids)
    prj_id = int(user)
    with open(os.path.join(QUERY_ROOT, prj_list[prj_id]), 'w') as file:
        print(prj_list[prj_id])
        for idx in query_ids:
            assert idx not in testlist, "Data leakage from query to test!"
            file.write(tpl_list[int(idx)] + ' ' + str(cnt) + '\n')
            cnt -= 1
        file.close()

with open(PROJECTS_TXT, 'w') as file:
    for user in test_dict.keys():
        file.write(prj_list[int(user)] + '\n')
    file.close()