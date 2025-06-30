import os
import numpy as np


def eye(path):
    eye_path = path + 'Eye-alignment'
    files = os.listdir(eye_path)[0]

    with open(eye_path + '/' + files, 'r') as f:
        data = f.readlines()

    errors = []

    for d in data:
        # d = 'Episode: 0, Target Position: 0, Ruler Position: -0.03888893\n'
        target = d.split('Target Position: ')[1].split(',')[0]
        ruler = d.split('Ruler Position: ')[1].split('\n')[0]
        errors.append(abs(float(target) - float(ruler)))
        

    return -sum(errors)/len(errors)

def theory(path):
    theory_path = path + 'Theory_of_bahavior_test'
    files = os.listdir(theory_path)[0]
    with open(theory_path + '/' + files, 'r') as f:
        data = f.readlines()

    errors = []
    for d in data:
        if 'Fail' in d:
            continue
        true = d.split('true position: (')[1].split('),')[0]
        true = np.array([float(i) for i in true.split(',')])
        pred = d.split('predicted position: (')[1].split(')')[0]
        pred = np.array([float(i) for i in pred.split(',')])
        errors.append(((true - pred)**2).sum())

    return -sum(errors)/len(errors)


def puzzle(path):
    puzzle_path = path + 'Puzzle_solving'
    files = os.listdir(puzzle_path)[0]
    with open(puzzle_path + '/' + files, 'r') as f:
        data = f.readlines()

    co = 0

    for d in data:
        if 'Not Answered' in d:
            continue
        correct = d.split('Correct Answer: ')[1].split(',')[0]
        chosen = d.split('Chosen Answer: ')[1].split(',')[0]

        if correct == chosen:
            co+= 1

    return co/12

def spatial(path):
    spatial_path = path + 'Spatial_mapping'
    files = os.listdir(spatial_path)[0]
    with open(spatial_path + '/' + files, 'r') as f:
        data = f.readlines()

    co = 0

    for d in data:
        if 'Not Answered' in d:
            continue
        correct = d.split('Correct Answer: ')[1].split(',')[0]
        chosen = d.split('Chosen Answer: ')[1].split(',')[0]

        if correct == chosen:
            co+= 1


    return co/6

def reflex(path):
    reflex_path = path + 'Reflection_test'
    files = os.listdir(reflex_path)[0]
    with open(reflex_path + '/' + files, 'r') as f:
        data = f.readlines()

    time = []

    for d in data:
        if 'Fail ' in d:
            continue
        t = d.split('Reflection Time: ')[1].split('\n')[0]
        time.append(float(t))

    if len(time) == 0:
        return -9999999

    return -sum(time)/len(time)




def overall(path):
    return {'sub': path.split('/')[-2][:2],
        'eye': eye(path),
        'theory': theory(path),
        'puzzle': puzzle(path),
        'spatial': spatial(path),
        'reflex': reflex(path)
    }



if __name__ == "__main__":
    subs = os.listdir('/home/grl/Desktop/CREW/Data/FINAL/cog data/')
    sub_data = []
    for sub in subs:
        sub = sub + '/'
        sub_data.append(overall('/home/grl/Desktop/CREW/Data/FINAL/cog data/' + sub))

    num_subs = len(sub_data)

    sub_data.sort(key=lambda x: x['eye'], reverse = True)

    for i in range(num_subs):
        sub_data[i]['eye_rank'] = i

    sub_data.sort(key=lambda x: x['theory'], reverse = True)
    for i in range(num_subs):
        sub_data[i]['theory_rank'] = i

    sub_data.sort(key=lambda x: x['puzzle'], reverse = True)
    for i in range(num_subs):
        sub_data[i]['puzzle_rank'] = i

    sub_data.sort(key=lambda x: x['spatial'], reverse = True)
    for i in range(num_subs):
        sub_data[i]['spatial_rank'] = i

    sub_data.sort(key=lambda x: x['reflex'], reverse = True)
    for i in range(num_subs):
        sub_data[i]['reflex_rank'] = i

    sub_data.sort(key=lambda x: x['eye_rank'] + x['theory_rank'] + x['puzzle_rank'] + x['spatial_rank'] + x['reflex_rank'])
    for i in range(num_subs):
        sub_data[i]['overall_rank'] = i

    for d in sub_data:
        print(d)

    print([s['sub'] for s in sub_data][:15])