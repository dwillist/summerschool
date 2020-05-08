import random
import json

# generate n points above and below y = ax + b
def gen_linear(a, b, n):
    inputs = []
    solutions = []
    total_count = 0
    above_count = 0
    while total_count < n:
        x_gen = random.random()
        y_gen = random.random()
        solution = None
        if (y_gen > (x_gen * a + b)) and above_count <= n/2:
            above_count += 1
            solution = (0,1)
        elif (total_count - above_count) <= n/2:
            solution = (1,0)
        if solution != None:
            total_count += 1
            inputs.append((x_gen, y_gen))
            solutions.append(solution)
    return inputs,solutions

# generate an equal number of points within a target shape centered at (c_x, c_y)
def gen_target(c_x, c_y, rad, n):
    inputs = []
    solutions = []
    inner_count = 0
    total_count = 0
    while total_count < n:
        x_gen = random.random()
        y_gen = random.random()
        if (inner_count <= n/2) and (euclidean_distance((x_gen, y_gen), (c_x, c_y)) < rad):
            inner_count += 1
            solutions.append((0,1))
            total_count += 1
            inputs.append((x_gen, y_gen))
        elif ((total_count - inner_count) <= n/2) and (euclidean_distance((x_gen, y_gen), (c_x, c_y)) >= rad):
            solutions.append((1,0))
            total_count += 1
            inputs.append((x_gen, y_gen))
    return inputs, solutions

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2)**2 + (y1 - y2)**2) ** (1/2)

def data_to_json(inputs, solutions, output_path):
    result = {}
    result['inputs'] = inputs
    result['solutions'] = solutions
    with open(output_path, 'w') as outfile:
        json.dump(result, outfile)


def main():
    #data_to_json(*gen_linear(.5,.5, 1000), '../bifurcated-test.json')
    data_to_json(*gen_target(.5,.5,.5,1000), '../target-test.json')


if __name__ == '__main__':
    main()
