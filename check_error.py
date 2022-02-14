import re
import os

TOPOLOGIES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240,
              260, 280, 300]
errors = []
for line in open('./Scalability/check_error.out'):
    match = re.search('MAPE: (\d+\.\d+)', line)
    if match:
        errors.append(float(match.group(1)))
errors = list(dict.fromkeys(errors))
print(len(errors))
print(len(TOPOLOGIES))
errors = [errors[i:i + 20] for i in range(0, len(errors), 20)]
for i, t in enumerate(TOPOLOGIES):
    print("TOPO SIZE: {}".format(t))
    count = 0
    for j, error in enumerate(errors[i]):
        if error < 15:
            count += 1
        else:
            os.remove("./data/removed/{}/results_400-2000_{}_{}.tar.gz".format(t, j, j))
    print("COUNT {}".format(count))
