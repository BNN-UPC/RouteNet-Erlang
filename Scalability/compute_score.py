from itertools import zip_longest

sample_num = 0
APE = []

# LOAD THREE FILES AT ONCE
with open("SolutionGNNNet2021/ground_truth.txt", "r") as ground_truth, open("test.txt", "r") as uploaded_file, open(
        "SolutionGNNNet2021/path_per_sample.txt", "r") as path_per_sample:
    # LOAD IT LINE BY LINE (NOT ALL AT ONCE)
    for actual, prediction, n_paths in zip_longest(ground_truth, uploaded_file, path_per_sample):
        # CASE LINE COUNT DOES NOT MATCH
        if n_paths is None:
            print("ERROR: File must contain 1560 lines in total. "
                  "Looks like the uploaded file has {}".format(sample_num))
            break
        if prediction is None:
            print("ERROR: File must have 1560 lines in total. "
                  "Looks like the uploaded file has {}".format(sample_num))
            break

        # REMOVE THE \n CHAR
        actual = actual.rstrip()
        prediction = prediction.rstrip()
        n_paths = n_paths.rstrip()

        # SPLIT THE LIST, CONVERT TO FLOAT AND THEN TO LIST
        actual = list(map(float, actual.split(";")))
        prediction = list(map(float, prediction.split(";")))

        # CASE SIZES DO NOT MATCH
        if len(prediction) != int(n_paths):
            print("ERROR in line {}: The line should have size {} but it has size {}".format(sample_num, n_paths,
                                                                                             len(prediction)))
            break

        # COMPUTE THE ABS. PERC. ERROR FOR EACH PATH
        for a, p in zip(actual, prediction):
            APE.append(abs((a - p) / a))

        sample_num += 1

    # COMPUTE FINAL SCORE
    MAPE = (sum(APE) / len(APE))*100
