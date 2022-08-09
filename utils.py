
import os
def countDirectories(d):
    count_dirs = 0
    count_files = 0
    for path in os.listdir(d):
        if os.path.isdir(os.path.join(d, path)):
            count_dirs += 1
            sub_path = os.path.join(d, path)
            for file in os.listdir(sub_path):
                if os.path.isfile(os.path.join(sub_path, file)):
                    count_files += 1

    return count_dirs, count_files

def preprocessLambda(func=lambda x: (x/255.0) * 2.0 - 1.0):
    def _preprocess(x):
        x = func(x)
        return x

    return _preprocess

def factorBy(factor=1.0/255.0):
    def _preprocess(x):
        x = x * factor
        return x

    return _preprocess