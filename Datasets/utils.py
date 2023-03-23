import os

def modify_split_file(src_path, dst_path, prefix):
    def save_pairs(pairs, save_path):

        # Modify prefix

        f = open(save_path, "w+")

        for pair in pairs:

            new_context = ''

            pair[0] = os.path.join(prefix, pair[0])
            pair[2] = os.path.join(prefix, pair[2])

            if not(os.path.isdir(pair[0]) and os.path.isdir(pair[2])):
                print('Dir %s / %s not exists' % (pair[0], pair[2]))
                break


            # Original train/test/val pairs
            path1, id1, path2, id2 = pair
            # path1, path2 = path1.replace('ActionVerification', 'CSV').replace('/p300', '/public/home/qianych'), path2.replace('ActionVerification', 'CSV').replace('/p300', '/public/home/qianych')
            # path1, path2 = path1.replace('frames', 'flows'), path2.replace('frames', 'flows')
            new_context += path1 + ' ' + id1 + ' ' + path2 + ' ' + id2 + '\n'

            f.write(new_context)

        f.close()
        print("Save %d pairs in total" % len(pairs))





    def read_pairs(txt_path):

        data = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                data.append(line.split(' '))

        return data

    pairs = read_pairs(src_path)
    save_pairs(pairs, dst_path)


if __name__ == "__main__":
    src_file = '/public/home/qianych/code/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/Datasets/CSV/test_pairs.txt'
    dst_file = '/storage/Data/qianych/dataset/CSV/splits/test_pairs.txt'
    prefix = '/storage/Data/qianych/dataset/CSV/frames'

    modify_split_file(src_file, dst_file, prefix)