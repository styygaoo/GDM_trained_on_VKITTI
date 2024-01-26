import os
import shutil

files = [ f for f in os.listdir("/HOMES/yigao/Downloads/eval_test/eval_tesst") if '.png' in f.lower() ]

# print(files)
sorted_files = sorted(files, key=lambda x: int(x.split('_', 3)[-1].split(".",2)[0]))
# print(sorted_files)


with open("/HOMES/yigao/Downloads/eval_test/desktop.ini") as f:
    lines = [line for line in f]
    # print(lines)

for line in lines[1:]:
    print(line)
    for file in sorted_files:
    # print(file.split(".", 2)[0].split("_", 3)[-1])
        if file.split(".", 2)[0].split("_", 3)[-1] == line.split("_", 2)[2].split(".",1)[0]:
            print(file)

            new_path = line.split("_", 2)[2].split(".",1)[0]
            if not os.path.exists("/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/eval_first/" + new_path):
                os.makedirs("/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/eval_first/" + new_path)

            shutil.copy("/HOMES/yigao/Downloads/eval_test/eval_tesst/" + file,
                        "/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/eval_first/" + new_path)
            print("File copied successfully.")
            # shutil.move("/HOMES/yigao/Downloads/eval_test/eval_tesst/" + file,
            #             "/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/eval_first/" + new_path)

# for i in lines[1:]:
#     print(i)
# for file in files:
#     print(file)
        # print(file.split(".", 2)[0].split("_", 3)[-1])
        # if file.split(".", 2)[0].split("_", 3)[-1] in i.split("_", 2)[2].split(".",1)[0]:
        #     print(file)
    # if any(file.split(".", 2)[0].split("_", 3)[-1] in i for i in lines):
    #     print(file)
    #     # matching = [i for i in lines if file.split("_", 2)[2].split(".", 1)[0] in i]
    #     # print(matching)
    #
    # else:
    #     print(22)


    # new_path = 'downloaded_images/' + image
    # shutil.move(image, new_path)
