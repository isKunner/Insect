import soundfile as sf
import os

pre_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
mid_path = r"/TreeData/field/field/train/clean/folder_8/fig1clips_22-02-2020_23-02-2020"
save_path = r"/TreeData/field/field/train/clean/folder_8/convert_fig1clips_22-02-2020_23-02-2020"

# i为0时对应的target为clean；i为1时对应的target为infested
path = pre_path + mid_path + '/'  # train from scratch using field data
# root 所指的是当前正在遍历的这个文件夹的本身的地址
# dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
# files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
# 函数会自动改变root的值使得遍历所有的子文件夹。所以返回的三元元组的个数为所有子文件夹（包括子子文件夹，子子子文件夹等等）加上1（根文件夹）
ogg_files = [f for f in os.listdir(path) if f.endswith('.ogg')]

for ogg_file in ogg_files:
    input_path = os.path.join(path, ogg_file)
    output_path = os.path.join(save_path, os.path.splitext(ogg_file)[0] + '.wav')
    try:
        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)
    except Exception as e:
        print(e)