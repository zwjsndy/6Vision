# 6Vision
6Vision: Image-encoding-based IPv6 Target Generation in Few-seed Scenarios
本模型解决少量种子场景下探测问题。要求输入的每个BGP下种子地址尽可能地少，如小于10个。本模型可以丰富这些BGP下种子地址的数目。但在处理丰富种子场景存在速度慢的问题，建议用别的方法。

环境要求
pytorch 1.12.0 GPU版本，CPU也行，就是跑的非常慢
totrchvision   0.13.0版本
Python 3.8.18
其余的一些包根据import安装就行


运行方式
输入：字典格式，每个字典中的元素格式如下，Key为BGP前缀，value 为种子地址的列表（示例见./new_allseeds.pkl）
(1)聚类（默认聚类成6类）
运行cluster.py，会得到聚类后的结果label.txt
(2)训练模型（训练6个模型）（这6个模型相互独立，互不影响）
运行Gatedpixcelcnn.py，会得到6个模型，在./model文件夹下
(3)地址生成
运行gen.py包含一个参数  --num   参数值可选取0-5的整数，表示用0-5这6个模型生成候选地址
建议0-5都运行，即并行这6个程序
python3 gen.py  --num 0
python3 gen.py  --num 1
python3 gen.py  --num 2
python3 gen.py  --num 3
python3 gen.py  --num 4
python3 gen.py  --num 5
生成的地址保存在./temp文件夹下
(4)别名检测与地址探活
（这一步需要用有zmapv6的机器）!!!!!!
运行aliaseddetect.py 注意将里面的sourceip参数改为自己主机的IP
(5)根据反馈结果微调每个模型
并行这6个程序
python3 retrain.py --num 0
python3 retrain.py --num 1
python3 retrain.py --num 2
python3 retrain.py --num 3
python3 retrain.py --num 4
python3 retrain.py --num 5
(6)汇聚这一轮收集到的所有地址
python3 alldata.py
数据保存在了all_data.txt下


若想探测更多轮次，重复运行（4）（5）（6）即可，只需要按顺序进行步骤（4）（5）（6）,其他的不需要运行
注意：（4）要在有zmapv6的机器上运行，其他的要在GPU机器上运行，CPU机器也行，会很慢
