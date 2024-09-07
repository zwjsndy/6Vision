res=[]
with open('./temp/res0.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./temp/res1.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./temp/res2.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./temp/res3.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./temp/res4.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./temp/res5.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./all_data.txt', 'r') as f:
    for line in f:
        res.append(line.strip('\n'))
f.close()
with open('./all_data.txt', 'w', encoding = 'utf-8') as f:
    for addr in list(set(res)):
        f.write(addr + '\n')
