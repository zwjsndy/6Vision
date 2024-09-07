import ipaddress
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import time
from collections import Counter
import re
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
def alias_Detect(b,source_ip): 
    temp=[]
    b=convert(b)
    print('alias_detecting--------')
    for word in b:
        word = word[:16]
        temp.append(word)
    all_prefix = list(set(temp))
    res = []
    baseMap = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
               8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    for word in all_prefix:
        for i in range(0, 16):
            y = word
            y += baseMap[i]
            for j in range(15):
                m = random.randint(0, 15)
                m = baseMap[m]
                y += m
            y = str2ipv6(y)
            res.append(y)
    temp=Scan(res, source_ip, './res', 1)
    temp = convert(temp)
    res1=[]
    for word in temp:
        word = word[:16]
        res1.append(word)
    prefix = Counter(res1)
    prefix = dict(prefix)
    aliased_prefix = []
    for key in list(prefix.keys()):
        if prefix[key] >= 15:
            aliased_prefix.append(key)

    all1 = []
    for word in b:
        if word[:16] not in aliased_prefix:
            try:
                all1.append(str2ipv6(word))
            except:
                continue

    return all1  


def Scan(addr_set, source_ip, output_file, tid):

    scan_input = output_file + '/zmap/scan_input_{}.txt'.format(tid)
    scan_output = output_file + '/zmap/scan_output_{}.txt'.format(tid)

    with open(scan_input, 'w', encoding='utf-8') as f:
        for addr in addr_set:
            f.write(addr + '\n')

    active_addrs = set()
    command = 'sudo zmap --ipv6-source-ip={} --ipv6-target-file={} -M icmp6_echoscan -p 80 -q -o {}'\
        .format(source_ip, scan_input, scan_output)
    print('[+]Scanning {} addresses...'.format(len(addr_set)))
    t_start = time.time()
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # ret = p.poll()
    while p.poll() == None:
        pass

    if p.poll() is 0:
        # with open(output_file, 'a', encoding='utf-8') as f:
        # time.sleep(1)
        for line in open(scan_output):
            if line != '':
                active_addrs.add(line[0:len(line) - 1])
                # f.write(line)

    print('[+]Over! Scanning duration:{} s'.format(time.time() - t_start))
    print('[+]{} active addresses detected!'
          .format(len(active_addrs)))
    return active_addrs


def convert(seeds):
    result = []
    for line in seeds:
        line = line.split(":")
        for i in range(len(line)):
            if len(line[i]) == 4:
                continue
            if len(line[i]) < 4 and len(line[i]) > 0:
                zero = "0"*(4 - len(line[i]))
                line[i] = zero + line[i]
            if len(line[i]) == 0:
                zeros = "0000"*(9 - len(line))
                line[i] = zeros
        result.append("".join(line)[:32])
    return result


def stdIPv6(addr: str):
    return ipaddress.ip_address(addr)


def str2ipv6(a: str):
    pattern = re.compile('.{4}')
    addr = ':'.join(pattern.findall(a))
    return str(stdIPv6(addr))


def numConversion(a):
    baseMap = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
               '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15}
    result = []
    for item in a:
        temp = []
        for word in item:
            try:
                temp.append(baseMap[word])
            except:
                print(word)
        result.append(temp)
    return result

import argparse
parse=argparse.ArgumentParser()
parse.add_argument('--source_ip', type=str)
args=parse.parse_args()
source_ip=args.source_ip
for cl in range(6):
    result1=[]
    path='./temp/res'+str(cl)+'.txt'
    with open(path, 'r') as f:
        for line in f:
            result1.append(line.strip('\n'))
    res = Scan(result1, source_ip, './res', 1)
    res=alias_Detect(res,source_ip)
    with open(path, 'w', encoding = 'utf-8') as f:
        for addr in list(set(res)):
            f.write(addr + '\n')
