# NiuTrans_Homework

课堂练习：[基于NiuTensor的FNNLM](https://gitee.com/JacksonLeon/NiuTrans-Homework.git)
通过简单调参，能够将ppl从231.69降到214.09
训练参数如下：
```
-dev=0
-lrate=0.006200
-wbatch=256
-minmax=0.080000
-nepoch=5
-n=3
-hdepth=1
-hsize=128
-esize=100
-train=data/wsj.train
-test=data/wsj.test
-output=work/wsj-8.prob
-vsize=10000
-model=work/wsj-8.model
-autodiff=true
```
