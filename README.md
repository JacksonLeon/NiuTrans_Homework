# NiuTrans_Homework

课堂练习：[基于NiuTensor的FNNLM](https://gitee.com/JacksonLeon/NiuTrans-Homework.git)

通过简单调参，能够将ppl从231.69降到201.54
执行命令：`bin/NiuTensor.GPU -fnnlm -dev 2 -lrate 0.006 -wbatch 256 -minmax 0.1 -nepoch 5 -n 5 -hdepth 1 -hsize 128 -esize 100 -train data/wsj.train -test data/wsj.test -output work/wsj-1.prob -vsize 10000 -model work/wsj-1.model -autodiff >> out-1.log 2<&1 &`

训练过程如下：
```
args:
 -dev=3
 -lrate=0.006000
 -wbatch=256
 -minmax=0.100000
 -nepoch=5
 -n=5
 -hdepth=1
 -hsize=128
 -esize=100
 -train=data/wsj.train
 -test=data/wsj.test
 -output=work/wsj.prob
 -vsize=10000
 -model=work/wsj.model
 -autodiff=true
[INFO] elapsed=29.9s, step=100, epoch=1, ngram=25600, ppl=1407.175
[INFO] elapsed=65.6s, step=200, epoch=1, ngram=51200, ppl=1026.999
[INFO] elapsed=101.3s, step=300, epoch=1, ngram=76800, ppl=855.181
[INFO] elapsed=139.7s, step=400, epoch=1, ngram=102400, ppl=751.295
[INFO] elapsed=174.6s, step=500, epoch=1, ngram=128000, ppl=688.269
[INFO] elapsed=201.6s, step=600, epoch=1, ngram=153600, ppl=635.385
[INFO] elapsed=237.5s, step=700, epoch=1, ngram=179200, ppl=594.812
[INFO] elapsed=272.3s, step=800, epoch=1, ngram=204800, ppl=569.784
[INFO] elapsed=311.0s, step=900, epoch=1, ngram=230400, ppl=541.213
[INFO] ppl=297.65
[INFO] test finished (took 44.3s, sentence=1359 and ngram=30192)
[INFO] elapsed=391.2s, step=1000, epoch=2, ngram=255992, ppl=344.523
[INFO] elapsed=430.7s, step=1100, epoch=2, ngram=281592, ppl=331.115
[INFO] elapsed=457.3s, step=1200, epoch=2, ngram=307192, ppl=319.903
[INFO] elapsed=485.4s, step=1300, epoch=2, ngram=332792, ppl=310.391
[INFO] elapsed=516.9s, step=1400, epoch=2, ngram=358392, ppl=307.981
[INFO] elapsed=545.3s, step=1500, epoch=2, ngram=383992, ppl=298.710
[INFO] elapsed=571.3s, step=1600, epoch=2, ngram=409592, ppl=293.196
[INFO] elapsed=599.2s, step=1700, epoch=2, ngram=435192, ppl=291.659
[INFO] elapsed=623.5s, step=1800, epoch=2, ngram=460792, ppl=286.164
[INFO] ppl=238.67
[INFO] test finished (took 34.3s, sentence=1359 and ngram=30192)
[INFO] elapsed=684.4s, step=1900, epoch=3, ngram=486384, ppl=251.827
[INFO] elapsed=709.2s, step=2000, epoch=3, ngram=511984, ppl=235.794
[INFO] elapsed=736.1s, step=2100, epoch=3, ngram=537584, ppl=230.734
[INFO] elapsed=746.7s, step=2200, epoch=3, ngram=563184, ppl=222.162
[INFO] elapsed=747.9s, step=2300, epoch=3, ngram=588784, ppl=223.054
[INFO] elapsed=749.1s, step=2400, epoch=3, ngram=614384, ppl=216.743
[INFO] elapsed=750.3s, step=2500, epoch=3, ngram=639984, ppl=214.492
[INFO] elapsed=751.5s, step=2600, epoch=3, ngram=665584, ppl=214.493
[INFO] elapsed=752.7s, step=2700, epoch=3, ngram=691184, ppl=211.622
[INFO] ppl=216.26
[INFO] test finished (took 18.7s, sentence=1359 and ngram=30192)
[INFO] elapsed=790.6s, step=2800, epoch=4, ngram=716776, ppl=197.779
[INFO] elapsed=817.9s, step=2900, epoch=4, ngram=742376, ppl=184.935
[INFO] elapsed=846.2s, step=3000, epoch=4, ngram=767976, ppl=182.793
[INFO] elapsed=872.3s, step=3100, epoch=4, ngram=793576, ppl=175.429
[INFO] elapsed=901.7s, step=3200, epoch=4, ngram=819176, ppl=176.372
[INFO] elapsed=926.4s, step=3300, epoch=4, ngram=844776, ppl=171.712
[INFO] elapsed=953.6s, step=3400, epoch=4, ngram=870376, ppl=170.018
[INFO] elapsed=983.0s, step=3500, epoch=4, ngram=895976, ppl=170.164
[INFO] elapsed=1009.8s, step=3600, epoch=4, ngram=921576, ppl=168.001
[INFO] ppl=205.10
[INFO] test finished (took 36.1s, sentence=1359 and ngram=30192)
[INFO] elapsed=1068.7s, step=3700, epoch=5, ngram=947168, ppl=163.334
[INFO] elapsed=1091.3s, step=3800, epoch=5, ngram=972768, ppl=150.263
[INFO] elapsed=1117.3s, step=3900, epoch=5, ngram=998368, ppl=150.108
[INFO] elapsed=1147.4s, step=4000, epoch=5, ngram=1023968, ppl=142.465
[INFO] elapsed=1168.6s, step=4100, epoch=5, ngram=1049568, ppl=143.813
[INFO] elapsed=1196.5s, step=4200, epoch=5, ngram=1075168, ppl=141.933
[INFO] elapsed=1227.2s, step=4300, epoch=5, ngram=1100768, ppl=139.638
[INFO] elapsed=1251.5s, step=4400, epoch=5, ngram=1126368, ppl=139.994
[INFO] elapsed=1276.8s, step=4500, epoch=5, ngram=1151968, ppl=138.458
[INFO] ppl=201.54
[INFO] test finished (took 34.3s, sentence=1359 and ngram=30192)
[INFO] elapsed=1323.7s, step=4550, epoch=5, ngram=1164760, ppl=138.094
[INFO] training finished (took 1323.7s, step=4550 and epoch=5)
[INFO] model saved
[INFO] model loaded
[INFO] ppl=201.54
[INFO] test finished (took 37.5s, sentence=1359 and ngram=30192)
```
