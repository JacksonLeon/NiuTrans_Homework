# NiuTrans_Homework

课堂练习：[基于NiuTensor的FNNLM](https://gitee.com/JacksonLeon/NiuTrans-Homework.git)

通过简单调参，能够将ppl从231.69降到203.60
训练过程如下：
```
nohup: ignoring input
args:
 -dev=2
 -lrate=0.006180
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
[INFO] elapsed=24.9s, step=100, epoch=1, ngram=25600, ppl=1418.793
[INFO] elapsed=46.4s, step=200, epoch=1, ngram=51200, ppl=1013.924
[INFO] elapsed=75.4s, step=300, epoch=1, ngram=76800, ppl=846.599
[INFO] elapsed=102.5s, step=400, epoch=1, ngram=102400, ppl=743.885
[INFO] elapsed=131.7s, step=500, epoch=1, ngram=128000, ppl=680.670
[INFO] elapsed=159.9s, step=600, epoch=1, ngram=153600, ppl=627.893
[INFO] elapsed=185.7s, step=700, epoch=1, ngram=179200, ppl=586.952
[INFO] elapsed=209.5s, step=800, epoch=1, ngram=204800, ppl=561.712
[INFO] elapsed=236.6s, step=900, epoch=1, ngram=230400, ppl=533.174
[INFO] ppl=292.74
[INFO] test finished (took 31.7s, sentence=1359 and ngram=30192)
[INFO] elapsed=297.9s, step=1000, epoch=2, ngram=255992, ppl=338.621
[INFO] elapsed=323.7s, step=1100, epoch=2, ngram=281592, ppl=323.171
[INFO] elapsed=352.9s, step=1200, epoch=2, ngram=307192, ppl=312.389
[INFO] elapsed=380.2s, step=1300, epoch=2, ngram=332792, ppl=303.099
[INFO] elapsed=406.0s, step=1400, epoch=2, ngram=358392, ppl=301.046
[INFO] elapsed=437.2s, step=1500, epoch=2, ngram=383992, ppl=291.934
[INFO] elapsed=463.7s, step=1600, epoch=2, ngram=409592, ppl=286.525
[INFO] elapsed=492.6s, step=1700, epoch=2, ngram=435192, ppl=284.960
[INFO] elapsed=517.5s, step=1800, epoch=2, ngram=460792, ppl=279.750
[INFO] ppl=235.81
[INFO] test finished (took 30.5s, sentence=1359 and ngram=30192)
[INFO] elapsed=574.8s, step=1900, epoch=3, ngram=486384, ppl=247.589
[INFO] elapsed=604.7s, step=2000, epoch=3, ngram=511984, ppl=230.753
[INFO] elapsed=616.4s, step=2100, epoch=3, ngram=537584, ppl=225.742
[INFO] elapsed=617.6s, step=2200, epoch=3, ngram=563184, ppl=217.470
[INFO] elapsed=618.7s, step=2300, epoch=3, ngram=588784, ppl=218.544
[INFO] elapsed=619.8s, step=2400, epoch=3, ngram=614384, ppl=212.152
[INFO] elapsed=620.9s, step=2500, epoch=3, ngram=639984, ppl=209.804
[INFO] elapsed=621.9s, step=2600, epoch=3, ngram=665584, ppl=209.728
[INFO] elapsed=623.0s, step=2700, epoch=3, ngram=691184, ppl=206.931
[INFO] ppl=214.71
[INFO] test finished (took 1.3s, sentence=1359 and ngram=30192)
[INFO] elapsed=625.4s, step=2800, epoch=4, ngram=716776, ppl=193.364
[INFO] elapsed=626.5s, step=2900, epoch=4, ngram=742376, ppl=180.370
[INFO] elapsed=627.6s, step=3000, epoch=4, ngram=767976, ppl=178.496
[INFO] elapsed=628.7s, step=3100, epoch=4, ngram=793576, ppl=171.427
[INFO] elapsed=629.8s, step=3200, epoch=4, ngram=819176, ppl=172.551
[INFO] elapsed=630.8s, step=3300, epoch=4, ngram=844776, ppl=167.811
[INFO] elapsed=631.9s, step=3400, epoch=4, ngram=870376, ppl=166.077
[INFO] elapsed=633.0s, step=3500, epoch=4, ngram=895976, ppl=166.226
[INFO] elapsed=634.1s, step=3600, epoch=4, ngram=921576, ppl=164.077
[INFO] ppl=205.67
[INFO] test finished (took 1.3s, sentence=1359 and ngram=30192)
[INFO] elapsed=636.5s, step=3700, epoch=5, ngram=947168, ppl=158.486
[INFO] elapsed=637.6s, step=3800, epoch=5, ngram=972768, ppl=145.785
[INFO] elapsed=663.7s, step=3900, epoch=5, ngram=998368, ppl=146.168
[INFO] elapsed=689.9s, step=4000, epoch=5, ngram=1023968, ppl=138.918
[INFO] elapsed=717.9s, step=4100, epoch=5, ngram=1049568, ppl=140.360
[INFO] elapsed=744.8s, step=4200, epoch=5, ngram=1075168, ppl=138.440
[INFO] elapsed=764.1s, step=4300, epoch=5, ngram=1100768, ppl=136.087
[INFO] elapsed=789.5s, step=4400, epoch=5, ngram=1126368, ppl=136.540
[INFO] elapsed=815.7s, step=4500, epoch=5, ngram=1151968, ppl=135.015
[INFO] ppl=203.60
[INFO] test finished (took 36.9s, sentence=1359 and ngram=30192)
[INFO] elapsed=865.1s, step=4550, epoch=5, ngram=1164760, ppl=134.665
[INFO] training finished (took 865.1s, step=4550 and epoch=5)
[INFO] model saved
[INFO] model loaded
[INFO] ppl=203.60
[INFO] test finished (took 33.9s, sentence=1359 and ngram=30192)
```
