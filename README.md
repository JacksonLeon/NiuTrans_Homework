# NiuTrans_Homework

课堂练习：[基于NiuTensor的FNNLM](https://gitee.com/JacksonLeon/NiuTrans-Homework.git)

通过简单调参，能够将ppl从231.69降到210.17
训练过程如下：
```
args:
 -dev=3
 -lrate=0.006180
 -wbatch=256
 -minmax=0.100000
 -nepoch=6
 -n=3
 -hdepth=1
 -hsize=128
 -esize=100
 -train=data/wsj.train
 -test=data/wsj.test
 -output=work/wsj.prob
 -vsize=10000
 -model=work/wsj.model
 -autodiff=true
[INFO] elapsed=16.9s, step=100, epoch=1, ngram=25600, ppl=1325.025
[INFO] elapsed=35.0s, step=200, epoch=1, ngram=51200, ppl=995.210
[INFO] elapsed=57.4s, step=300, epoch=1, ngram=76800, ppl=847.983
[INFO] elapsed=76.4s, step=400, epoch=1, ngram=102400, ppl=742.279
[INFO] elapsed=99.5s, step=500, epoch=1, ngram=128000, ppl=680.781
[INFO] elapsed=119.1s, step=600, epoch=1, ngram=153600, ppl=633.062
[INFO] elapsed=136.8s, step=700, epoch=1, ngram=179200, ppl=591.041
[INFO] elapsed=157.6s, step=800, epoch=1, ngram=204800, ppl=565.659
[INFO] elapsed=173.3s, step=900, epoch=1, ngram=230400, ppl=542.913
[INFO] ppl=310.13
[INFO] test finished (took 42.0s, sentence=2000 and ngram=49560)
[INFO] elapsed=232.8s, step=1000, epoch=2, ngram=255768, ppl=375.837
[INFO] elapsed=251.9s, step=1100, epoch=2, ngram=281368, ppl=338.102
[INFO] elapsed=270.8s, step=1200, epoch=2, ngram=306968, ppl=328.650
[INFO] elapsed=291.0s, step=1300, epoch=2, ngram=332568, ppl=323.085
[INFO] elapsed=309.4s, step=1400, epoch=2, ngram=358168, ppl=313.553
[INFO] elapsed=326.3s, step=1500, epoch=2, ngram=383768, ppl=311.641
[INFO] elapsed=342.8s, step=1600, epoch=2, ngram=409368, ppl=304.638
[INFO] elapsed=360.9s, step=1700, epoch=2, ngram=434968, ppl=297.285
[INFO] elapsed=380.5s, step=1800, epoch=2, ngram=460568, ppl=295.798
[INFO] elapsed=394.9s, step=1900, epoch=2, ngram=486168, ppl=292.425
[INFO] ppl=252.30
[INFO] test finished (took 44.1s, sentence=2000 and ngram=49560)
[INFO] elapsed=456.8s, step=2000, epoch=3, ngram=511536, ppl=261.260
[INFO] elapsed=475.9s, step=2100, epoch=3, ngram=537136, ppl=247.811
[INFO] elapsed=494.4s, step=2200, epoch=3, ngram=562736, ppl=245.836
[INFO] elapsed=509.8s, step=2300, epoch=3, ngram=588336, ppl=238.028
[INFO] elapsed=528.4s, step=2400, epoch=3, ngram=613936, ppl=234.123
[INFO] elapsed=551.0s, step=2500, epoch=3, ngram=639536, ppl=234.425
[INFO] elapsed=573.0s, step=2600, epoch=3, ngram=665136, ppl=228.984
[INFO] elapsed=592.1s, step=2700, epoch=3, ngram=690736, ppl=226.867
[INFO] elapsed=607.3s, step=2800, epoch=3, ngram=716336, ppl=226.430
[INFO] elapsed=627.2s, step=2900, epoch=3, ngram=741936, ppl=224.015
[INFO] ppl=228.40
[INFO] test finished (took 38.7s, sentence=2000 and ngram=49560)
[INFO] elapsed=684.6s, step=3000, epoch=4, ngram=767304, ppl=205.120
[INFO] elapsed=703.4s, step=3100, epoch=4, ngram=792904, ppl=198.016
[INFO] elapsed=727.5s, step=3200, epoch=4, ngram=818504, ppl=197.783
[INFO] elapsed=748.0s, step=3300, epoch=4, ngram=844104, ppl=193.355
[INFO] elapsed=766.8s, step=3400, epoch=4, ngram=869704, ppl=189.736
[INFO] elapsed=786.8s, step=3500, epoch=4, ngram=895304, ppl=191.304
[INFO] elapsed=806.8s, step=3600, epoch=4, ngram=920904, ppl=186.740
[INFO] elapsed=827.3s, step=3700, epoch=4, ngram=946504, ppl=185.786
[INFO] elapsed=846.1s, step=3800, epoch=4, ngram=972104, ppl=185.845
[INFO] elapsed=866.6s, step=3900, epoch=4, ngram=997704, ppl=184.253
[INFO] ppl=216.74
[INFO] test finished (took 33.0s, sentence=2000 and ngram=49560)
[INFO] elapsed=918.9s, step=4000, epoch=5, ngram=1023072, ppl=172.953
[INFO] elapsed=940.6s, step=4100, epoch=5, ngram=1048672, ppl=169.736
[INFO] elapsed=958.4s, step=4200, epoch=5, ngram=1074272, ppl=168.502
[INFO] elapsed=977.6s, step=4300, epoch=5, ngram=1099872, ppl=162.651
[INFO] elapsed=997.7s, step=4400, epoch=5, ngram=1125472, ppl=159.602
[INFO] elapsed=1017.4s, step=4500, epoch=5, ngram=1151072, ppl=161.656
[INFO] elapsed=1040.3s, step=4600, epoch=5, ngram=1176672, ppl=158.616
[INFO] elapsed=1058.9s, step=4700, epoch=5, ngram=1202272, ppl=157.631
[INFO] elapsed=1080.5s, step=4800, epoch=5, ngram=1227872, ppl=157.747
[INFO] elapsed=1100.1s, step=4900, epoch=5, ngram=1253472, ppl=156.416
[INFO] ppl=211.46
[INFO] test finished (took 42.9s, sentence=2000 and ngram=49560)
[INFO] elapsed=1165.9s, step=5000, epoch=6, ngram=1278840, ppl=150.920
[INFO] elapsed=1186.3s, step=5100, epoch=6, ngram=1304440, ppl=146.644
[INFO] elapsed=1205.9s, step=5200, epoch=6, ngram=1330040, ppl=144.425
[INFO] elapsed=1227.3s, step=5300, epoch=6, ngram=1355640, ppl=139.775
[INFO] elapsed=1240.6s, step=5400, epoch=6, ngram=1381240, ppl=137.645
[INFO] elapsed=1259.0s, step=5500, epoch=6, ngram=1406840, ppl=139.662
[INFO] elapsed=1276.7s, step=5600, epoch=6, ngram=1432440, ppl=136.943
[INFO] elapsed=1290.9s, step=5700, epoch=6, ngram=1458040, ppl=136.630
[INFO] elapsed=1307.3s, step=5800, epoch=6, ngram=1483640, ppl=136.751
[INFO] elapsed=1325.2s, step=5900, epoch=6, ngram=1509240, ppl=135.813
[INFO] ppl=210.17
[INFO] test finished (took 34.5s, sentence=2000 and ngram=49560)
[INFO] elapsed=1366.6s, step=5934, epoch=6, ngram=1517712, ppl=135.530
[INFO] training finished (took 1366.6s, step=5934 and epoch=6)
[INFO] model saved
[INFO] model loaded
[INFO] ppl=210.17
[INFO] test finished (took 42.0s, sentence=2000 and ngram=49560)
```
