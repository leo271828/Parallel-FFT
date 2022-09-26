# Parallel-Fast Fourier Transform
Use OMP to implement parallel-FFT 

這是我在修**快速計算法**這門課所做的期末專題報告，老師上課其實就已經教了基本的 FFT 的寫法，但是主要是以遞迴來寫，因此我先把遞迴型的FFT改成迭代型，接著才開始進行平行化處理

## Process
### 1. Iterative
把遞迴型 FFT 拆成迭代型，其實就是把迴圈展開，但還是需要能把 Bit-reverse 搞定的辦法

### 2. Bit reverse
如果是以 2 為次方數的話，那就很簡單了，但這裡我想以 $2^p3^q5^r$ 為主下去實作，但理論上這裡沒有成功

### 3. Parallel in loop
從這裡才開始計時，利用 `<omp.h>` 來把迴圈能加速的部分都加速
> 注意：Threads 並非越高越好，8 緒 & 4 緒其實執行時間會一樣，因為那只是把他每一條多分兩條出來，實際上並沒有增加
## Conclusion
結果而言還可以，在基數非常大的時候，大概到 $2^{24}$ 的時候就會有極為明顯的差距出現了

* 左圖：FFT vs PFFT
* 右圖：迭代型FFT vs 迭代型PFFT
![image](https://user-images.githubusercontent.com/47287400/192219060-9d5363ea-1fd0-4332-829a-a15e43d5c5f8.png)
> 圖是用 Python 來做的，檔案也附在 `Plot_FFT.py`
