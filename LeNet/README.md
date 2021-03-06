# LeNet-5

## Result
- ### FashionMNIST
    - Parameters
        ```
        Epoch = 50
        BatchSize = 32
        LearingRate = 0.001 
        ```
    - Result
        ```
        train_acc = 97.492%
         test_acc = 88.520%
        ```
    - Graph  
    <img src="./pic/fm_acc_exp1.png" width = "320"/>
    <img src="./pic/fm_loss_exp1.png" width = "320"/>


- ### CIFAR-10  
    Firstly, I want to say that `LeNet`'s performance on `CIFAR-10` is **NOT** really good. The highest accuracy I get is `63%`. By the way , some guys got 67% accuracy on their LeNet. 🤪
1. Exp1
    - Parameters
        ```
        Epoch = 50
        BatchSize = 32
        LearingRate = 0.001
        ```
    - Result
        ```
        train_acc = 81%
         test_acc = 59%
        ```
    - Graph   
    <img src="./pic/cf_acc_exp1.jpg" width = "320"/>
    <img src="./pic/cf_loss_exp1.jpg" width = "320"/>  
      

    > 🤨Bad performance on test dataset. Maybe **overfitting**.  
    > There are some ways that could be used to improve the performance:
    >   1. Change param.  
    > eg. Let `Learning Rate` become `dynamic Learning Rate` ; Increase `Epoch` and ...  
    >   2. Improve our Net.   
    eg. `Regularization` ; `Batch Normalization` and ...  
    >   
    > Going to show some results of these ways on next experiments.

2. Exp2  
    Change `Learning Rate` to `dynamic Learning Rate`.  
    - Parameters
        ```
        Epoch = 50
        BatchSize = 32
        LearingRate = 0.001   (Epoch<=15)
                    = 0.0002  (15<Epoch<=30)
                    = 0.00004 (30<Epoch<=40)
                    = 0.000008(40<Epoch<=50)
        ```
    - Result *(Better performence on test dataset, but lower train_acc)*
        ```
        train_acc = 76.462%
         test_acc = 63.250%
        ```
    - Graph   
    <img src="./pic/cf_acc_exp2.png" width = "320"/>
    <img src="./pic/cf_loss_exp2.png" width = "320"/>

    > 🙄It seems that `dynamic Learing Rate` accelerates the decline of `loss`, just like what the pic shows (on about 23000th step, there is a **sharp** decline). 

3. Exp3  
    Change param such as `BatchSize` & `Epoch`.  
    In this case, I think more `Epoch` and bigger `BatchSize` are meaningless, but I still make a experiment about bigger `BatchSize`.
    - Parameters
        ```
        Epoch = 50
        BatchSize = 100
        LearingRate = 0.001
        ```
    - Result
        ```
        train_acc = 79.656%
         test_acc = 62.580%
        ```
    - Graph  
    <img src="./pic/cf_acc_exp3.png" width = "320"/>
    <img src="./pic/cf_loss_exp3.png" width = "320"/>

4. Exp4  
    After watching many kinds of training, I try to replace `ReLu` with `LeakyReLu`, which improve the accuracy of train dataset into `91%` but still bad performance on test dataset. 🙄Just for fun.
    - Parameters *(Same as Exp1, only replace `ReLu` with `Leaky ReLu`)*
    - Result
        ```
        train_acc = 91.682% (about 10% improvement than Exp1)
         test_acc = 58.120%
        ```
    - Graph  
    <img src="./pic/leaky_relu_acc.png" width = "320"/>
    <img src="./pic/leaky_relu_loss.png" width= "320"/>






        
