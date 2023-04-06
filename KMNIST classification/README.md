My first try to create image classifier using Convolution Neural Networks. I use LeNet architecture

Dataset:
- KMNIST

Used libraries:
- PyTorch
- Numpy
- Matplotlib

Results:

           \  precision    recall  f1-score   support
           o       0.93      0.94      0.94      1000
          ki       0.97      0.94      0.95      1000
          su       0.92      0.89      0.90      1000
         tsu       0.95      0.97      0.96      1000
          na       0.92      0.94      0.93      1000
          ha       0.94      0.94      0.94      1000
          ma       0.91      0.97      0.94      1000
          ya       0.97      0.92      0.94      1000
          re       0.97      0.96      0.96      1000
          wo       0.95      0.96      0.95      1000

    accuracy                           0.94     10000
    macro_avg      0.94      0.94      0.94     10000
    weighted_avg   0.94      0.94      0.94     10000

Traning plot:

![plot](https://user-images.githubusercontent.com/45041977/228349852-58df18cc-4c90-4cca-abd2-2d0acf8ab041.png)

Sample from test set:

![336660826_768577274449528_6826466207788651435_n](https://user-images.githubusercontent.com/45041977/230396047-30d2334c-0a4e-4783-9fbc-389adf363b6c.png)
