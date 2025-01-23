# ЧЕКПОИНТ

В качестве оптимизатора я выбрал SGD. Размер батча в даталоадерах 64

Сначала я взял тупо блок из 1 домашки и он дал качество что-то около 20% на валидации на 15 эпохах

Затем я добавил в сеть второй блок. Получил следующее:

**My Net**
* SGD(lr=0.1, momentum=0.9)
* batch_size = 64
* 15 эпох

**Резы**
* Train: 44.8%
* Val: 23.5%
* Test: 20.7%

[<img src="image-4.png" width="600" />](image-4.png)


# Дальнейшие эксперименты

Затем я решил добавить аугментацию RandomHorizontalFlip и начал эксперименты с ResNet

**ResNet18**
* RandomHorizontalFlip(0.5)
* SGD(lr=0.01, momentum=0.9)
* batch_size = 64
* 20 эпох

**Резы**
* Train: -
* Val: -
* Test: 28%

Прочерки потому что не помню результатов

Затем я посмотрел на архитектуру ResNet18 и увидел что там в самом начале используется MaxPooling. Для больших картинок он очевидно нужен, но для маленьких - только мешает. Я его убрал

Помимо этого там есть свёртки с stride = 2. Для маленьких картинок как у нас - это не есть хорошо. Я сделал stride = 1 в первых трёх свёртках, где stride равнялся 2

Помимо этого я решил поиграться с SGD, накинув туда ещё параметров

**My ResNet18**
* RandomHorizontalFlip(0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 64
* 10 эпох

**Резы**
* Train: 44%
* Val: 37%
* Test: 34.8%

[<img src="image-3.png" width="600" />](image-3.png)


Потом решил поэксперементировать с efficient net. Взял efficientnet_v2_s на 10 эпох и сделал stride = 1 в первых двух свёрточных слоях со stride = 2

Помимо этого я добавил scheduler CosineAnnealingLR(T_max = 15). Сделал это потому, что заметил что на поздних эпохах возникает переобучение, попытался так с ним бороться, плавно уменьшая lr

**My efficientnet_v2_s**
* RandomHorizontalFlip(0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 64
* CosineAnnealingLR(T_max = 15)
* 15 эпох

**Резы**
* Train: 51.9%
* Val: 39.8%
* Test: 37.56%

[<img src="image-2.png" width="600" />](image-2.png)

Затем я решил изменить dropout перед линейным слоем с 0.2 на 0.4 (чтобы снизить переобучение) и увеличить количество эпох до 20. А так же добавил ещё одну аугментацию - RandomSolarize

**My efficientnet_v2_s**
* Аугментации:
    * RandomHorizontalFlip(0.5)
    * RandomSolarize(0.4, 0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 64
* CosineAnnealingLR(T_max = 20)
* dropout = 0.4
* 20 эпох

**Резы**
* Train: 55%
* Val: 41.8%
* Test: 39.6%

[<img src="image-1.png" width="600" />](image-1.png)

Почти 40%!!!

Я решил сделать batch_size = 32

**My efficientnet_v2_s**
* Аугментации:
    * RandomHorizontalFlip(0.5)
    * RandomSolarize(0.4, 0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 20)
* dropout = 0.4
* 20 эпох

**Резы**
* Train: 59.8%
* Val: 43%
* Test: 40.5%

[<img src="image.png" width="600" />](image.png)

Победа!!!

Дальше я посмотрел на динамику изменения lr и качества на трейне и валидации. Я увидел, что CosineAnnealingLR плохо справляется со своей задачей и не успевает достаточно понизить lr к 11-й эпохе (тогда начинается переобучение). Я решил попробовать поставить параметр T_max = 10. Но от этого стало только хуже

**My efficientnet_v2_s**
* Аугментации:
    * RandomHorizontalFlip(0.5)
    * RandomSolarize(0.4, 0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 10)
* dropout = 0.4
* 20 эпох

**Резы**
* Train: 37.6%
* Val: 36.5%
* Test: -

[<img src="image-5.png" width="600" />](image-5.png)


Я решил попробовать MultuStepLr. Я увидел, что на 11-й эпохе наступает переобучение. Я попробую уменьшить на 11-й эпохе lr в 10 раз и посмотреть что будет

**My efficientnet_v2_s**
* Аугментации:
    * RandomHorizontalFlip(0.5)
    * RandomSolarize(0.4, 0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* MultiStepLR([10], gamma=0.1)
* dropout = 0.4
* 20 эпох

**Резы**
* Train: 52%
* Val: 42.3%
* Test: 39%

[<img src="image-6.png" width="600" />](image-6.png)

Как видим на 11-й эпохе когда понижаем лр то получаем хороший скачок качества вверх. Я попробовал ещё поиграться с эпохой на которой понижаем лр. Но всё равно это приводило к переобучению. В итоге все-таки решил оставить понижение lr на 11-й эпохе, потому что мне почему-то нравится этот скачок качества

Я решил попробовать заменить аугментацию RandomSolarize(0.4, 0.5) на RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6). Она смещает картинку на 1/8 от размера. Также я решил попробовать уменьшить lr на 12-й эпохе, вдруг это тоже даст скачок качества

Чтобы узнать качество на тесте я решил попробовать сделать Test-Time augmentation. Я прогонял одну и ту же картинку 5 раз с аугментациями и выбирал класс к которому картинка оносится чаще всего

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* MultiStepLR([10, 11], gamma=0.1)
* dropout = 0.4
* 20 эпох

**Резы**
* Train: 42.3%
* Val: 40.6%
* Test: 37.6%

[<img src="image-7.png" width="600" />](image-7.png)

Мда, выходим на плато. Надо что-то менять

Я решил вернуться к модели, которая давала 43% на валидации. Я решил сделать в ней такую же замену аугментации и использовать Test-Time augmentation

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 20)
* dropout = 0.4
* 20 эпох

**Резы**
* Train: 54.7%
* Val: 45%
* Test (with test-time aug): 41.8
* Test: 41.48

[<img src="image-8.png" width="600" />](image-8.png)

Как мы видим переобучение стало меньше, при этом качество на валидации и тесте стало выше. При этом можно увидеть, что test-time augmentation действительно помогает

Я решил попробовать сделать dropout=0.6 и поставить 30 эпох обучения

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 30)
* dropout = 0.6
* 30 эпох

**Резы**
* Train: 64.2%
* Val: 47.3%
* Test (with test-time aug): 44.6% 

[<img src="image-9.png" width="600" />](image-9.png)


Хочу попробовать поменять нормализацию. Я сейчас не учитываю распределение пикселей, просто считаю что они равномерно распределены от 0 до 256, но это не так. По идее разумнее взять дисперсию и среднее

means = (0.5669, 0.5426, 0.4914)
stds = (0.2377, 0.2326, 0.2506)

Вот такие значения у меня получились, код можно посмотреть в ноутбуке, он закомменчен


**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 30)
* dropout = 0.6
* 30 эпох

**Резы**
* Train: 64.5%
* Val: 47.2%
* Test: -

[<img src="image-10.png" width="600" />](image-10.png)

Несмотря на то, что качество на валидации чуть ниже, была эпоха (29-я) когда качество было выше чем в предыдущем эксперименте

[<img src="image-11.png" width="600" />](image-11.png)


Добавим аугментацию поворотом картинки на 10 градусов 

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
  * RandomApply([RandomAffine(degrees=10)], p=0.6)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 30)
* dropout = 0.6
* 30 эпох

**Резы**
* Train: 60.8%
* Val: 47.4%
* Test: -


[<img src="image-12.png" width="600" />](image-12.png)

Качество выросло! Добавим еще одну аугментацию - гауссовый блюр

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
  * RandomApply([RandomAffine(degrees=10)], p=0.6),
  * RandomApply([GaussianBlur(3)], p=0.3)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 30)
* dropout = 0.6
* 30 эпох

**Резы**
* Train: 57%
* Val: 46.3%
* Test: -

[<img src="image-13.png" width="600" />](image-13.png)

Переобучение уменьшилось, но качество на валидации тоже. Попробуем зафиксировать sigma=0.6

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
  * RandomApply([RandomAffine(degrees=10)], p=0.6),
  * RandomApply([GaussianBlur(3, 0.6)], p=0.3)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 30)
* dropout = 0.6
* 30 эпох

**Резы**
* Train: 58.2%
* Val: 46.7%
* Test: -

Качество выросло!

[<img src="image-14.png" width="600" />](image-14.png)

Попробую поставить 40 эпох

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
  * RandomApply([RandomAffine(degrees=10)], p=0.6),
  * RandomApply([GaussianBlur(3, 0.6)], p=0.3)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 40)
* dropout = 0.6
* 40 эпох

**Резы**
* Train: 69.7%
* Val: 47.2%
* Test: 44.96%

[<img src="image-15.png" width="600" />](image-15.png)

Качество на тесте выросло, но это потому, что я в test-time augmentation прогонял картинку 10 раз, а не 5

Как мы видим аугментация сделала только хуже и ничего не смогло ей помочь сделать что-то лучше

Попробуем запустить на 40 эпох поставив dropout=0.7

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.6)
  * RandomApply([RandomAffine(degrees=10)], p=0.6)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 40)
* dropout = 0.7
* 40 эпох

**Резы**
* Train: 67.8%
* Val: 47.7%
* Test: 45.74%

[<img src="image-16.png" width="600" />](image-16.png)

Попробуем увеличить вероятность применения аугментаций, это должно помочь уменьшить переобучение

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.8)
  * RandomApply([RandomAffine(degrees=10)], p=0.8)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 40)
* dropout = 0.7
* 40 эпох

**Резы**
* Train: 65.8%
* Val: 47.7%
* Test: -

[<img src="image-19.png" width="600" />](image-19.png)

Как мы видим на валидации качество осталось неизменным, а вот переобучение чуть упало - победа

Попробуем использовать другой scheduler. Меня заинтересовал ReduceLROnPlateau. Как только аккураси на валидации выходит на плато - будем уменьшать LR в 10 раз

В голове идея звучало классно, но на деле оказалась фигней: я провел несколько запусков и в каждом шедулер очень быстро по сути занулял lr, что не давало модели обучаться

Я решил дать еще один шанс MultiStepLR

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.8)
  * RandomApply([RandomAffine(degrees=10)], p=0.8)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* MultiStepLR([20, 30])
* dropout = 0.7
* 40 эпох

**Резы**
* Train: 54.3%
* Val: 46.8%
* Test: -

[<img src="image-20.png" width="600" />](image-20.png)

Все же нет, плохая была это идея

Почитав статьи я наткнулся на ShakeDrop. Я решил попробовать поставить его куда-то в центр эфишентнета и посмотреть что получится

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.8)
  * RandomApply([RandomAffine(degrees=10)], p=0.8)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* CosineAnnealingLR(T_max = 40)
* ShakeDrop(0.5)
* dropout = 0.4
* 40 эпох

**Резы**
* Train: 76.1%
* Val: 49.4%
* Test: -

[<img src="image-21.png" width="600" />](image-21.png)

Качество на валидации выросло!

Попроуем увеличить количество эпох до 100 и сравнить 2 шедулера - CosineAnnealingLR(T_max = 100) и MultiStepLR([50,75])

CosineAnnealingLR:

[<img src="image-22.png" width="600" />](image-22.png)

MultiStepLR: (Test: 47.06%)

[<img src="image-23.png" width="600" />](image-23.png)

На середине обучения лучше всех себя показал MultiStepLR. Он выдал качество на валидации 51,3%

[<img src="image-24.png" width="600" />](image-24.png)

Попробуем уменьшить количество эпох и увеличить dropout


**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.8)
  * RandomApply([RandomAffine(degrees=10)], p=0.8)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* MultiStepLR([30, 45])
* ShakeDrop(0.5)
* dropout = 0.8
* 60 эпох

**Резы**
* Train: 65.2%
* Val: 52%
* Test: 48.4%

[<img src="image-25.png" width="600" />](image-25.png)

На разных этапах обучения я сохранял веса модели, что позволило мне протестировать не только финальный вариант

Например вот 46 эпоха обучения:

[<img src="image-26.png" width="600" />](image-26.png)

Я сохранил веса и обнаружил, что на ней качество на тесте составляет 49.06%

Это лучший результат который у меня был. Получаем итог:

**My efficientnet_v2_s**
* Аугментации:
  * RandomHorizontalFlip(0.5)
  * RandomApply([RandomAffine(degrees=0, translate=(1/8,1/8))], p=0.8)
  * RandomApply([RandomAffine(degrees=10)], p=0.8)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* batch_size = 32
* MultiStepLR([30, 45])
* ShakeDrop(0.5)
* dropout = 0.8
* 46 эпох

**Резы**
* Train: 65.2%
* Val: 52%
* Test: 49.06%