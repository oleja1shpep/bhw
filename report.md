# ЧЕКПОИНТ

В качестве оптимизатора я выбрал SGD. Размер батча в даталоадерах 64

Сначала я взял тупо блок из 1 домашки и он дал качество что-то около 20% на валидации на 15 эпохах

Затем я добавил в сеть второй блок. Получил следующее:

**My Net**
* SGD(lr=0.1, momentum=0.9)
* 15 эпох

Train: 44.8%
Val: 23.5%
Test: 20.7%


# Дальнейшие эксперименты

Затем я решил добавить аугментацию RandomHorizontalFlip и начал эксперименты с ResNet

**ResNet18**
* RandomHorizontalFlip(0.5)
* SGD(lr=0.01, momentum=0.9)
* 20 эпох

Train: -
Val: -
Test: 28%

Прочерки потому что не помню результатов

Затем я посмотрел на архитектуру ResNet18 и увидел что там в самом начале используется MaxPooling. Для больших картинок он очевидно нужен, но для маленьких - только мешает. Я его убрал

Помимо этого там есть свёртки с stride = 2. Для маленьких картинок как у нас - это не есть хорошо. Я сделал stride = 1 в первых трёх свёртках, где stride равнялся 2

Помимо этого я решил поиграться с SGD, накинув туда ещё параметров

**My ResNet18**
* RandomHorizontalFlip(0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* 10 эпох

Train: 44%
Val: 37%
Test: 34.8%

Потом решил поэксперементировать с efficient net. Взял efficientnet_v2_s на 10 эпох и сделал stride = 1 в первых двух свёрточных слоях со stride = 2

Помимо этого я добавил scheduler CosineAnnealingLR(T_max = 15). Сделал это потому, что заметил что на поздних эпохах возникает переобучение, попытался так с ним бороться, плавно уменьшая lr

**My efficientnet_v2_s**
* RandomHorizontalFlip(0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* CosineAnnealingLR(T_max = 15)
* 15 эпох

Train: 51.9%
Val: 39.8%
Test: 37.56%

Затем я решил изменить dropout перед линейным слоем с 0.2 на 0.4 (чтобы снизить переобучение) и увеличить количество эпох до 20. А так же добавил ещё одну аугментацию - RandomSolarize

**My efficientnet_v2_s**
* RandomHorizontalFlip(0.5), RandomSolarize(0.4, 0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* CosineAnnealingLR(T_max = 20)
* 20 эпох

Train: 55%
Val: 41.8%
Test: 39.6%

Почти 40%!!!

Я решил сделать batch_size = 32

**My efficientnet_v2_s**
* RandomHorizontalFlip(0.5), RandomSolarize(0.4, 0.5)
* SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
* CosineAnnealingLR(T_max = 20)
* 20 эпох

Train: 59.8%
Val: 43%
Test: 40.5%

