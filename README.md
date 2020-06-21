# Style-Transfer-homework
Домашнее задание по Style Transfering курсов по машинному обучению Deep Learning School 

Домашнее задание посвящено переносу сразу двух стилей на одно изображение без повтороного обучения(т.е. с реализацией двух стилей в одной модели). Для решения задачи использовалась простая идея видоизменения функции потерь на 

$	\mathcal L_{total} = \alpha \mathcal L_{content}(input,content) + \beta \mathcal L_{style_1}(input,style_1) + \gamma \mathcal L_{style_2}(input,style_2)$

