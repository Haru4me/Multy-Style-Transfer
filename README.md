# Style-Transfer-homework
Домашнее задание по Style Transfering курсов по машинному обучению Deep Learning School 

Домашнее задание посвящено переносу сразу двух стилей на одно изображение без повтороного обучения(т.е. с реализацией двух стилей в одной модели). Для решения задачи использовалась простая идея видоизменения функции потерь на 


<img src="https://latex.codecogs.com/svg.latex? \mathcal{L}_{total}=\alpha\mathcal{L}_{content}(input,content)+\beta\mathcal{L}_{style_1}(input,style_1)+\gamma\mathcal{L}_{style_2}(input,style_2)"/>

