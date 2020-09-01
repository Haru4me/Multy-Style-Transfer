# <center>Multy Style Transfer


Задание посвящено переносу сразу двух стилей на одно изображение без повтороного обучения(т.е. с реализацией двух стилей в одной модели). Для решения задачи использовалась простая идея видоизменения функции потерь на 

<img src="https://latex.codecogs.com/svg.latex?\centering&space;\mathcal{L}_{total}=\alpha\mathcal{L}_{content}(input,content)+\beta\mathcal{L}_{style_1}(input,style_1)+\gamma\mathcal{L}_{style_2}(input,style_2)"/>

Получившийся результат представлен ниже
<p align="center">
    <img src="/results/Unknown-4.png" width="480"\>
</p>

<p align="center">
    <img src="/results/Unknown-5.png" width="480"\>
</p>

Изначально в ноутбуке перенос стиля применяется к второму случаю
