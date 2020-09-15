# vrec_pp
МИФИ Проектная Практика 2020 "Голосовая идентификация студентов"

**07.09-13.09**  
# *Вторая неделя*  

Что ещё сделать в будущем:  
- Установить Mendeley Web Importer

Что сделано:
1. создан этот репозиторий;
2. установлен pytorch-kaldi (установлены anaconda, kaldi, pytorch);  
  
![alt-текст](https://github.com/Shennor/vrec_pp/blob/master/1.png "pytorch-kaldi установлен!)")  
  
  ```console
  If you really want to use python 3.8.3 as default, add an empty file /home/hupu/kaldi/tools/python/.use_default_python and run this script again.  
  extras/check_dependencies.sh: all OK.
  ```
3. найдены некоторые готовые датасеты:  
(!) https://archive.org/details/daps_dataset (20 докладчиков (10 женщин и 10 мужчин), каждый из которых читает по 5 отрывков из общедоступных книг (что обеспечивает около 14 минут данных на каждого выступающего))  
http://kahlan.eps.surrey.ac.uk/savee/Database.html (4 актера мужского пола в 7 различных эмоциях, всего 480 высказываний на британском английском)  
https://www.kaggle.com/mozillaorg/common-voice  
https://www.kaggle.com/primaryobjects/voicegender  
ещё ссылки на датасеты тут, можно и поискать ещё:  
https://github.com/jim-schwoebel/voice_datasets
4. прочитано 60/400 "Глубокое обучение на Python" Франсуа Шолле  


Решения проблем:  
*если добавил путь в PATH, перезапусти терминал, прежде чем страдать*  
*программы или инструменты, оказывается, недостаточно скачать, их нужно ещё и установить*  

Ссылки на источники:  
[anaconda](https://anaconda.org/anaconda/python)  
[pytorch](https://pytorch.org/)  
[pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi)  
[kaldi](https://kaldi-asr.org)  

[Шпаргалка по Markdown](http://bustep.ru/markdown/shpargalka-po-markdown.html)

**14.09-20.09**  
# *Третья неделя*  

Цели:  
1) протестировать нейросеть на данных TIMIT
2) понять, что она делает
3) дочитать "Глубокое обучение на Python" Франсуа Шолле  

Ссылки на полезную информацию и источники:  
[TIMIT dataset](https://github.com/philipperemy/timit)  
