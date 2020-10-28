# vrec_pp
МИФИ Проектная Практика 2020 "Голосовая идентификация студентов"

**08.09-15.09**  
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

**15.09-22.09**  
# *Третья неделя*  

Цели:  
1) протестировать нейросеть на данных TIMIT
2) понять, что она делает!!!
3) дочитать "Глубокое обучение на Python" Франсуа Шолле  
4) узнать о MFCC (Mel-Frequency Cepstral Coefficients)  
5) Long-Term Spectral Divergence  

Датасеты на русском языке:  
https://github.com/snakers4/open_stt/  
https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/  

Варианты нейросетей:  
https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system  

Ссылки на полезную информацию и источники:  
[Librosa](https://github.com/librosa/librosa)  
[Преобразовать WAV в MFCC](https://github.com/dspavankumar/compute-mfcc)  
[TIMIT dataset](https://github.com/philipperemy/timit)  

Проблемы при подготовки TIMIT для pytorch-kaldi:  
1) qsub:  
![alt-текст](https://github.com/Shennor/vrec_pp/blob/master/2.jpg "")  

Найти код, который:
1) делает звуковые дорожки одинаковыми по размеру  
2) переводит их в mfcc  
3) содержит нейросеть  

**22.09-29.09**  
# *Четвёртая неделя*  

[Проект Deep-Speaker](https://github.com/philipperemy/deep-speaker)
 который работает так:  
 1) принимает формат flac, преобразует герцы в мелы и нормализует до 1  
 2) для обучения используется cosine simitiarity (косинусный коэфициент) - определяется косинус "угла" между предсказанным и ожидаемым векторами, чем ближе полученное значение к 1, тем меньше угол и, соответственно, больше точность предсказания  
 
 ### Пробные тесты обученной нейосети ResCNN_softmax_pre_training_checkpoint_102.h5 с моим голосом:  
 1) cosine similiarity мой голос - русск (номера - записи отрывков из книг, SAME SPEAKER - значение между указанными номерами, DIFF SPEAKER - между номером и записью другого человека (samples/PhilippeRemy/PhilippeRemy_002.wav)):  
 - 1 и 3 SAME SPEAKER [0.58378375] (плоховато)  
 - 1 DIFF SPEAKER [-0.13367558]  
 - 3 и 4 SAME SPEAKER [0.3899359] (плохо)  
 - 3 DIFF SPEAKER [0.020014]
 - 4 и 5 SAME SPEAKER [0.6028304]  
 - 5 DIFF SPEAKER [0.01057245]
 - 6 и 5 SAME SPEAKER [0.31917924] (плохо)  
 - 6 DIFF SPEAKER [0.01787219]
 - 2 и 5 SAME SPEAKER [0.5059043] (плоховато)  
 - 2 DIFF SPEAKER [-0.00309112]  
 2) cosine similiarity голос автора - англ:  
 SAME SPEAKER [0.7715848]  
 DIFF SPEAKER [-0.12826478]  
 3) cosine similiarity англ LibriSpeech
  женский женский мужской  
  samples/LibriSpeechSamples/19/198/19-198-0001.wav  &  samples/LibriSpeechSamples/19/198/19-198-0022.wav  
  SAME SPEAKER [0.7663208]  
  samples/LibriSpeechSamples/19/198/19-198-0001.wav  &  samples/PhilippeRemy/PhilippeRemy_001.wav  
  DIFF SPEAKER [-0.12552892]  
  samples/LibriSpeechSamples/19/198/19-198-0001.wav  &  samples/LibriSpeechSamples/19/227/19-227-0016.wav  
  SAME SPEAKER [0.62854576]  
  samples/LibriSpeechSamples/19/198/19-198-0001.wav  &  samples/PhilippeRemy/PhilippeRemy_001.wav  
  DIFF SPEAKER [-0.12552892] 
  мужской мужской мужской  
  samples/LibriSpeechSamples/26/496/26-496-0004.wav  &  samples/LibriSpeechSamples/26/496/26-496-0026.wav  
  SAME SPEAKER [0.70082587]  
  samples/LibriSpeechSamples/26/496/26-496-0004.wav  &  samples/PhilippeRemy/PhilippeRemy_001.wav  
  DIFF SPEAKER [0.21883845]  
  samples/LibriSpeechSamples/26/495/26-495-0019.wav  &  samples/LibriSpeechSamples/26/496/26-496-0026.wav  
  SAME SPEAKER [0.51492935]  
  samples/LibriSpeechSamples/26/495/26-495-0019.wav  &  samples/PhilippeRemy/PhilippeRemy_003.wav  
  DIFF SPEAKER [0.10637974]  

[Диаризация на основе модели GMM-UBM и алгоритма MAP adaptation] (https://m.habr.com/ru/post/420515/)  

## Способы идентификации человека с помощью MFCC  
1) вычисление Евклидова расстояния между коэфициентами (чем ближе к 0, тем более похожи голоса)  
```np.sum((x - y)**2) #x и y - np.sum от вектора MFCC```
2) вычисление косинусного коэфициента между векторами коэфициентов (чем ближе к 1, тем более похожи голоса)  

[Статья про машинный слух на Хакере](https://xakep.ru/issues/xa/245/)  

## Какими должны быть обучающие данные для deep-speaker  
Данные необходимо предоставить в папке, каждый файл имеет путь вида:  
```# 'audio/dev-other/116/288045/116-288045-0000.flac'```  
где -3-й элемент - идентификатор говорящего, а последний элемент - идентификатор высказывания  

## Откуда брать аудио для датасета

[Бесплатные любительские аудиокниги в открытом доступе](https://prochtu.ru/)  
(спросила администратора сайта про авторские права на аудиофайлы и можно ли их использовать)  


# Шестая неделя  

1) Попытка обучения на Colab:  
https://colab.research.google.com/drive/1GPQ7H0FO21JLASm9HOSz6EpRlsrVfBNA?usp=sharing  
Датасет для обучения Voxforge+audiobook (ссылка от автора):  
https://github.com/vlomme/Multi-Tacotron-Voice-Cloning  
Результаты за выделяемые сисемой 12 часов:  
```
2020-10-09 04:25:23.081216: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Using TensorFlow backend.
2020-10-09 04:25:26,438 - INFO - Training with the triplet loss.
2020-10-09 04:25:26.525584: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-10-09 04:25:26.592431: E tensorflow/stream_executor/cuda/cuda_driver.cc:314] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2020-10-09 04:25:26.592511: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (b068085c1773): /proc/driver/nvidia/version does not exist
2020-10-09 04:25:26.626756: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2300000000 Hz
2020-10-09 04:25:26.627025: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7b068c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-10-09 04:25:26.627060: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-10-09 04:25:33,551 - INFO - Picking audio from /content/drive/My Drive/Colab Notebooks/deep_speaker/samples/book.
Initializing the batcher: 100% 4/4 [02:40<00:00, 40.01s/it]
Build test set: 100% 200/200 [02:54<00:00,  1.14it/s]

Epoch 1/1000
1099/2000 [===============>..............] - ETA: 9:44:45 - loss: 0.6841
```
2) Сбор датасета некоторых студентов:  
https://github.com/Shennor/vrec_pp/tree/master/samples/student_samples  

# Седьмая неделя

Обучение на Colab GPU:  
```
Epoch 88/1065
22176/22176 [==============================] - 51s 2ms/sample - loss: 0.3241 - accuracy: 0.9991 - val_loss: 0.9072 - val_accuracy: 0.9482
```
Результаты на тестах:  
```
2020-10-15 15:49:17,555 - INFO - f-measure = 0.776, true positive rate = 0.703, accuracy = 0.996, equal error rate = 0.101
```

# Восьмая неделя  

![alt-текст](https://github.com/Shennor/vrec_pp/blob/master/app_architecrure.png)  



https://github.com/ppwwyyxx/speaker-recognition
