# Методика сравнения скорости и памяти моделей обнаружения табака

Дата: **21 ноября 2026 года**. Объект: реализация в
[`fps_measure.ipynb`](../../manual_notebooks/fps_measure.ipynb) и
[`scripts/benchmark`](../../manual_notebooks/scripts/benchmark/README.md), схема
результатов `schema_version=3` на указанную дату. В версии 4 каждый режим и каждая
входная частота измеряются один раз для пары модель/устройство. Актуальный порядок
измерений описан в [протоколе](../../manual_notebooks/scripts/benchmark/README.md).

Измеритель предназначен для ресурсного скрининга конкретных сочетаний модели,
весов, программного адаптера и устройства. Он оценивает последовательное
предсказание при batch=1, поведение под синтетической потоковой нагрузкой и
потребление памяти после прогрева. Раздельные проходы времени и памяти,
повторные запуски в новых процессах и фиксация входов уменьшают несколько
источников систематического смещения. Эти меры ещё не устанавливают
репрезентативность изображений или статистическую значимость различий.

**Статус: методика реализована и прошла программные проверки и короткую CPU-проверку
на реальной модели; итоговое сравнительное исследование для статьи не выполнено.**
Далее отдельно указаны фактическое поведение программы, текущие настройки
инженерного пилота и требования к будущему эксперименту. Готовые формулировки
для англоязычной статьи приведены в разделе 13.

## 1 Постановка и границы выводов

Основная задача проекта — получить маску сорняков по цепочке
**детекция табака → защитные области → HSV-маска растительности → маска сорняков**.
Текущий измеритель охватывает вызов адаптера модели до получения нормализованных
предсказаний на CPU. Защитные области, HSV и итоговая маска сорняков в этом
измерении не вычисляются. Поэтому результаты такого запуска следует называть
*adapter-level inference measurements*, а результаты будущего полного пути —
*full-pipeline measurements*. Чистое время нейросетевого forward также является
другой областью измерения.

Проверяемые ресурсные вопросы:

1. Какова последовательная пропускная способность выбранной конфигурации на
   конкретном устройстве при batch=1?
2. Как меняются задержка завершённых кадров и число заменённых кадров при
   заданной скорости поступления изображений?
3. Каков наблюдаемый объём RAM процесса и памяти аллокатора PyTorch CUDA после
   прогрева при одинаковом наборе изображений?
4. Насколько эти показатели воспроизводятся при новых запусках процесса?

Выбор системы для статьи дополнительно требует Weed IoU, Weed precision/recall,
Crop→weed error и потерь сорняков внутри защитных областей. Быстрое выполнение
само по себе не доказывает пригодность модели для конечной задачи.
Рамочный mAP остаётся диагностическим показателем. Измеритель не оценивает
качество реального видео, робототехнику, наведение или физическую сохранность
растений. Границы исследования определены в [плане статьи](../../notes/todo.md).

## 2 Методологические основания

Работа [Reddi et al., MLPerf Inference Benchmark, arXiv v2](https://arxiv.org/abs/1911.02549v2)
рассматривает производительность вместе с требованиями к качеству и сценарием
поступления запросов. В разделах III.B–D и VII.B различаются задержка,
пропускная способность и ограничения на хвост распределения задержек.
Это обосновывает необходимость указывать сценарий рядом с числом FPS.
Наши direct и stream — собственные режимы: они не воспроизводят правила
MLPerf, его LoadGen, объёмы запросов или процедуру сертификации. В частности,
последовательный batch=1 direct не является MLPerf Offline.

[Kalibera и Jones, Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/)
различают уровни повторения эксперимента и показывают, почему повторение
итераций внутри одного исполнения не заменяет повторных исполнений.
Раздел 6 отдельно рассматривает устранение инициализации и статистическую
независимость наблюдений. В нашей реализации новый процесс ограничивает
перенос состояния аллокаторов, но независимость от теплового состояния машины
не гарантируется. Модель выбора числа повторов и доверительные интервалы
из этой работы в программе не реализованы; значение `REPEATS=3` является
начальной инженерной настройкой, а не заимствованной статистической нормой.

Официальный [PyTorch benchmark guide](https://docs.pytorch.org/docs/2.15/benchmark_utils.html)
поддерживает использование прогрева, синхронизации ускорителя, контроля потоков
и повторных измерений. Здесь применяется собственный runner, а не
`torch.utils.benchmark.Timer`. Это позволяет сохранить общий интерфейс адаптеров,
исследовать очередь синтетического потока и раздельно собирать память.

PDF обеих научных работ находятся в [articles](articles/); версии, SHA-256,
официальные страницы и результаты проверки записаны в [list.csv](list.csv).
Статьи цитируются через [list.bib](list.bib), программная документация — через
[benchmark_software.bib](benchmark_software.bib). Дата проверки источников —
21.11.2026. Документация PyTorch 2.15 не подменяет фактическую версию 2.14.1
в локальном CPU smoke test.

PDF сохранены локально и исключены из Git общим правилом `*.pdf`. Для переноса
архива доступны точные URL загрузки и контрольные суммы из `list.csv`;
одного клонирования репозитория для получения этих файлов недостаточно.

## 3 Единица сравнения и текущая конфигурация

Единицу сравнения удобно обозначать

$$
c=(m,w,b,d,p,s,\theta,\mathcal I,e),
$$

где $m$ — модель, $w$ — checkpoint с контрольной суммой,
$b$ — backend и его версия, $d$ — устройство, $p$ — численная точность,
$s$ — политика размера и преобразования входа, $\theta$ — параметры
предсказания, $\mathcal I$ — упорядоченный набор изображений,
$e$ — программное и аппаратное окружение. Один идентификатор модели
не определяет все компоненты $c$.

Текущие значения notebook являются настройками следующего запуска, а не
параметрами уже проведённого основного исследования.

| Parameter | Notebook value | Interpretation |
| --- | --- | --- |
| Input list | `data/tobacco_migrated/train.txt` | Provisional engineering pool |
| Image count | 24 | Sampling without replacement from the local 25-image list |
| Image seed | 20261028 | Reproducible selection and order |
| Batch size | 1 | One image per adapter call |
| Device | `cpu` | CUDA is configured explicitly as `cuda:N` |
| Warm-up frames | 30 | At least one visit to every selected image |
| Timing duration | 30 s | Target window; final inference may finish later |
| Process repeats | 3 | Fresh worker for every configuration and repeat |
| Memory passes | 2 | 48 direct calls for 24 selected images |
| RSS interval | 0.01 s | Requested interval, not a real-time guarantee |
| Stream input rate | 30 FPS | Requested producer rate |
| Job-order seed | 20261028 | Shuffling within each repeat |
| PyTorch threads | 1 intra-op, 1 inter-op | Does not constrain every third-party library |
| Ultralytics options | `imgsz=640`, `rect=True`, `conf=0.25` | Configured size does not establish actual tensor shape |
| RF-DETR options | `conf=0.25` | Resolution is restored from checkpoint |

`BenchmarkConfig` в Python API сохраняет другие значения по умолчанию:
`repeats=1`, `memory_passes=0`, `torch_num_threads=None`. Расширенный протокол
включается notebook явно. При описании эксперимента следует читать сохранённую
конфигурацию конкретного запуска, а не опираться на значения по умолчанию API.

### 3.1 Происхождение и представительность изображений

Локальные списки мигрированного датасета содержат 25/25/2458 записей для
train/validation/test. [План проекта](../../notes/todo.md) прямо относит такое
разбиение к истории программной проверки. Выбор 24 изображений из его train
расширяет нагрузку относительно одной фотографии, но не делает её научно
репрезентативной. Автоматической проверки кампаний, освещения, плотности растений,
перекрытия кадров и доверенности разметки в `select_images` нет.

Для основного эксперимента требуется заранее определённый ресурсный набор
из допустимых train/validation-источников, с проверенным происхождением и
покрытием условий. Его нельзя формировать по тому, какая модель на нём быстрее.
Финальный test следует сохранить для зафиксированной конфигурации по общему
протоколу проекта. Исторические результаты на 110 изображениях не являются
baseline нового исследования.

Все модели получают одинаковый список и порядок. Однако одинаковый список
ещё не означает одинаковое число обработок каждого кадра: окно FPS ограничено
временем, а stream дополнительно заменяет непрочитанные кадры. Только отдельный
проход памяти гарантирует одинаковое число посещений всех изображений.

## 4 Порядок выполнения одного измерения

Для каждой комбинации модели, устройства, режима, входной частоты и номера
повтора запускается новый Python-процесс. Задания выполняются последовательно;
внутри каждого повтора их порядок перемешивается с сохранённым seed. Direct и
stream в notebook вызываются разными ячейками и обычно создают разные каталоги;
между этими двумя вызовами общей рандомизации нет.

Последовательность внутри worker:

1. Импорт окружения, выбор устройства, установка потоков, проверка контрольных
   сумм и чтение изображений в память как BGR-массивов.
2. Регистрация RSS до загрузки модели и начальной статистики CUDA.
3. Ленивое создание модели через адаптер и прогрев.
4. Проверка числа intra-op потоков; если SDK изменил его при загрузке,
   восстановление заданного значения и повтор полного прогрева.
5. Фиксированный проход памяти, если `memory_passes > 0`.
6. Новый сброс CUDA peak counters и временное окно direct либо stream.
7. Расчёт сводок, сохранение сырых наблюдений и метаданных; завершение процесса.

Число вызовов одного прохода прогрева равно

$$
N_{\mathrm{warm}}=\max(W,|\mathcal I|).
$$

Это обеспечивает посещение каждого изображения, но не является автоматическим
тестом выхода на тепловое или статистическое стационарное состояние. Фиксированный
проход памяти дополнительно прогревает модель перед окном FPS. При сравнении
следует сохранять одинаковый порядок стадий и одинаковую настройку memory passes.

После прогрева, прохода памяти и окна FPS проверяется заданное число intra-op
потоков. Обнаруженное несовпадение завершает измерение ошибкой. Проверки выполняются
на границах стадий, а не непрерывно. В manifest записывается фактическое число
потоков после измерения. Потоки OpenCV, BLAS, affinity, частоты CPU/GPU и режим
питания этим механизмом не фиксируются.

## 5 Время предсказания и пропускная способность

### 5.1 Границы времени

Для кадра $i$ обозначим начало вызова адаптера через $a_i$, момент завершения
вызова и последующей синхронизации устройства — через $b_i$. В программе
используется `time.perf_counter`, то есть прошедшее астрономическое время,
включающее ожидания и вытеснение процесса. Это не счётчик CPU time.
Семантика часов описана в [документации Python](https://docs.python.org/3/library/time.html#time.perf_counter).

$$
L_i^{\mathrm{pred}}=1000(b_i-a_i)\quad[\mathrm{ms}].
$$

Измеренный вызов включает подготовку входа адаптером/SDK, выполнение модели,
постобработку и перевод результатов в общий формат на CPU. На CUDA
синхронизация выполняется до начала таймера и после вызова; последняя входит
в задержку. Освобождение возвращённого объекта и запись скалярного наблюдения
выполняются после $b_i$, поэтому входят в общее время цикла, но не в
`prediction_ms` этого кадра. Чтение и декодирование файлов исключены заранее.

### 5.2 Пропускная способность окна

Пусть $t_0,t_1$ — сохранённые границы окна, $T=t_1-t_0$,
$N$ — число завершённых предсказаний. Тогда

$$
F=\frac{N}{T}\quad[\mathrm{frames/s}].
$$

Вызов, пересёкший заданную длительность, завершается и учитывается; знаменатель
содержит фактическое время. В direct сюда входят затраты цикла и записи
наблюдений; в stream также ожидание следующего кадра. Поэтому `processing_fps`
не следует заменять на $1000/\operatorname{median}(L_i^{\mathrm{pred}})$.

Внутри одного запуска сохраняются среднее, медиана, p95 и p99 задержки
(`numpy.percentile`, используемая по умолчанию линейная интерполяция).
Это эмпирические квантили завершённых вызовов, а не гарантии соблюдения дедлайна.
Особенно p99 требует достаточного числа наблюдений и проверки устойчивости хвоста.

График оконного FPS делит число завершений на реальную ширину каждого окна,
включая последнее неполное. Мгновенный ряд использует
$1/(b_i-b_{i-1})$. Среднее такого ряда не является оценкой $N/T$.
IQR-фильтрация в notebook предназначена для просмотра графиков: она не удаляет
наблюдения из исходных файлов и не меняет сводные результаты.

### 5.3 Синтетический поток

Производитель циклически копирует изображения с заданной частотой. Между ним
и потребителем находится один слот: новый кадр заменяет старый, если тот ещё
не считан. Буфер поэтому не растёт без ограничения. Источник не моделирует
кодек, камеру или движение сцены.

Если $g_i$ — время перед копированием кадра производителем, то

$$
L_i^{\mathrm{frame}}=1000(b_i-g_i).
$$

Эта задержка включает копирование, ожидание в слоте, планирование потоков и
предсказание. Она вычисляется только для завершённых кадров: малая задержка
может сочетаться с большим числом замен, поэтому необходим совместный отчёт.
Для приращений счётчиков генерации $G$ и замен $D$ внутри окна программа
сохраняет

$$
F_{\mathrm{gen}}=G/T,\qquad R_{\mathrm{drop}}=D/T.
$$

**Поле `drop_rate` имеет единицу frames/s, а не проценты.** Доля $D/G$
сейчас отдельно не сохраняется; при её дополнительном расчёте нужно назвать
знаменатель и учесть границы окна. Начальный или конечный незавершённый кадр
не позволяет автоматически считать $G=N+D$. Фактический `generation_fps`
следует проверять вместе с заданной частотой: производитель также зависит от
планировщика и стоимости копирования.

При достаточной скорости моделей поток с подачей 30 FPS обычно ограничивает
наблюдаемую пропускную способность примерно этой частотой. Поэтому запас
скорости ранжируется по direct, а stream характеризует работу при конкретной
нагрузке. Конечное короткое окно может иметь граничные отклонения от 30 FPS.

## 6 Измерение памяти

### 6.1 Почему память измеряется отдельным проходом

Во временном окне накапливаются словари кадров и RSS-отсчёты. Число записей
кадров растёт с числом завершений: при одинаковой длительности быстрая модель
создаёт больше записей. Упрощённо, собственная память этих журналов растёт как

$$
O(N+K),\qquad N\approx FT,\quad K\approx T/\Delta t.
$$

Это свойство текущей реализации, а не характеристика нейросети. Уменьшение
интервала RSS увеличивает число записей и нагрузку sampler, но не гарантирует
обнаружение каждого краткого пика.

Отдельный `measure_memory` после прогрева выполняет

$$
N_{\mathrm{probe}}=P|\mathcal I|
$$

предсказаний: $P$ полных проходов по всем изображениям в одинаковом порядке.
Возвращённый объект удаляется после каждого вызова; per-frame записи не
создаются. Sampler хранит только первый, последний и максимальный отсчёты,
их количество и максимальный наблюдённый временной промежуток. Количество
сохраняемых объектов измерителя ограничено и не растёт с FPS или числом кадров.
Это уменьшает вмешательство измерителя, но не делает его строго нулевым.

Probe всегда работает как direct, даже если за ним идёт stream. Следовательно,
`probe_*` описывает фиксированную нагрузку без очереди, а поля памяти без
префикса — временное окно с накоплением наблюдений и, для stream, с очередью.
Сравнивать или объединять эти области как один показатель нельзя.

### 6.2 Память процесса

`psutil.Process().memory_info().rss` возвращает резидентную физическую память
текущего worker; на Windows ей соответствует working set. Это включает
совместно используемые страницы и не тождественно private memory.
[Документация psutil](https://psutil.readthedocs.io/stable/#psutil.Process.memory_info)
определяет также USS/PSS, которые здесь не собираются.

Для RSS-функции $R(t)$ и фактических моментов опроса $\tau_j$
сохраняемый максимум имеет вид

$$
\widehat R_{\max}=\max_j R(\tau_j),\qquad
\widehat R_{\max}\leq\sup_{t\in\mathcal W}R(t),
$$

где $\mathcal W$ охватывает интервал между граничными отсчётами sampler.
Неравенство описывает дискретное наблюдение одной и той же величины RSS;
оно не превращает RSS в оценку всей физической памяти системы.
Запрошенные 10 мс не равны гарантированному фактическому шагу: GIL,
планировщик и стоимость опроса могут увеличить промежутки.

| Field | Scope and meaning |
| --- | --- |
| `rss_baseline_bytes` | Worker after PyTorch import and image loading, before model loading |
| `probe_rss_warm_bytes` | First RSS sample before the fixed-workload memory pass |
| `probe_rss_sampled_peak_bytes` | Maximum RSS sample during that pass, including boundary samples |
| `rss_warm_bytes` | First sample before the timed window, after any memory probe |
| `rss_sampled_peak_bytes` | Maximum sample around the timed window, including logging and stream buffers |
| `memory_probe.sample_count` | Number of RSS observations in the separate pass |
| `memory_probe.max_sample_gap_s` | Largest actual interval between those observations |

RSS включает Python, библиотеки, исходные изображения, модель, временные буферы
и служебные объекты. Память notebook и отдельных дочерних процессов SDK не
суммируется с RSS worker. Предварительное чтение всех изображений сохраняет
I/O вне таймера, но увеличивает RAM; изменение числа и разрешения фотографий
изменяет измеряемую систему.

Разность `probe_rss_warm_bytes − rss_baseline_bytes` характеризует изменение
памяти между двумя стадиями, а не точный размер модели. Между стадиями также
импортируются SDK и создаются рабочие буферы. Для эксплуатационного сравнения
нужно сохранять абсолютный footprint; вычитание baseline не должно заменять
его и не должно использоваться как доказательство объёма весов в RAM.

У probe нет полного временного ряда RSS: сохранились только агрегаты и три
опорных отсчёта. `memory_records` содержит ряд отдельного окна FPS. По нему
нельзя восстановить пропущенные пики probe или очистить его оценку задним числом.

### 6.3 Память CUDA

Для выбранного устройства используются текущие и максимальные значения
`memory_allocated`, `memory_reserved`, `max_memory_allocated`,
`max_memory_reserved`. Перед каждым из двух измерительных окон выполняются
синхронизация и `reset_peak_memory_stats`.

Allocated относится к памяти, занятой тензорами, reserved — к памяти под
управлением кэширующего аллокатора. Они перекрываются; их нельзя складывать.
Сброс peak counters не освобождает cache: рабочий footprint после прогрева
остаётся включённым. Статистика относится к аллокатору PyTorch, а не ко всей
VRAM процесса или устройства. Контекст CUDA и сторонние аллокаторы могут
добавлять расходы. Значимость отдельных показателей зависит и от backend
аллокатора; его необходимо фиксировать при окончательных CUDA-экспериментах.
[PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-memory-management)

Поля `probe_cuda_*_bytes_warm/peak` относятся к фиксированному проходу;
`cuda_*_bytes_warm/peak` без префикса — к окну FPS. Пиковые показатели являются
абсолютными максимумами после сброса, а не приростами относительно warm.
На CPU они равны `null`, а не нулю.

Пик загрузки checkpoint и холодного старта не измеряется. Поэтому эти результаты
не определяют минимальную память, достаточную для запуска приложения от начала
до конца. На Jetson нельзя складывать RSS и CUDA-память как независимые физические
пулы; проверка общей памяти и сбор системной телеметрии требуют отдельного
протокола. Такой протокол в notebook пока не реализован.

Все сырые объёмы записываются в байтах; в таблицах применяется
$1\ \mathrm{MiB}=2^{20}$ байт. Размер checkpoint на диске — отдельная величина.

## 7 Повторы и статистическое представление

Для одной конфигурации $c$ и режима сначала вычисляется сводка каждого
завершённого повтора $r$: $F_{c,r}$, квантили задержки и пики памяти.
Затем `comparison_table` вычисляет по этим сводкам

$$
\widetilde x_c=\operatorname{median}_{r=1}^{n_c}x_{c,r},\qquad
\mathrm{IQR}_c=Q_{0.75}(x_{c,r})-Q_{0.25}(x_{c,r}),
$$

а также minimum и maximum. Каждый повтор имеет одинаковый вес независимо
от того, сколько кадров успела обработать модель. Например, медиана значений
p95 отдельных запусков — это **median of per-run p95**, а не p95 всех кадров,
объединённых в один массив. В таблице статьи нужно сохранять это различие.

`n` — число доступных завершённых повторов с конкретной метрикой; оно может
быть меньше заданного `REPEATS` после сбоя. Отсутствующие CUDA-метрики CPU
не заменяются нулями. Завершённые результаты прерванного запуска доступны
для диагностики, но нельзя молча считать его полной сбалансированной серией.

Диапазон min–max и IQR являются описательными характеристиками. Они не являются
доверительными интервалами и не дают критерия значимости по одному факту
пересечения диапазонов. Три запуска полезны для первого обнаружения разброса,
но не гарантируют достаточную точность оценки p95, памяти или небольшого speedup.
Большое число повторений одних и тех же фотографий не увеличивает число
независимых сцен и не устраняет возможную автокорреляцию времён.

Для итогового исследования следует заранее установить целевую точность и
бюджет, исследовать межзапусковый разброс и зависимость временных рядов, затем
обосновать длительность и число повторов. Если нужны интервалы для различий
моделей, анализ должен учитывать единицу повторения и дизайн эксперимента.
Для качества масок единицей неопределённости служат независимые поля/кампании,
а не тысячи коррелированных тайлов. Эти расчёты не входят в текущую реализацию.

Группировка в `comparison_table` использует модель, устройство, режим и
входную частоту. Она не проверяет равенство checkpoint, изображений, точности
и окружения между произвольно объединёнными каталогами. Основная область
применения — один каталог `run_benchmarks` с общей конфигурацией. При объединении
экспериментов нужно сначала сверить манифесты и разделить разные настройки.

## 8 Угрозы валидности и остаточные ограничения

| Issue | Implemented control | Remaining limitation |
| --- | --- | --- |
| Repeated single photograph | Explicit shared image list and reproducible selection | Current 24-image pool has no verified coverage of field conditions |
| More timing records for faster models | Separate memory pass with bounded bookkeeping | Timing-window RSS still includes records |
| Unequal visits during memory comparison | Every image is processed exactly P times | Slower models expose RSS to more time samples; missed-peak bias can remain |
| State from previous models or rates | Fresh process for each job | OS caches, background load and thermal state persist |
| Run order | Seeded shuffling inside repeat blocks | Separate notebook calls have separate schedules |
| Lazy initialization | Warm-up visits every selected input | No automatic stationarity or temperature test |
| SDK changes thread count | Restore, rewarm and check stage boundaries | Other library threads and transient internal changes are not continuously monitored |
| Different input policies | Save options and available runtime size/precision | Actual input tensor shapes are not captured; RF-DETR precision is unknown |
| Producer saturation | Record actual generation FPS and replacements | Synthetic producer is not a camera or a real-video decoder |
| RAM peak | Boundary samples and sampling-gap diagnostics | Brief spikes may be missed; startup peak excluded |
| GPU memory | Separate allocated/reserved statistics | No total-process VRAM, NVML telemetry or hardware CUDA validation of this update |
| Observer overhead | Progress UI runs in parent; memory probe avoids growing logs | Sampling threads and timing records still consume resources |
| Model quality | Same adapter boundary and explicit checkpoints | Benchmark does not evaluate weed masks or validate exports |

Даже при одном и том же `imgsz=640` прямоугольное дополнение изображения,
квадратное изменение размера и ограничения checkpoint дают разные нагрузки.
Сравнение таких конфигураций полезно для выбора практической системы, если
качество проверяется одновременно. Оно не изолирует влияние архитектуры.
Не следует молча менять разрешение или precision для получения желаемого ранга.

Windows поддерживается используемыми переносимыми механизмами запуска и RSS,
но обновлённая реализация на native Windows не проверялась. Её результаты
следует вести отдельно от Linux. MPS runner не поддерживает. Desktop CUDA
не заменяет измерений на Jetson.

## 9 Протокол окончательного эксперимента

Следующая последовательность описывает требуемую работу, а не уже полученные
результаты.

1. Завершить проверку происхождения данных и независимого split; подготовить
   ресурсную подвыборку, покрывающую обычные и сложные условия.
2. Для каждого кандидата зафиксировать checkpoint, backend, разрешение,
   precision, параметры выдачи предсказаний и режим оптимизации. Отдельно
   определить, сравниваются ли одинаковые вычислительные условия или лучшие
   практически пригодные конфигурации при заданных ограничениях качества.
3. На каждой аппаратной платформе записать характеристики, версии драйверов,
   режим питания, ограничения частот и фоновые процессы. Выполнить пилот
   длительности и повторов; проверить временные ряды и фактическую подачу кадров.
4. Зафиксировать выбранный протокол до основного сравнения. Выполнить direct
   и релевантные stream-режимы, проверить завершённость всех повторов и
   достаточность RSS-отсчётов. Не удалять неудобные выбросы из основного отчёта.
5. После обучения короткого списка повторить измерения на финальных весах:
   число предсказаний и стоимость постобработки могли измениться. Проверить
   качество каждого экспортированного или квантованного варианта отдельно.
6. Реализовать и измерить полный путь до маски сорняков как самостоятельную
   область измерения. Не получать его скорость суммированием отдельных медиан:
   общие буферы, преобразования и порядок операций могут отличаться.
7. Сопоставить ресурсы с Weed IoU/precision/recall, Crop→weed error и потерями
   сорняков в защите на validation. Зафиксировать выбранные конфигурации до
   итоговой test-оценки. Публиковать компромиссы, а не универсального победителя.

Минимальная длительность одних временных окон равна

$$
T_{\mathrm{windows}}=R\,M\,D\,S\,T,
$$

где $M$ — число моделей, $D$ — устройств, $S$ — число режимов/частот,
$R$ — повторов. Для текущих трёх повторов direct и одного stream это 180 с
на модель/устройство, без загрузок, прогрева и проходов памяти. Это нижняя
оценка календарного времени, а не оценка всего исследования. GPU-время
следует учитывать суммарно по устройствам в пределах бюджета 72 GPU-часа
из [плана проекта](../../notes/todo.md), включая другие стадии работы.

## 10 Воспроизводимость и сохраняемые артефакты

Корневой manifest схемы 3 хранит конфигурацию, упорядоченные пути изображений,
их контрольные суммы, контрольные суммы checkpoint, состав моделей, задания
`jobs`, номер повтора, порядок исполнения и завершённые worker. Worker получает
один режим/частоту, проверяет входы, записывает окружение и собственный manifest.

Для завершённого окна сохраняются `run-*.json`, скалярные времена каждого кадра,
RSS-ряд окна FPS, `memory_probe`, сводки JSON/CSV и лог. Пути можно менять между
машинами, но содержимое входов и весов должно оставаться тем же. Контрольные
суммы обнаруживают изменение файла; сами файлы они не архивируют.

| Recorded automatically | Must be fixed or added for the final experiment |
| --- | --- |
| Exact selected image order, hashes and source shapes | Field/campaign provenance and coverage rationale |
| Checkpoint hash and provider versions | Training-run identity and model-selection rationale |
| Job-order seed and explicit jobs | Image-selection recipe and source list version; `IMAGE_SEED` is not a separate manifest field |
| Python, platform, package versions, CUDA/cuDNN versions | Full hardware inventory, driver version, power state and clock policy |
| PyTorch threads and selected math flags | Other library threads, CPU affinity and allocator configuration |
| Available runtime size, rect and precision | Observed input tensor dimensions and unreported precision details |
| Completed runs and failure status | Source-code commit or snapshot hash and environment lock/export |

В текущем рабочем дереве есть незакоммиченные изменения. Схема `3` сама по себе
не идентифицирует точную реализацию; перед публикационным экспериментом нужно
сохранить commit либо снимок исходников. Сведения `RUN_NOTES` полезны, но не
заменяют машинно читаемую фиксацию окружения.

Чтение старых JSON поддерживается. Отсутствующие `probe_*` или `repeat` в старом
запуске нельзя восстановить из одного пика RSS или приписать ему задним числом.
Сохранённый вывод notebook также не доказывает, что он получен текущими ячейками:
использовать следует исходные JSON и manifest соответствующего запуска.

## 11 Проверки реализации

На 21.11.2026 после обновления измерителя зафиксированы следующие проверки.

| Check | Evidence | What it establishes |
| --- | --- | --- |
| CPU unit tests with mocked providers | 79 tests passed | Scheduling, memory bookkeeping, aggregation and adapter contracts |
| Notebook validation | JSON and code-cell syntax checked; existing outputs preserved | Structural correctness and preservation of earlier outputs |
| Real-model smoke test | YOLO26n on Linux CPU, two images, direct and stream 30 FPS, two repeats | Four fresh workers completed; probes, summaries, metadata and plots were readable |
| CUDA API behavior | Mock tests | Correct calls and field handling; no hardware measurement claim |
| Native Windows, updated RF-DETR path, Jetson | No new hardware validation in this update | Compatibility remains to be checked on these targets |

Команда программной проверки:

```bash
.venv-notebooks/bin/python -m unittest discover -s manual_notebooks/tests -v
```

Последний подтверждённый CPU smoke test использовал Python 3.14.7, PyTorch
2.14.1, Ultralytics 8.4.191, NumPy 2.5.4 и psutil 7.2.2. Оба изображения имели
исходную форму 352×480×3; SDK сообщал configured size 640×640, `rect=True`,
FP32 и один intra-op поток. Это не доказательство формы входного тензора 640×640.
Было задано два кадра прогрева, один проход памяти и окно 0.2 с, поэтому
числа этого запуска нельзя использовать как оценки производительности для статьи.

Временные артефакты проверки:
`/tmp/tobacco-benchmark-smoke/20261121T115630Z-401955d3`;
журнал автономных тестов: `/tmp/benchmark-tests.log`. Эти пути не являются
постоянным архивом и могут исчезнуть. Статус проверки записан здесь для
разграничения выполненной валидации и будущего эксперимента, без переноса
smoke-значений FPS или RAM в сравнительную таблицу.

## 12 Таблицы и рисунки для статьи

Результаты разных устройств и разных областей измерения следует показывать
отдельно. Строка таблицы соответствует полной конфигурации; рядом приводится
число завершённых повторов. Для памяти предпочтителен отдельный direct probe;
stream-память временного окна можно добавить отдельной колонкой с явной подписью.

Шаблон таблицы производительности:

| Configuration | Device | Input policy | Precision | n | Direct FPS median [Q1, Q3] | Prediction p95 median across runs, ms |
| --- | --- | --- | --- | --- | --- | --- |
| To be measured | To be recorded | To be verified | To be verified | — | — | — |

Шаблон таблицы памяти:

| Configuration | Scope | Sampled peak RSS median / maximum, MiB | Peak CUDA allocated median / maximum, MiB | Peak CUDA reserved median / maximum, MiB | RSS sample count and maximum gap |
| --- | --- | --- | --- | --- | --- |
| To be measured | Fixed-workload direct probe | — | — | — | — |

Шаблон потокового результата:

| Configuration | Requested / actual input FPS | Processing FPS | Frame p95 median across runs, ms | Replaced frames/s | Completed repeats |
| --- | --- | --- | --- | --- | --- |
| To be measured | — | — | — | — | — |

Для итоговой статьи нужны: отдельные точки повторов с медианой и разбросом;
временные ряды для обнаружения дрейфа; графики качества итоговой маски против
FPS и памяти; примеры ошибок на независимых сценах. Диапазоны на ресурсных
графиках следует подписывать как наблюдаемый min–max, если именно его показывает
программа. Они не должны обозначаться как 95% CI.

## 13 Английские формулировки для Methods

Следующий текст описывает реализованный протокол в настоящем времени.
Обозначения `R`, `T`, `W`, `P`, `K` следует заменить значениями из окончательных
манифестов; инженерные значения 3/30/30/2/24 не являются автоматически
параметрами публикационного эксперимента. Утверждения о проведённом исследовании
можно переводить в прошедшее время только после его выполнения.

### Measurement scope

> Computational performance is measured at the prediction-adapter boundary using
> batch size one. Each call includes provider-specific input preparation, model
> execution, postprocessing, and conversion to a common CPU-based output format.
> Input images are decoded before measurement. Model loading, file decoding, and
> visualization are excluded from the timed window. These measurements characterize
> the prediction implementation; the protection-region and HSV stages require a
> separate full-pipeline evaluation.

### Experimental execution

> Each model, device, operating mode, input rate, and repetition is executed in a
> fresh Python worker process. The same ordered set of K images and checkpoint
> identities is maintained through file checksums. Job order is shuffled within
> repetition blocks using a recorded seed. Warm-up comprises at least W predictions
> and visits every selected image. The requested PyTorch thread count is restored
> after provider initialization if necessary, followed by another warm-up. Process
> isolation limits allocator-state carryover but does not establish independence
> from machine-wide thermal or scheduling effects.

### Timing and stream behavior

> Prediction latency is measured with a wall-clock performance counter. CUDA
> execution is synchronized before and after each prediction. Throughput is the
> number of completed predictions divided by the actual measurement-window
> duration, with a target duration of T seconds. A separate synthetic-stream mode
> supplies copies of the selected photographs at a specified rate and retains only
> the latest unconsumed frame. Frame latency starts before the producer copies the
> image and ends after prediction completion. Actual input rate and replaced frames
> per second are reported alongside latency; this workload does not establish
> performance or segmentation quality on real video.

### Memory measurement

> After warm-up, memory is evaluated in a separate sequential pass that processes
> every selected image P times. This pass retains neither per-frame timing records
> nor an expanding memory-sample history. Process resident memory is sampled at a
> requested interval of 10 ms, and the sample count and largest observed sampling
> gap are retained. The reported RAM peak is therefore a sampled process-RSS peak,
> rather than an exact model-only footprint. On CUDA devices, peak allocated and
> reserved memory are obtained separately from the PyTorch allocator after resetting
> its peak counters. These quantities are not added together and do not represent
> total device VRAM. Loading and cold-start peaks are outside this measurement scope.

### Aggregation and interpretation

> Performance summaries are first computed for each completed run and then
> aggregated across R process repetitions with equal weight per run. We report
> medians, quartiles, and observed ranges. A reported median of per-run p95 values
> is distinguished from a pooled frame-level p95. These descriptive ranges are not
> confidence intervals. Resource comparisons are interpreted jointly with the
> final weed-mask quality metrics and with explicitly reported input-resolution,
> numerical-precision, runtime, and device settings.

Для обоснования дизайна подходят ключи `reddi2020mlperfinference` и
`kalibera2013rigorous`; для API — `pytorch2026benchmark`,
`pytorch2026cudamemory`, `psutil2026memory`, `python2026perfclock`.
Текст не заявляет соответствия MLPerf или реализации статистического метода
Kalibera и Jones.

## 14 Карта реализации и источников

| Component | Implementation |
| --- | --- |
| Configuration, input selection, job order | [core.py](../../manual_notebooks/scripts/benchmark/core.py): `BenchmarkConfig`, `select_images`, `make_jobs` |
| Warm-up, timing and fixed memory pass | [core.py](../../manual_notebooks/scripts/benchmark/core.py): `measure_run`, `measure_memory`, `MemorySampler`, `DeviceMetrics` |
| Environment, hashes and worker lifecycle | [worker.py](../../manual_notebooks/scripts/benchmark/worker.py): `run_worker` |
| Per-run and cross-run tables | [plots.py](../../manual_notebooks/scripts/benchmark/plots.py): `summary_table`, `comparison_table`, `configuration_table` |
| Time-series slicing and exploratory filtering | [measurement.py](../../manual_notebooks/scripts/benchmark/measurement.py) |
| Provider-specific runtime metadata | [Ultralytics adapter](../../manual_notebooks/scripts/backends/ultralytics.py), [RF-DETR adapter](../../manual_notebooks/scripts/backends/rfdetr.py) |
| Synthetic checks | [test_benchmark.py](../../manual_notebooks/tests/test_benchmark.py), [test_measurement.py](../../manual_notebooks/tests/test_measurement.py) |

Версия Reddi et al. в архиве — **arXiv v2 от 09.05.2020**, хотя первая подача
была в 2019 году. Запись `reddi2020mlperfinference` описывает именно эту ревизию;
выходные данные ISCA в неё не подмешаны. У Kalibera и Jones официальный
репозиторий указывает, что размещённая авторская версия исправляет версию
ISMM 2013. Это различие сохранено в BibTeX и CSV. Математические определения
собственных полей измерителя и анализ его ограничений выше основаны на коде
проекта; выбранные числа кадров, проходов и повторов не приписываются этим работам.
