# Модели детекции и сегментации растений и других объектов для видео в реальном времени

**Срез литературы: 19 октября 2026 года. Период отбора: 19.10.2022–19.10.2026.**

## 1. Основные выводы

Для распознавания заранее заданных видов растений разумно начинать с компактного обучаемого детектора или instance-сегментатора. В список кандидатов входят Ultralytics YOLOv8/YOLO11/YOLO26, независимые ветви YOLO, RTMDet и современные DETR-модели. Выбор между ними следует делать на собственных растениях и целевом устройстве: положение на COCO не определяет победителя на аграрных данных. Это подтверждают и сравнительные эксперименты на трёх культурах, и проверка переноса между наборами сорняков. [31](#ref-tijjani2026agriaiseg) [32](#ref-zdraveska2026weedbenchmark)

Если необходима сплошная карта «почва / культура / сорняк», следует рассматривать **семантическую сегментацию**, в частности PIDNet и EfficientViT. Если нужны маски отдельных растений — **instance-сегментацию**. Если задача состоит в сопровождении уже указанного объекта по кадрам — **SAM 2**; для поиска и сопровождения объектов по текстовому понятию — **SAM 3**, с отдельной оценкой вычислительных затрат. Эти постановки не взаимозаменяемы. [18](#ref-xu2023pidnet) [19](#ref-cai2023efficientvit) [24](#ref-ravi2024sam2) [25](#ref-carion2026sam3)

**YOLO — семейство работ разных разработчиков, а не последовательность продуктов одного провайдера.** Ultralytics, Meituan, Alibaba, академические группы YOLOv9/v10/v12/v13 и авторы YOLO-World имеют разные реализации и исследовательские задачи. Наличие модели в библиотеке Ultralytics не делает Ultralytics автором её исходной архитектуры. Карта официально поддерживаемых моделей: [Ultralytics Models](https://docs.ultralytics.com/models/).

## 2. Как собран обзор

В корпус включены **32 полных PDF**: архитектурные работы, исследования эффективной сегментации и семь аграрных работ. Все файлы находятся в `articles/`; библиография — в [list.bib](list.bib), реестр — в [list.csv](list.csv). Для каждой статьи проверены официальный ресурс, название и авторы в PDF, а также использованные разделы методов и результатов. Контрольные суммы, версии, даты и заметки проверки сохранены в [source_audit.json](source_audit.json); исключения — в [source_audit.md](source_audit.md).

Это тематический научно-технический обзор, а не исчерпывающий систематический поиск. Источники: arXiv, CVF Open Access, Frontiers, официальные страницы конференций и разработчиков. Поисковые направления: `real-time object detection`, `real-time semantic/instance segmentation`, `crop weed edge detection`, `fruit segmentation`, названия моделей и их новых версий. Поиск англоязычный; доступность полного PDF создаёт смещение в сторону открытых публикаций. Новые модели проверялись отдельно, включая работы 2026 года.

**Правило дат.** Для препринтов основание включения — первая подача в выбранном интервале; для издательских статей — дата публикации. Например, PIDNet и EfficientViT включены по конференционным публикациям 2023 года, даже если ранний препринт появился раньше. Обновление старого препринта само по себе не делает его новой работой. В имени PDF и поле `year` BibTeX указан год **сохранённой версии**; первая дата отдельно записана в реестре. Поэтому сравнение яблок, впервые размещённое в 2024 году, здесь цитируется как версия v4 2026 года. Результаты разных версий не объединялись. [28](#ref-sapkota2026fruitseg)

Статус источника указан в списке литературы. Авторская версия принятой конференционной работы остаётся записью arXiv в BibTeX: её заголовок и страницы не заменены данными другой версии. Для части технических отчётов рецензирование сохранённой версии не подтверждено; это не утверждение об отсутствии любых последующих публикаций. Официальная документация используется только как дополнительный источник о программной поддержке и лицензиях. Все научные результаты ниже опираются на локальные PDF. Собственные обучение и измерения не проводились.

## 3. Что именно должна делать модель

| Задача | Результат | Пример для растений | Ограничение |
|---|---|---|---|
| Детекция | Прямоугольник, класс, уверенность | Найти табак, сорняк, плод | Рамка не задаёт точный контур или точку воздействия |
| Семантическая сегментация | Класс каждого пикселя | Карта культуры, сорняка и почвы | Соседние растения одного класса могут сливаться |
| Instance-сегментация | Отдельная маска каждого экземпляра | Разделить перекрывающиеся растения или плоды | Требует разметки экземпляров; декодирование масок тоже занимает время |
| Сегментация видео / сопровождение | Маска и идентичность по времени | Сопровождать выбранное растение при движении камеры | Нужны инициализация, обработка исчезновения и восстановления объекта |
| Открытый словарь / подсказки | Объекты по тексту, точке, рамке или примеру | Найти растения заданного вида без фиксированного списка классов | Название «сорняк» зависит от культуры и контекста; перенос надо проверять |

Большинство быстрых детекторов обрабатывают кадры независимо. Для подсчёта без повторов к ним добавляют сопровождение: сопоставляют объекты между кадрами и поддерживают ID. Это отдельный компонент, не автоматическое свойство YOLO. В SAM 2 память по времени встроена в модель; исходный SAM предназначен прежде всего для сегментации изображений по подсказкам. [20](#ref-kirillov2023sam) [24](#ref-ravi2024sam2)

## 4. YOLO: версии, авторы и назначение

Год в этой таблице обозначает появление рассматриваемой работы/линейки; точная сохранённая версия указана в библиографии. «Детекция» не означает, что любой checkpoint автоматически поддерживает сегментацию.

| Ветвь | Авторы / экосистема | Что изменено и зачем рассматривать | Что учитывать |
|---|---|---|---|
| YOLOv5; YOLOv7 как контекст | Ultralytics; отдельная ветвь авторов YOLOv7 | Старые базовые модели продолжают использоваться в новых аграрных исследованиях | Их первоначальные статьи вне данного корпуса; рассматриваются через TIA-YOLOv5 и YOLOC-tiny. [26](#ref-wang2022tiayolo) [27](#ref-tang2024yoloc) |
| YOLOv6 v3.0, 2023 | Meituan | Промышленный детектор; развитие neck, обучения и дистилляции | В статье отдельно показаны throughput при batch=32 и задержка при batch=1. Первое не равно скорости одиночной камеры. [1](#ref-li2023yolov6) |
| DAMO-YOLO, 2022–2023 | Alibaba | NAS для backbone, RepGFPN, компактная head и AlignedOTA; варианты для CPU и GPU | Собственная ветвь; её CPU-измерения нельзя подменять GPU-рейтингом. [9](#ref-xu2023damoyolo) |
| YOLOv8, 2023; YOLO11, 2024 | Ultralytics | Практические линейки с отдельными весами для detection и instance segmentation; полезны как воспроизводимые базовые решения | Программный релиз не равен отдельной первичной статье. Здесь научные сведения об их применении взяты из сравнений растений и плодов. [28](#ref-sapkota2026fruitseg) [31](#ref-tijjani2026agriaiseg) |
| YOLOv9, 2024 | Wang, Yeh, Liao; `WongKinYiu/yolov9` | PGI улучшает передачу обучающего сигнала; GELAN организует агрегацию признаков | Это изменения обучения и архитектуры, не временная модель видео. [2](#ref-wang2024yolov9) |
| YOLOv10, 2024 | Tsinghua, THU-MIG | Согласованные dual assignments и архитектурная оптимизация для работы без NMS | Для воспроизведения важны путь инференса и экспорт. В протоколе статьи I/O overhead исключён. [3](#ref-wang2024yolov10) |
| YOLOv12, 2025 | Tian, Ye, Doermann; `sunsmarterjie/yolov12` | Area attention и R-ELAN; попытка получить преимущества внимания при малой задержке | Эффективность зависит от поддерживаемых операций и GPU. Здесь используется technical report v1. [4](#ref-tian2025yolov12) |
| YOLOv13, 2025 | Lei и соавт.; iMoonLab | HyperACE и FullPAD для связей между позициями и масштабами | Сложность архитектуры сама по себе не доказывает преимущество на edge или сорняках. [5](#ref-lei2025yolov13) |
| YOLO26, 2026 | Ultralytics | Ветка без NMS, отказ от DFL, Progressive Loss, STAL и MuSGD; несколько зрительных задач | В таблице статьи AP с NMS и AP E2E различаются. Сравнивать нужно ту ветвь, которая действительно развёрнута. [6](#ref-jocher2026yolo26) |
| YOLO-World, 2024 | Tencent / HUST | Детекция с открытым словарём, RepVL-PAN, предварительная подготовка текстовых признаков | Проверка на LVIS не доказывает распознавание конкретных сорняков; маски требуют соответствующей конфигурации. [8](#ref-cheng2024yoloworld) |
| YOLOE, 2025 | Tsinghua; также интеграция в Ultralytics | Детекция и сегментация с текстовыми, визуальными и prompt-free режимами | Режимы различаются по входам и вычислительной стоимости; полезно для разведочного поиска и подготовки разметки. [7](#ref-wang2025yoloe) |

YOLOv12/v13 не следует считать обязательными промежуточными обновлениями Ultralytics между YOLO11 и YOLO26. Выбираются конкретные репозиторий, версия, задача и checkpoint, а не «наибольший номер YOLO».

## 5. Альтернативные детекторы

| Семейство | Суть | Значение для видеопотока | Основное ограничение |
|---|---|---|---|
| RTMDet / RTMDet-Ins, 2022 | Shanghai AI Lab / OpenMMLab; согласованный backbone/neck, depth-wise свёртки, soft-label matching | Один стек для рамок, масок экземпляров и ориентированных рамок | Скорость оригинальной статьи измерена на RTX 3090; перенос на Jetson требует отдельной проверки. [10](#ref-lyu2022rtmdet) |
| RT-DETR, 2023–2024 | Baidu / Peking University; эффективный гибридный encoder и отбор queries | Детекция без NMS; можно менять число decoder-слоёв | Экспорт и операторы attention могут определять фактическую скорость. [11](#ref-zhao2024rtdetr) |
| RT-DETRv2, 2024 | Развитие RT-DETR: выбор sampling points, опциональная дискретная выборка и улучшение обучения | Улучшение качества без обязательного увеличения инференсной стоимости | Для v1 опубликованы одни результаты; поздние таблицы других авторов могут содержать уже другие значения. [12](#ref-lv2024rtdetrv2) |
| D-FINE, 2024 | USTC; уточнение распределений координат FDR и самодистилляция GO-LSD | Сильная альтернатива для точной локализации | Предобучение на Objects365 и обучение только на COCO нужно сравнивать раздельно. [13](#ref-peng2024dfine) |
| DEIM, 2024–2025 | Intellindust и соавторы; Dense O2O и matching-aware loss | Прежде всего ускоряет сходимость обучения DETR | Меньшее число эпох не означает меньшую задержку камеры. [14](#ref-huang2025deim) |
| DEIMv2, 2025–2026 | DINOv3 и Spatial Tuning Adapter для части размеров; HGNetv2 для самых малых | Диапазон от крупных моделей до Atto/Pico даёт кандидатов под разные бюджеты | Не все размеры используют DINOv3; число параметров не заменяет замер runtime. [15](#ref-huang2026deimv2) |
| RT-DETRv4, 2025–2026 | DSI/GAM для переноса знаний foundation model в компактный детектор | Усиление модели за счёт обучения без переноса тяжёлого teacher в инференс | Проверять, что развёртывается student, а не весь обучающий граф. [16](#ref-liao2026rtdetrv4) |
| RF-DETR, 2025–2026 | Roboflow / CMU; предобученный ViT и weight-sharing NAS; detection и instance segmentation | Представляет интерес для дообучения на новых предметных областях | Авторы отдельно предупреждают: их buffered latency не измеряет непрерывную пропускную способность. [17](#ref-robinson2026rfdetr) |

Для растений полезны два направления: улучшение локализации мелких объектов и перенос предобученных представлений. Однако **учитель при обучении** и **тяжёлая модель в рабочем потоке** — разные вычислительные затраты. Сравнение по FLOPs без учёта операторов и runtime недостаточно. [13](#ref-peng2024dfine) [16](#ref-liao2026rtdetrv4)

## 6. Сегментация: быстрые маски и модели с подсказками

| Модель | Тип задачи и устройство модели | Практическая роль |
|---|---|---|
| PIDNet, CVPR 2023 | Семантическая сегментация; отдельные ветви деталей, контекста и границ | Кандидат для карты «культура / сорняк / фон». Доказательства скорости получены на городских сценах, поэтому требуется аграрное дообучение. [18](#ref-xu2023pidnet) |
| EfficientViT, ICCV 2023 | Семантическая сегментация; лёгкое multi-scale attention | Интересен для высокого разрешения и разных платформ. В работе есть edge GPU и mobile CPU; данные классификации нельзя выдавать за скорость сегментации. [19](#ref-cai2023efficientvit) |
| YOLO-seg, RTMDet-Ins, RF-DETR segmentation | Instance-сегментация обученных категорий | Прямые кандидаты для отдельных растений и плодов; измерять время декодирования и обработки масок. [10](#ref-lyu2022rtmdet) [17](#ref-robinson2026rfdetr) [28](#ref-sapkota2026fruitseg) |
| SAM, 2023 | Маска по подсказке; image encoder, prompt encoder и mask decoder | Базовый инструмент разметки и исследования переносимости. Быстрый decoder не устраняет стоимость encoder на каждом новом кадре. [20](#ref-kirillov2023sam) |
| FastSAM, 2023 | CNN генерирует instance-маски, затем выбирает их по подсказке | Быстрая альтернатива для изображений; не тождественна дистилляции SAM и не содержит видеопамять SAM 2. [21](#ref-zhao2023fastsam) |
| MobileSAM, 2023 | Дистилляция тяжёлого image encoder в компактный, совместимый с decoder SAM | Кандидат для ограниченных ресурсов. В сохранённом тексте есть несогласованные числа 10/12 ms; точное сравнение скорости здесь не используется. [22](#ref-zhang2023mobilesam) |
| EfficientViT-SAM, 2024 | Замена encoder SAM на EfficientViT с обучением/дистилляцией | Подходит для ускорения promptable segmentation. Таблица throughput на A100 не является измерением видеокамеры на Jetson. [23](#ref-zhang2024efficientvitsam) |
| SAM 2, 2024 | Сегментация изображений и видео с потоковой памятью | Для переноса маски выбранного объекта между кадрами; появляющиеся новые объекты требуют логики обнаружения/инициализации. [24](#ref-ravi2024sam2) |
| SAM 3, 2025–2026 | Детектор понятий и memory-based tracker с общим backbone | Поиск, маски и ID по тексту/примеру. Стоимость зависит от числа объектов; в приложении описан Multiplex и оптимизированная внутренняя реализация. Это не готовое доказательство 30 FPS на edge. [25](#ref-carion2026sam3) |

Для получения центров растений instance-маски удобнее рамок, но геометрический центроид видимой листвы может не совпадать со стеблем или точкой обработки. Семантическую маску также можно разделять на компоненты, однако соединённые листья создают неоднозначность. Выбор точки воздействия нуждается в собственной разметке и оценке ошибки; высокий mask AP не заменяет такую проверку.

## 7. Что означают опубликованные скорости

Ориентир для обычной камеры — **25–30 обработанных кадров/с при batch=1**, то есть около 40–33 мс на кадр. Это рабочее определение данного обзора, не универсальная граница. Для движущегося сельхозробота важнее допустимая задержка решения при заданной скорости движения.

Следующая таблица — **не рейтинг**. Строки взяты из разных протоколов; даже одинаковый GPU не делает их напрямую сопоставимыми. `н/у` означает, что необходимая деталь в использованном описании измерения не установлена. Параметры приведены для конкретной указанной конфигурации, не для всего семейства.

| Модель и источник измерения | Данные, качество | Размер / параметры | Платформа и runtime | Batch / точность | Опубликованное время или FPS |
|---|---|---|---|---|---|
| YOLO26n, табл. 7 | COCO val: AP 40,9 с NMS; 40,1 E2E | 640; 2,4M; 5,4 GFLOPs | T4, TensorRT 10; отдельно Xeon 2,00 GHz, ONNX | Batch н/у в табл.; GPU FP16 | 1,7 мс GPU; 38,9 мс CPU. Это опубликованные model benchmarks, не полный видеопоток. [6](#ref-jocher2026yolo26) |
| YOLOv12-N, табл. 1 / методика | COCO AP 40,6 | 640; 2,6M; 6,5 GFLOPs | T4, TensorRT | Batch н/у в использованном описании; FP16 | 1,64 мс; данные technical report v1. [4](#ref-tian2025yolov12) |
| RT-DETRv2-S, табл. 2 v1 | COCO val AP 47,9 | 640×640; 20M | T4, TensorRT | 1 / FP16 | 217 FPS по протоколу авторов. [12](#ref-lv2024rtdetrv2) |
| RTMDet-x, табл. 2 и §4.1 | COCO val AP 52,8 | 640×640; 94,9M | RTX 3090, TensorRT 8.4.3 | 1 / FP16 | 3,10 мс; нельзя сравнивать с T4 без повторного замера. [10](#ref-lyu2022rtmdet) |
| RF-DETR-M, табл. 1 | COCO AP 54,7 в FP16 | Конфигурация M из статьи | T4, TensorRT 10.4, CUDA 12.4 | FP16; batch н/у в табл. | 4,4 мс с паузой 200 мс между проходами; **не sustained FPS**. [17](#ref-robinson2026rfdetr) |
| PIDNet-S, табл. 6 | Cityscapes test mIoU 78,6 | 2048×1024; 7,6M; 47,6 GFLOPs | RTX 3090, PyTorch 1.8 | 1 / численная точность н/у | 93,2 FPS, без TensorRT. [18](#ref-xu2023pidnet) |
| SAM 2 Hiera-B+ / L, §6.3 | Video segmentation, несколько наборов; единый AP неприменим | Разные encoder; вход 1024×1024 в конфигурации модели | A100; детали реализации отдельно от TensorRT-детекторов | 1 / точность в данном сравнении н/у | 43,8 / 30,2 FPS; это видеорежим, не FPS image encoder. [24](#ref-ravi2024sam2) |

**Три причины осторожно читать speed–accuracy графики:**

1. RF-DETR вводит паузы между проходами для стабилизации частот и прямо отделяет эту задержку от непрерывного throughput. Из 4,4 мс нельзя объявить производительность камеры 227 FPS. [17](#ref-robinson2026rfdetr)
2. У YOLO26 одна таблица содержит две колонки AP для разных inference branches. Сравнение должно фиксировать NMS/E2E и используемый артефакт, включая численную точность. [6](#ref-jocher2026yolo26)
3. В сравнении яблок таблица 3 даёт для YOLO11n-seg `0,3 + 4,8 + 56,7 = 61,8 мс`, для YOLOv8n-seg `0,3 + 3,3 + 2,5 = 6,1 мс`. Это **сумма опубликованных стадий**, вычисленная для обзора, а не наш замер камеры. Большая постобработка YOLO11n — результат конкретной работы, который следует перепроверить, а не универсальная характеристика всех реализаций. [28](#ref-sapkota2026fruitseg)

### Протокол проверки на собственной камере

Фиксировать модель, checkpoint и commit, разрешение, режим питания, GPU/CPU, runtime, FP32/FP16/INT8, batch=1 и число объектов. Измерять после прогрева на достаточно длинной непрерывной записи, чтобы увидеть нагрев и пропуски кадров. Отдельно учитывать захват/декодирование, resize, перенос данных, модель, NMS/маски, сопровождение и выдачу решения. Сообщать FPS, медиану и p95/p99 задержки, память, dropped frames и возраст кадра в момент решения.

Качество проверять на **том же экспортированном артефакте**, на котором измерена скорость. Для растений нужны AP50:95 и AP малых объектов, recall сорняков, ложные срабатывания на культуре; для масок — mask AP или class-wise IoU, для видео — стабильность ID и пропуски по последовательности. При тайлинге высокого разрешения измерять обработку всего исходного кадра с объединением результатов. Это предлагаемый протокол будущих экспериментов, а не выполненные здесь испытания.

## 8. Что показывают исследования растений

| Работа | Данные и задача | Проверенный результат | Ограничение для реального видео |
|---|---|---|---|
| TIA-YOLOv5, Wang и соавт., 2022 | Сахарная свёкла и сорняки; детекция, синтез обучающих примеров | mAP50 90,0%; AP сорняков 80,8%; F1 сорняков 70,0%; 20,8 FPS на PC | PyTorch 1.10.2, вход 1296×966, GPU в статье назван «GTX3080Ti». Отдельно заявлено около 70 FPS на Jetson NX с TensorRT, но полный deployment-протокол не раскрыт. [26](#ref-wang2022tiayolo) |
| YOLOC-tiny, Tang и соавт., 2024 | Три вида цитрусовых разной спелости; YOLOv7-based модель, pruning | 4,2M параметров; mAP 83,0% в основном эксперименте; 59 FPS и accuracy 92,8% в опыте на роботе | На роботе PC3 с i7-1165G7 и GeForce MX450, а не Jetson. Accuracy робота не является mAP; условия и датасеты различаются. [27](#ref-tang2024yoloc) |
| YOLO11-seg против YOLOv8-seg, Sapkota и Karkee, первая подача 2024, v4 2026 | Незрелые зелёные яблоки с перекрытиями и без них | YOLO11m-seg: mask mAP50 0,860 по всем категориям; отдельно приведены стадии обработки | Описана станция TITAN Xp, вход 640×640; training batch=8 не доказывает inference batch=1. См. аномалию постобработки выше. [28](#ref-sapkota2026fruitseg) |
| Lightweight Multispectral Crop-Weed Segmentation, 2025 | WeedsGalore; RGB+NIR+Red-Edge, гибрид CNN/transformer | mIoU 78,88%; RGB-only 63,08%; 8,7M параметров | В тексте есть заявления о real-time, но нет измеренного FPS/latency с платформой. Это кандидат, не подтверждённое edge-решение; нужен мультиспектральный сенсор. [29](#ref-galymzhankyzy2025multispectral) |
| WeedSense, 2025 | 16 видов, тепличные растения, 120 341 кадр из 349 видео; маски, высота и стадия роста | mIoU 89,78%; 30,50M параметров; 160 FPS, V100S, 512×512, evaluation batch=1 | Тепличные данные не равны смешанному полю. Высокая близость кадров требует проверки независимости split по растениям и видео; утечка здесь не утверждается. [30](#ref-sarker2025weedsense) |
| AgriAISeg comparison, 2026 | Капуста, томат, кунжут; YOLOv5/v8/11/26, RT-DETR, Faster R-CNN | RT-DETR AP50:95 0,624; YOLOv8 0,621; YOLO26 0,606 в таблице авторов | Нет достаточного протокола inference FPS. Колонка `Time (hrs)` относится к обучению; размеры вариантов раскрыты недостаточно для общего рейтинга архитектур. [31](#ref-tijjani2026agriaiseg) |
| Multi-Dataset Weed Benchmark, 2026 | Семь наборов; YOLO26n/s/m, междоменная проверка | Для YOLO26s средний AP50:95 в одно-классовом анализе падает с 0,603 внутри домена до 0,148 вне домена | GTX 1080 Ti, вход 640×640; для скорости не полностью раскрыты batch/precision/runtime. Главный вывод — риск domain shift, а не гарантированный FPS на роботе. [32](#ref-zdraveska2026weedbenchmark) |

Аграрных работ в таблице **семь**: детекция полевых культур, сорняков и плодов, instance- и семантическая сегментация. Их значения качества нельзя ранжировать в одной колонке «accuracy»: задачи, классы, IoU-пороги, сенсоры и разбиения различаются. В частности, пиксельный mIoU, mask AP, box AP и accuracy опыта на роботе измеряют разные свойства.

### Практические следствия для табака и сорняков

- **Мелкие объекты.** Уменьшение кадра ускоряет инференс, но может убрать маленький сорняк. Сравнивать разрешения и тайлинг по полноте обнаружения и полной задержке кадра.
- **Перекрытия.** Отдельно оценивать рамки и маски на перекрытых листьях; лучший общий box AP не гарантирует разделение экземпляров.
- **Разметка.** Для карты «табак / сорняк / почва» нужны пиксельные классы; для отдельных растений — instance-маски. Вычитание рамок табака из зелёной маски само по себе не доказывает, что остаток является сорняком.
- **Перенос.** Разделять обучение и тестирование по участкам, датам и видеопроходам, а не случайным соседним кадрам. Нужен отдельный тест другого поля, освещения и стадии роста.
- **Предобучение и подсказки.** SAM/YOLOE могут ускорять подготовку черновой разметки; качество псевдомасок всё равно проверяется вручную. Открытый словарь не заменяет независимый тест ботанических категорий.

Эти пункты — рекомендации автора обзора по проектированию исследования. Представленные статьи не подтверждают точность масок, координат или успешность лазерной обработки именно табака.

## 9. Кандидаты по платформам

| Условия | С чего начать сравнение | Следующий кандидат | Что решает выбор |
|---|---|---|---|
| NVIDIA GPU, рамки растений | YOLOv8n/s как baseline; YOLO26n/s с зафиксированным E2E-режимом | RF-DETR, D-FINE/DEIMv2, RT-DETRv2/v4 | Полнота мелких сорняков, AP50:95, экспорт, p95 задержки; не номер версии |
| NVIDIA GPU, отдельные маски | YOLOv8-seg / YOLO11-seg / YOLO26-seg | RTMDet-Ins, RF-DETR segmentation | Mask AP, контуры и стоимость postprocess |
| Jetson / edge | Nano/small YOLO с TensorRT; компактный RTMDet или DEIMv2 как альтернативы | PIDNet-S / EfficientViT для семантической карты | Реальный режим питания и память; поддержка операций; FP16/INT8 проверяются по качеству |
| CPU | Самые малые YOLO/DAMO-YOLO; уменьшение входа только после проверки recall | Компактная семантическая модель при необходимости масок | Замеры на конкретном CPU с фиксированным числом потоков. Например, 38,9 мс YOLO26n на указанном Xeon уже оставляют мало бюджета для полного цикла 25 FPS. [6](#ref-jocher2026yolo26) |
| Маска выбранного объекта на видео | SAM 2 | SAM 3, если нужен поиск по понятиям | Число объектов, стоимость инициализации, потеря/возврат объекта, память |
| Подготовка разметки | SAM / EfficientViT-SAM; YOLOE для поиска кандидатов | SAM 3 для текстовых понятий | Качество и время исправления человеком; не обязательно real-time deployment |

Это **список для испытаний**, а не утверждение, что все кандидаты уже достигают 30 FPS на каждом устройстве. Для первого исследования табака достаточно сопоставить один привычный YOLO baseline, одну современную YOLO-модель и один DETR. Добавлять тяжёлые и open-vocabulary варианты следует, если они отвечают конкретной задаче данных или разметки.

### Программная доступность

На дату проверки корневые LICENSE официальных репозиториев указывают: [Ultralytics — AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [RF-DETR — Apache-2.0](https://github.com/roboflow/rf-detr/blob/develop/LICENSE), [MMDetection — Apache-2.0](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE), [RT-DETR — Apache-2.0](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE), [SAM 2 — Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE), [SAM 3 — собственная SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Это сведения о проверенных репозиториях, не единое разрешение для любых весов, данных и производных пакетов. Перед фиксацией экспериментальной среды сохраняются конкретные версии и условия используемых артефактов.

## 10. Что остаётся неизвестным

Для целевого видеопотока табака не измерены качество, задержка и перенос между полями ни одной из новых моделей. Особенно слабо обоснованы перенос GPU-результатов на CPU/Jetson, работа при большом числе перекрытых растений и точность конечной координаты. Часть статей сообщает только model latency, часть — недостаточно подробный FPS. Препринты могут пересматриваться; повторное сравнение требует использовать сохранённую версию либо явно обновить весь связанный набор результатов.

Следующий обоснованный шаг — согласованный benchmark на независимых видеопроходах: одинаковая разметка, зафиксированные splits, общий бюджет обучения и измерение качества и полного цикла на одном устройстве. В этом отчёте такого эксперимента нет.

## 11. Литература и локальные PDF

Номер ссылки ведёт к записи ниже; `BibTeX` — ключ в `list.bib`. Для arXiv год означает год сохранённой версии. Полный список авторов и машинно-читаемые сведения находятся в реестре.

<a id="ref-li2023yolov6"></a>

**[1] Li, Chuyi и соавт. (2023).** YOLOv6 v3.0: A Full-Scale Reloading. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2023-01-13; сохранена v1 от 2023-01-13. [Официальный ресурс](https://arxiv.org/abs/2301.05586) · [PDF](articles/Li%202023%20YOLOv6%20v3.0%20A%20Full-Scale%20Reloading.pdf). BibTeX: `li2023yolov6`.

<a id="ref-wang2024yolov9"></a>

**[2] Wang, Chien-Yao и соавт. (2024).** YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2024-02-21; сохранена v2 от 2024-02-29. [Официальный ресурс](https://arxiv.org/abs/2402.13616) · [PDF](articles/Wang%202024%20YOLOv9%20Learning%20What%20You%20Want%20to%20Learn%20Using%20Programmable%20Gradient%20Information.pdf). BibTeX: `wang2024yolov9`.

<a id="ref-wang2024yolov10"></a>

**[3] Wang, Ao и соавт. (2024).** YOLOv10: Real-Time End-to-End Object Detection. Авторская версия; NeurIPS 2024 camera-ready указан на arXiv. Первая подача: 2024-05-23; сохранена v2 от 2024-10-30. [Официальный ресурс](https://arxiv.org/abs/2405.14458) · [PDF](articles/Wang%202024%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.pdf). BibTeX: `wang2024yolov10`.

<a id="ref-tian2025yolov12"></a>

**[4] Tian, Yunjie и соавт. (2025).** YOLOv12: Attention-Centric Real-Time Object Detectors. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2025-02-18; сохранена v1 от 2025-02-18. [Официальный ресурс](https://arxiv.org/abs/2502.12524) · [PDF](articles/Tian%202025%20YOLOv12%20Attention-Centric%20Real-Time%20Object%20Detectors.pdf). BibTeX: `tian2025yolov12`.

<a id="ref-lei2025yolov13"></a>

**[5] Lei, Mengqi и соавт. (2025).** YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2025-06-21; сохранена v2 от 2025-09-05. [Официальный ресурс](https://arxiv.org/abs/2506.17733) · [PDF](articles/Lei%202025%20YOLOv13%20Real-Time%20Object%20Detection%20with%20Hypergraph-Enhanced%20Adaptive%20Visual%20Perception.pdf). BibTeX: `lei2025yolov13`.

<a id="ref-jocher2026yolo26"></a>

**[6] Jocher, Glenn и соавт. (2026).** Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2026-06-02; сохранена v1 от 2026-06-02. [Официальный ресурс](https://arxiv.org/abs/2606.03748) · [PDF](articles/Jocher%202026%20Ultralytics%20YOLO26%20Unified%20Real-Time%20End-to-End%20Vision%20Models.pdf). BibTeX: `jocher2026yolo26`.

<a id="ref-wang2025yoloe"></a>

**[7] Wang, Ao и соавт. (2025).** YOLOE: Real-Time Seeing Anything. Авторская версия; ICCV 2025 camera-ready указан на arXiv. Первая подача: 2025-03-10; сохранена v2 от 2025-10-17. [Официальный ресурс](https://arxiv.org/abs/2503.07465) · [PDF](articles/Wang%202025%20YOLOE%20Real-Time%20Seeing%20Anything.pdf). BibTeX: `wang2025yoloe`.

<a id="ref-cheng2024yoloworld"></a>

**[8] Cheng, Tianheng и соавт. (2024).** YOLO-World: Real-Time Open-Vocabulary Object Detection. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2024-01-30; сохранена v3 от 2024-02-22. [Официальный ресурс](https://arxiv.org/abs/2401.17270) · [PDF](articles/Cheng%202024%20YOLO-World%20Real-Time%20Open-Vocabulary%20Object%20Detection.pdf). BibTeX: `cheng2024yoloworld`.

<a id="ref-xu2023damoyolo"></a>

**[9] Xu, Xianzhe и соавт. (2023).** DAMO-YOLO : A Report on Real-Time Object Detection Design. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2022-11-23; сохранена v4 от 2023-04-24. [Официальный ресурс](https://arxiv.org/abs/2211.15444) · [PDF](articles/Xu%202023%20DAMO-YOLO%20A%20Report%20on%20Real-Time%20Object%20Detection%20Design.pdf). BibTeX: `xu2023damoyolo`.

<a id="ref-lyu2022rtmdet"></a>

**[10] Lyu, Chengqi и соавт. (2022).** RTMDet: An Empirical Study of Designing Real-Time Object Detectors. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2022-12-14; сохранена v2 от 2022-12-16. [Официальный ресурс](https://arxiv.org/abs/2212.07784) · [PDF](articles/Lyu%202022%20RTMDet%20An%20Empirical%20Study%20of%20Designing%20Real-Time%20Object%20Detectors.pdf). BibTeX: `lyu2022rtmdet`.

<a id="ref-zhao2024rtdetr"></a>

**[11] Zhao, Yian и соавт. (2024).** DETRs Beat YOLOs on Real-time Object Detection. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2023-04-17; сохранена v3 от 2024-04-03. [Официальный ресурс](https://arxiv.org/abs/2304.08069) · [PDF](articles/Zhao%202024%20DETRs%20Beat%20YOLOs%20on%20Real-time%20Object%20Detection.pdf). BibTeX: `zhao2024rtdetr`.

<a id="ref-lv2024rtdetrv2"></a>

**[12] Lv, Wenyu и соавт. (2024).** RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2024-07-24; сохранена v1 от 2024-07-24. [Официальный ресурс](https://arxiv.org/abs/2407.17140) · [PDF](articles/Lv%202024%20RT-DETRv2%20Improved%20Baseline%20with%20Bag-of-Freebies%20for%20Real-Time%20Detection%20Transformer.pdf). BibTeX: `lv2024rtdetrv2`.

<a id="ref-peng2024dfine"></a>

**[13] Peng, Yansong и соавт. (2024).** D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2024-10-17; сохранена v1 от 2024-10-17. [Официальный ресурс](https://arxiv.org/abs/2410.13842) · [PDF](articles/Peng%202024%20D-FINE%20Redefine%20Regression%20Task%20in%20DETRs%20as%20Fine-grained%20Distribution%20Refinement.pdf). BibTeX: `peng2024dfine`.

<a id="ref-huang2025deim"></a>

**[14] Huang, Shihua и соавт. (2025).** DEIM: DETR with Improved Matching for Fast Convergence. Авторская версия; CVPR 2025 указан на arXiv. Первая подача: 2024-12-05; сохранена v3 от 2025-03-26. [Официальный ресурс](https://arxiv.org/abs/2412.04234) · [PDF](articles/Huang%202025%20DEIM%20DETR%20with%20Improved%20Matching%20for%20Fast%20Convergence.pdf). BibTeX: `huang2025deim`.

<a id="ref-huang2026deimv2"></a>

**[15] Huang, Shihua и соавт. (2026).** Real-Time Object Detection Meets DINOv3. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2025-09-25; сохранена v4 от 2026-01-26. [Официальный ресурс](https://arxiv.org/abs/2509.20787) · [PDF](articles/Huang%202026%20Real-Time%20Object%20Detection%20Meets%20DINOv3.pdf). BibTeX: `huang2026deimv2`.

<a id="ref-liao2026rtdetrv4"></a>

**[16] Liao, Zijun и соавт. (2026).** RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2025-10-29; сохранена v2 от 2026-10-08. [Официальный ресурс](https://arxiv.org/abs/2510.25257) · [PDF](articles/Liao%202026%20RT-DETRv4%20Painlessly%20Furthering%20Real-Time%20Object%20Detection%20with%20Vision%20Foundation%20Models.pdf). BibTeX: `liao2026rtdetrv4`.

<a id="ref-robinson2026rfdetr"></a>

**[17] Robinson, Isaac и соавт. (2026).** RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. Авторская версия; ICLR 2026 указан на arXiv и в PDF. Первая подача: 2025-11-12; сохранена v2 от 2026-02-03. [Официальный ресурс](https://arxiv.org/abs/2511.09554) · [PDF](articles/Robinson%202026%20RF-DETR%20Neural%20Architecture%20Search%20for%20Real-Time%20Detection%20Transformers.pdf). BibTeX: `robinson2026rfdetr`.

<a id="ref-xu2023pidnet"></a>

**[18] Xu, Jiacong и соавт. (2023).** PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers. Рецензируемая публикация; издательский PDF. Издательская версия. [Официальный ресурс](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) · [PDF](articles/Xu%202023%20PIDNet%20A%20Real-Time%20Semantic%20Segmentation%20Network%20Inspired%20by%20PID%20Controllers.pdf). BibTeX: `xu2023pidnet`.

<a id="ref-cai2023efficientvit"></a>

**[19] Cai, Han и соавт. (2023).** EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction. Рецензируемая публикация; издательский PDF. Издательская версия. [Официальный ресурс](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html) · [PDF](articles/Cai%202023%20EfficientViT%20Lightweight%20Multi-Scale%20Attention%20for%20High-Resolution%20Dense%20Prediction.pdf). BibTeX: `cai2023efficientvit`.

<a id="ref-kirillov2023sam"></a>

**[20] Kirillov, Alexander и соавт. (2023).** Segment Anything. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2023-04-05; сохранена v1 от 2023-04-05. [Официальный ресурс](https://arxiv.org/abs/2304.02643) · [PDF](articles/Kirillov%202023%20Segment%20Anything.pdf). BibTeX: `kirillov2023sam`.

<a id="ref-zhao2023fastsam"></a>

**[21] Zhao, Xu и соавт. (2023).** Fast Segment Anything. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2023-06-21; сохранена v1 от 2023-06-21. [Официальный ресурс](https://arxiv.org/abs/2306.12156) · [PDF](articles/Zhao%202023%20Fast%20Segment%20Anything.pdf). BibTeX: `zhao2023fastsam`.

<a id="ref-zhang2023mobilesam"></a>

**[22] Zhang, Chaoning и соавт. (2023).** Faster Segment Anything: Towards Lightweight SAM for Mobile Applications. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2023-06-25; сохранена v2 от 2023-07-01. [Официальный ресурс](https://arxiv.org/abs/2306.14289) · [PDF](articles/Zhang%202023%20Faster%20Segment%20Anything%20Towards%20Lightweight%20SAM%20for%20Mobile%20Applications.pdf). BibTeX: `zhang2023mobilesam`.

<a id="ref-zhang2024efficientvitsam"></a>

**[23] Zhang, Zhuoyang и соавт. (2024).** EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss. Авторская версия; CVPR 2024 Workshop указан на arXiv. Первая подача: 2024-02-07; сохранена v2 от 2024-05-16. [Официальный ресурс](https://arxiv.org/abs/2402.05008) · [PDF](articles/Zhang%202024%20EfficientViT-SAM%20Accelerated%20Segment%20Anything%20Model%20Without%20Accuracy%20Loss.pdf). BibTeX: `zhang2024efficientvitsam`.

<a id="ref-ravi2024sam2"></a>

**[24] Ravi, Nikhila и соавт. (2024).** SAM 2: Segment Anything in Images and Videos. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2024-08-01; сохранена v2 от 2024-10-28. [Официальный ресурс](https://arxiv.org/abs/2408.00714) · [PDF](articles/Ravi%202024%20SAM%202%20Segment%20Anything%20in%20Images%20and%20Videos.pdf). BibTeX: `ravi2024sam2`.

<a id="ref-carion2026sam3"></a>

**[25] Carion, Nicolas и соавт. (2026).** SAM 3: Segment Anything with Concepts. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2025-11-20; сохранена v2 от 2026-03-28. [Официальный ресурс](https://arxiv.org/abs/2511.16719) · [PDF](articles/Carion%202026%20SAM%203%20Segment%20Anything%20with%20Concepts.pdf). BibTeX: `carion2026sam3`.

<a id="ref-wang2022tiayolo"></a>

**[26] Wang, Aichen и соавт. (2022).** TIA-YOLOv5: An improved YOLOv5 network for real-time detection of crop and weed in the field. Рецензируемая публикация; издательский PDF. Издательская версия. [Официальный ресурс](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1091655/full) · [PDF](articles/Wang%202022%20TIA-YOLOv5%20An%20improved%20YOLOv5%20network%20for%20real-time%20detection%20of%20crop%20and%20weed%20in%20the%20field.pdf). BibTeX: `wang2022tiayolo`.

<a id="ref-tang2024yoloc"></a>

**[27] Tang, Zuoliang и соавт. (2024).** YOLOC-tiny: a generalized lightweight real-time detection model for multiripeness fruits of large non-green-ripe citrus in unstructured environments. Рецензируемая публикация; издательский PDF. Издательская версия. [Официальный ресурс](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1415006/full) · [PDF](articles/Tang%202024%20YOLOC-tiny%20a%20generalized%20lightweight%20real-time%20detection%20model%20for%20multiripeness%20fruits%20of%20large%20non-green-ripe%20citrus%20in%20unstructured%20environments.pdf). BibTeX: `tang2024yoloc`.

<a id="ref-sapkota2026fruitseg"></a>

**[28] Sapkota, Ranjan; Karkee, Manoj (2026).** Comparing YOLOv11 and YOLOv8 for instance segmentation of occluded and non-occluded immature green fruits in complex orchard environment. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2024-10-24; сохранена v4 от 2026-09-24. [Официальный ресурс](https://arxiv.org/abs/2410.19869) · [PDF](articles/Sapkota%202026%20Comparing%20YOLOv11%20and%20YOLOv8%20for%20instance%20segmentation%20of%20occluded%20and%20non-occluded%20immature%20green%20fruits%20in%20complex%20orchard%20environment.pdf). BibTeX: `sapkota2026fruitseg`.

<a id="ref-galymzhankyzy2025multispectral"></a>

**[29] Galymzhankyzy, Zeynep; Martinson, Eric (2025).** Lightweight Multispectral Crop-Weed Segmentation for Precision Agriculture. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2025-05-12; сохранена v1 от 2025-05-12. [Официальный ресурс](https://arxiv.org/abs/2505.07444) · [PDF](articles/Galymzhankyzy%202025%20Lightweight%20Multispectral%20Crop-Weed%20Segmentation%20for%20Precision%20Agriculture.pdf). BibTeX: `galymzhankyzy2025multispectral`.

<a id="ref-sarker2025weedsense"></a>

**[30] Sarker, Toqi Tahamid и соавт. (2025).** WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification. Авторская версия; принятие на ICCVW 2025 указано на arXiv. Первая подача: 2025-08-20; сохранена v1 от 2025-08-20. [Официальный ресурс](https://arxiv.org/abs/2508.14486) · [PDF](articles/Sarker%202025%20WeedSense%20Multi-Task%20Learning%20for%20Weed%20Segmentation%2C%20Height%20Estimation%2C%20and%20Growth%20Stage%20Classification.pdf). BibTeX: `sarker2025weedsense`.

<a id="ref-tijjani2026agriaiseg"></a>

**[31] Tijjani, Ismail Ismail и соавт. (2026).** A Comparative Evaluation of Deep Learning Object Detection Models on a Real-World Multi-Plant Dataset from Africa. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2026-08-11; сохранена v1 от 2026-08-11. [Официальный ресурс](https://arxiv.org/abs/2608.11053) · [PDF](articles/Tijjani%202026%20A%20Comparative%20Evaluation%20of%20Deep%20Learning%20Object%20Detection%20Models%20on%20a%20Real-World%20Multi-Plant%20Dataset%20from%20Africa.pdf). BibTeX: `tijjani2026agriaiseg`.

<a id="ref-zdraveska2026weedbenchmark"></a>

**[32] Zdraveska, Hristina и соавт. (2026).** A Multi-Dataset Benchmark of YOLO-Based Weed Detection in Precision Agriculture. Препринт / технический отчёт arXiv; рецензирование этой версии не подтверждено. Первая подача: 2026-09-27; сохранена v1 от 2026-09-27. [Официальный ресурс](https://arxiv.org/abs/2609.33991) · [PDF](articles/Zdraveska%202026%20A%20Multi-Dataset%20Benchmark%20of%20YOLO-Based%20Weed%20Detection%20in%20Precision%20Agriculture.pdf). BibTeX: `zdraveska2026weedbenchmark`.

