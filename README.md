# 🎓 Attendance System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/FastAPI-0.142.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Multi--stage-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/OpenCV-Headless-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Active_Directory-LDAP-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Active Directory">
  <img src="https://img.shields.io/badge/Offline-Ready-success?style=for-the-badge" alt="Offline Ready">
</p>

Система автоматизированного контроля посещаемости на базе компьютерного зрения и распознавания лиц[cite: 21]. Разработана для закрытых корпоративных и ведомственных сетей без постоянного доступа к интернету. Поддерживает аутентификацию учетных записей Active Directory (Windows Server 2012+), регистрацию студентов снимком с камеры в один клик и выгрузку форматированных отчетов в Excel.

---

## 🚀 Основные возможности

* **Распознавание лиц в реальном времени**: многопоточная обработка видеопотока (RTSP IP-камеры и локальные USB-устройства) с кэшированием дескрипторов лиц.
* **Интеграция с Active Directory**: проверка паролей по протоколу LDAP через контроллер домена Windows Server с разграничением ролей (Администратор / Просмотр).
* **Работа в автономном контуре (Offline)**: фронтенд функционирует на базе локального скрипта `static/tailwind.js`[cite: 13, 19] без обращения к внешним CDN.
* **Снимок лица без внешних устройств**: захват эталонного фото студента напрямую из веб-интерфейса трансляции без необходимости загружать файлы с телефонов или флешек.
* **Сводный табель в Excel**: генерация двусторонней таблицы с матрицей присутствия (цветовая маркировка по дням, процент явки) и детального хронологического журнала.
* **Оптимизированный Docker-образ**: многоэтапная сборка (Multi-stage build) сжата до ~466 МБ с полным сохранением C++ компонентов `dlib` и headless-библиотек[cite: 2, 8].

---

## 🛠 Технологический стек

| Направление | Стек |
| :--- | :--- |
| **Backend & API** | Python 3.11, FastAPI, Uvicorn, Starlette[cite: 10, 11] |
| **Computer Vision** | OpenCV Headless, Dlib, Face Recognition[cite: 4, 8] |
| **База данных** | SQLite3[cite: 3] |
| **Аутентификация** | LDAP3, ItsDangerous (SessionMiddleware)[cite: 10, 11] |
| **Генерация отчетов** | OpenPyXL |
| **Контейнеризация и CI/CD** | Docker, Docker Buildx, GitHub Actions[cite: 9, 21] |

---

## 📋 Структура проекта

```text
├── .github/
│   └── workflows/
│       └── build.yml          # Автоматическая сборка Docker-образа в GitHub Actions[cite: 9, 22]
├── static/
│   └── tailwind.js            # Локальный standalone-скрипт Tailwind CSS[cite: 13, 19]
├── templates/
│   ├── index.html             # Главная страница: стрим камеры, журнал посещений, выгрузка Excel
│   ├── upload.html            # Картотека: фото с веб-камеры, загрузка файла, удаление
│   ├── login.html             # Авторизация через Active Directory или Dev-режим
│   └── admins.html            # Панель управления списком администраторов
├── uploads/                   # Каталог с фотографиями лиц студентов
├── camera.py                  # Потокобезопасный модуль видеозахвата и детекции лиц
├── config.json                # Конфигурация камеры, домена и учетных записей[cite: 22]
├── database.py                # Слой SQLite: хранение отметок и прав администраторов[cite: 3, 22]
├── excel_export.py            # Построитель стилизованных отчетов Excel (.xlsx)
├── main.py                    # Точка входа, эндпоинты FastAPI и middleware сессий[cite: 22]
├── Dockerfile                 # Многоэтапный Dockerfile (Multi-stage build)[cite: 22]
├── .dockerignore              # Исключение лишних файлов (venv, .git, .idea, кэш) из сборки[cite: 19, 21]
└── requirements.txt           # Зафиксированные версии библиотек проекта[cite: 22]
```

---

## ⚙️ Конфигурация (config.json)
### Параметры окружения и подключения настраиваются в файле config.json[cite: 22]:

```JSON
{
  "video_source": "0",
  "ad_server": "192.168.1.10",
  "ad_domain": "CORP.LOCAL",
  "initial_admin": "admin",
  "dev_mode": true,
  "dev_password": "admin"
}
```

* `video_source`: номер камеры (0, 1) либо RTSP-ссылка вида `rtsp://admin:12345@192.168.1.50:554/h264`.
* `ad_server`: сетевое имя или IP-адрес контроллера домена Windows Server.
* `ad_domain`: имя домена Active Directory (например, `CORP.LOCAL`).
* `initial_admin`: логин первого пользователя Windows, автоматически получающего права администратора при создании базы.
* `dev_mode`: true для локального тестирования без доступа к домену (вход по dev_password); `false` для боевой проверки через LDAP Windows Server.

---

## 📦 Развертывание и запуск
### 1. Запуск через готовый Docker-образ (рекомендуемый)
1. Перейдите во вкладку Actions в репозитории.
2. Откройте последний запуск воркфлоу Build and Export Docker Image и скачайте артефакт docker-image-linux[cite: 18].
3. Распакуйте скачанный zip-архив, чтобы получить файл attendance-system.tar.gz.
4. Импортируйте образ в Docker:
```Bash
    docker load < attendance-system.tar.gz
```
5. Запустите контейнер с привязкой постоянных директорий и конфига:
```Bash
    docker run -d \
      -p 8000:8000 \
      --name attendance-service \
      -v $(pwd)/config.json:/app/config.json \
      -v $(pwd)/uploads:/app/uploads \
      -v $(pwd)/attendance.db:/app/attendance.db \
      attendance-system:latest
```
6. Откройте веб-интерфейс в браузере: http://localhost:8000.
### 2. Локальный запуск для разработки
1. Склонируйте репозиторий:
```Bash
    git clone [https://github.com/Major4ik0/attendance-system-.git](https://github.com/Major4ik0/attendance-system-.git)
    cd attendance-system-
```

2. Разверните виртуальное окружение:
```Bash
    python3.11 -m venv venv
    source venv/bin/activate  # Для Windows: venv\Scripts\activate
```

3. Установите зависимости:
```Bash
    pip install --upgrade pip
    pip install -r requirements.txt
```

4. Запустите приложение:
```Bash
    python main.py
```

---

## 📊 Формат экспорта Excel
### При нажатии кнопки «Скачать Excel» система формирует документ `.xlsx` с двумя листами:
1. Сводный табель:
   * Матрица студентов и всех дней заданного периода.
   * Зеленый маркер `БЫЛ` при фиксации системой и серый маркер `Н` при отсутствии.
   * Подсчет общего количества посещений и процентной явки с закрепленной шапкой для прокрутки.
2. Детальный журнал:
   * Построчный хронологический реестр с датой, ФИО студента, статусом и точным временем фиксации.

---

<div align="center">
  <a href="https://github.com/Major4ik0">
    <img src="https://github.com/Major4ik0.png" width="80px;" style="border-radius: 50%;" alt="Major4ik0"/><br />
    <sub><b>Major4ik0</b></sub>
  </a>
</div>

---