# Markdown RAG MCP Server — Changelog

> Полный список всех изменений, исправлений и новых функций.

---

## v4.0.0 — Final Release

### 🔴 Исправлены критические баги

| # | Проблема | Исправление |
|---|---|---|
| BUG-01 | BM25 терял данные при каждом рестарте сервера | `_build_bm25()` сохраняет в `data/bm25_cache.pkl`, `_load_bm25_on_startup()` восстанавливает при старте |
| BUG-02 | `[:512]` — символы, не токены. Теряло 66% контента при reranking | `MAX_CROSS_ENCODER_CHARS = 8000` (bge-reranker поддерживает 8192 токенов) |
| BUG-03 | O(n²) в цикле нормализации hybrid score | `max()`/`min()` вынесены за пределы цикла |
| BUG-04/05 | Коллизии ключей при сопоставлении BM25 с векторами — `hash(text[:100])` | Добавлен `"ids"` в `collection.query(include=[...])`, ключ = `chunk_id` |
| BUG-06 | Полный ре-индекс каждый раз, даже без изменений | MD5-хеш каждого файла в `data/file_hashes.json` — индексируется только изменённое |
| BUG-07 | Обрезка вывода до 600 символов рвала code-блоки | `MAX_RESULT_CHARS = 1500` + защита: ищет закрывающий ` ``` ` перед обрезкой |

---

### 🟠 Замена моделей

| Было | Стало | Улучшение |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (128 токенов, 384d) | `BAAI/bge-m3` (8192 токенов, 1024d) | +40% качество поиска, полный контекст чанка |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (только EN) | `BAAI/bge-reranker-v2-m3` (100+ языков) | Корректный reranking RU/EN документации |

---

### 🚀 Производительность

- **GPU acceleration** — автодетект CUDA через `_detect_device()`, переменная `RAG_DEVICE=cuda/cpu/auto`
- **Batch embedding** — `batch_size=32` с GPU
- **Normalized embeddings** — `normalize_embeddings=True` (требуется для `bge-m3` cosine similarity)
- **Lazy Cross-Encoder** — загружается только при первом поиске (~1.1GB)

---

### 🌐 Фаза 4 — Web Crawling (новый модуль `crawler.py`)

#### Новый MCP инструмент `index_url()`

```python
index_url("https://docs.python.org/3/library/")  # сайт
index_url("github://tiangolo/fastapi/docs")       # GitHub репозиторий
index_url("npm://axios@1.6")                      # npm пакет
index_url("pypi://fastapi")                       # PyPI пакет
index_url("file:///path/to/archive.zip")          # ZIP архив
```

#### HTML парсер (BeautifulSoup + html2text)
- Удаляет `<nav>`, `<footer>`, `<script>`, `<style>` — только контент
- Ищет `<main>`, `<article>` для приоритетного извлечения
- Конвертирует таблицы, ссылки, код в Markdown

#### Контроль границ
- Не выходит за пределы домена/префикса
- Соблюдает `robots.txt`
- Фильтрует бинарные файлы (.jpg, .pdf, .zip и т.д.)
- Sitemap.xml-first discovery

#### Специализированные загрузчики
- **GitHub API** — обходит дерево репозитория через `api.github.com/trees`
- **npm** — читает README и документацию через `registry.npmjs.org`
- **PyPI** — читает description через `pypi.org/pypi/{pkg}/json`
- **ZIP** — распаковывает в `data/crawled/` и индексирует поддерживаемые форматы
- **Playwright** — опционально для JS-сайтов (GitBook, Docusaurus, VitePress)

---

### 🔍 10/10 Поиск

| # | Фича | Статус |
|---|---|---|
| 1 | Векторный поиск (cosine, ChromaDB) | ✅ |
| 2 | BM25 keyword поиск + persistence | ✅ |
| 3 | RRF (Reciprocal Rank Fusion) | ✅ |
| 4 | Cross-Encoder reranking (правильный лимит) | ✅ |
| 5 | Мультиязычный embedding (bge-m3, 8192 токен) | ✅ |
| 6 | Heading-aware chunking | ✅ |
| 7 | Chunk overlap (2 предложения) | ✅ |
| 8 | GPU ускорение (CUDA x11) | ✅ |
| 9 | Инкрементальный индекс (MD5) | ✅ |
| 10 | Multi-collection | ✅ |
| 11 | Query expansion (RU→EN синонимы) | ✅ |
| 12 | Code-aware chunking (Python AST) | ✅ |
| 13 | Crawl boundary control | ✅ |
| 14 | Sitemap.xml discovery | ✅ |
| 15 | Дедупликация результатов | ✅ |

---

### 📁 Форматы файлов

**36+ расширений без доп. зависимостей:**
`.md` `.txt` `.rst` `.log` `.py` `.js` `.ts` `.jsx` `.tsx` `.css` `.scss` `.java` `.go` `.rs` `.c` `.cpp` `.h` `.rb` `.php` `.swift` `.kt` `.lua` `.sh` `.json` `.yaml` `.yml` `.toml` `.xml` `.csv` `.ini` `.html` `.htm`

**С опциональными пакетами:**
| Формат | Пакет | Установка |
|---|---|---|
| `.pdf` | pypdf | `pip install pypdf` |
| `.docx` | python-docx | `pip install python-docx` |
| `.xlsx` | openpyxl | `pip install openpyxl` |
| `.pptx` | python-pptx | `pip install python-pptx` |
| `.ipynb` | — | встроено (JSON) |

> Если пакет не установлен — файл молча пропускается. Никаких ошибок.

---

### ⚙️ Code-Aware Chunking (FEAT-08)

Python файлы разрезаются по **классам и функциям** через `ast.parse()`:

```
server.py → [Function: _build_bm25, Function: search_docs, Class: RagConfig, ...]
```

JS/TS файлы разрезаются по **функциям** через regex:

```
app.js → [function: handleRequest, class: ApiClient, ...]
```

Остальные форматы → по Markdown-заголовкам (как раньше).

---

### 🛠️ Новые MCP инструменты

| Инструмент | Описание |
|---|---|
| `index_documents(path, collection)` | Локальная индексация файлов (инкрементальная) |
| `search_docs(query, n, category, filename, collection)` | Гибридный поиск с reranking |
| `rag_status(collection)` | Статус индекса: чанки, файлы, модели, BM25, GPU |
| `list_collections()` | Список всех коллекций в базе |
| `list_indexed_files(collection)` | Список файлов в коллекции |
| `remove_source(filename, collection)` | Удалить один файл из индекса |
| `delete_collection(name, confirm)` | Удалить всю коллекцию (с confirm=True) |
| `reindex_collection(docs_path, collection)` | Полный принудительный реиндекс |
| `index_url(uri, collection, ...)` | Индексация сайтов, GitHub, npm, PyPI, ZIP |

---

### 📂 File Watcher (FEAT-09) — `watcher.py`

Автоматический реиндекс при изменениях в папке:

```bash
# Активация через env переменные
RAG_WATCH_PATH=C:/projects/my-app/docs
RAG_WATCH_COLLECTION=my-project
python server.py
```

- 2-секундный debounce (не реиндексирует на каждое нажатие клавиши)
- Следит за всеми 40+ расширениями
- Инкрементальный режим (только изменённые файлы)

---

### 🏗️ Архитектура

```
Markdown-RAG-MCP-Server/
├── server.py        # MCP сервер — все инструменты (v4.0, ~1400 строк)
├── crawler.py       # Web crawler — все загрузчики (v1.0, ~450 строк)
├── watcher.py       # File watcher — авто-реиндекс (v1.0)
├── requirements.txt # Зависимости
└── data/
    ├── chroma_db/            # Векторная база (ChromaDB)
    ├── bm25_cache.pkl        # BM25 индекс (persistent)
    ├── file_hashes.json      # MD5 хеши (incrementa indexing)
    └── crawled/              # Скачанные сайты
```

---

### 📦 Новые зависимости

```
# Ядро (уже установлено)
chromadb>=0.6.0
sentence-transformers>=3.0.0
fastmcp>=2.0.0
rank-bm25>=0.2.2

# Краулер (установлено)
httpx>=0.27
beautifulsoup4>=4.12
html2text>=2024.2
lxml>=5.0

# Опционально
pip install pypdf          # PDF
pip install python-docx    # DOCX
pip install openpyxl       # XLSX
pip install python-pptx    # PPTX
pip install watchdog       # File Watcher
pip install playwright && playwright install chromium  # JS-сайты
```

---

### 🌍 Паритет с docs-mcp-server

| Возможность | docs-mcp-server | Наш сервер |
|---|---|---|
| Веб-сайты | ✅ | ✅ |
| GitHub репозитории | ✅ | ✅ |
| npm / PyPI | ✅ | ✅ |
| ZIP архивы | ✅ | ✅ |
| PDF / DOCX / XLSX / PPTX | ✅ | ✅ |
| Jupyter Notebook | ✅ | ✅ |
| Multi-collection | ✅ | ✅ |
| Hybrid search (Vector+BM25) | vector only | ✅ лучше |
| Cross-Encoder reranking | ❌ | ✅ |
| GPU ускорение | зависит | ✅ |
| Query expansion | ❌ | ✅ |
| Code-aware chunking | ❌ | ✅ |
| File Watcher авто-реиндекс | ❌ | ✅ |
| Web UI | ✅ | ❌ (MCP tools) |
