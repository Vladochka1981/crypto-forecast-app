Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "      Запуск Crypto Forecast App" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ОШИБКА] Docker не найден! Пожалуйста, установите Docker Desktop." -ForegroundColor Red
    pause
    exit
}

# 1. Создание .env (БЕЗ BOM)
if (-not (Test-Path ".env")) {
    Write-Host "[1/5] Создаем файл конфигурации .env..." -ForegroundColor Yellow
    
    if (Test-Path ".env.example") {
        # Копируем шаблон без BOM
        $content = Get-Content ".env.example" -Raw -Encoding UTF8
        [System.IO.File]::WriteAllText("$PWD\.env", $content, (New-Object System.Text.UTF8Encoding $false))
    } else {
        Write-Host "[ОШИБКА] Файл .env.example не найден!" -ForegroundColor Red
        pause
        exit
    }
    
    Write-Host ""
    Write-Host "Введите ваш ключ FRED API (если есть). Если ключа нет, нажмите Enter."
    $fredKey = Read-Host "FRED_API_KEY"
    
    if (-not [string]::IsNullOrWhiteSpace($fredKey)) {
        $content = Get-Content ".env"
        $content = $content -replace "^FRED_API_KEY=.*", "FRED_API_KEY=$fredKey"
        # Запись БЕЗ BOM
        [System.IO.File]::WriteAllText("$PWD\.env", ($content -join "`r`n"), (New-Object System.Text.UTF8Encoding $false))
        Write-Host "Ключ сохранен!" -ForegroundColor Green
    } else {
        Write-Host "Ключ пропущен." -ForegroundColor Gray
    }
} else {
    Write-Host "[1/5] Файл .env уже существует. Используем сохраненные настройки." -ForegroundColor Yellow
}

# 2. Запуск контейнеров
Write-Host "[2/5] Собираем и запускаем контейнеры (это может занять время)..." -ForegroundColor Yellow
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ОШИБКА] Не удалось запустить Docker Compose!" -ForegroundColor Red
    pause
    exit
}

# 3. Ожидание Postgres
Write-Host "[3/5] Ожидаем готовности PostgreSQL" -ForegroundColor Yellow -NoNewline
$dots = "."
while ($true) {
    docker compose exec -T postgres pg_isready -U crypto -d crypto *> $null
    if ($LASTEXITCODE -eq 0) { break }
    
    Write-Host $dots -NoNewline
    Start-Sleep -Seconds 1
    $dots += "."
    if ($dots.Length -gt 4) { $dots = "." }
}
Write-Host ""
Write-Host "База данных готова!" -ForegroundColor Green

# 4. МИГРАЦИЯ (принудительно)
Write-Host "[4/5] Запускаем миграцию данных из SQLite в Postgres..." -ForegroundColor Yellow

# Проверяем, что файл базы вообще существует внутри контейнера
Write-Host "  Проверяем наличие файла /app/data/crypto.db..." -ForegroundColor Gray
docker compose exec -T app ls -la /app/data/crypto.db
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ОШИБКА] Файл crypto.db не найден в контейнере! Проверьте docker-compose.yml (volumes)." -ForegroundColor Red
    Write-Host "  Текущий docker-compose.yml должен содержать в сервисе app:" -ForegroundColor Yellow
    Write-Host "    volumes:" -ForegroundColor Gray
    Write-Host "      - ./data:/app/data" -ForegroundColor Gray
    pause
    exit
}

# Запускаем миграцию внутри контейнера
Write-Host "  Запускаем скрипт миграции..." -ForegroundColor Gray
docker compose exec -T app python scripts/migrate_sqlite_to_pg.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ОШИБКА] Миграция завершилась с ошибкой. Проверьте логи выше." -ForegroundColor Red
    pause
    exit
} else {
    Write-Host "Миграция завершена успешно!" -ForegroundColor Green
}

# 5. Ожидание готовности приложения (чтобы сайт не открылся раньше времени)
Write-Host "[5/5] Ожидаем полной загрузки приложения..." -ForegroundColor Yellow -NoNewline
$appReady = $false
$attempts = 0
while (-not $appReady -and $attempts -lt 30) {
    Start-Sleep -Seconds 1
    $attempts++
    # Пробуем достучаться до эндпоинта /
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $appReady = $true
        }
    } catch {
        # Еще не готово
    }
    Write-Host "." -NoNewline
}
Write-Host ""

if ($appReady) {
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "✅ Приложение успешно запущено!" -ForegroundColor Green
    Write-Host "🌐 Открываем http://localhost:8000/" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Start-Process "http://localhost:8000/"
} else {
    Write-Host "⚠️ Приложение запущено, но не отвечает. Проверьте логи:" -ForegroundColor Red
    Write-Host "   docker compose logs app --tail 50" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Для остановки приложения выполните: docker compose down"
pause