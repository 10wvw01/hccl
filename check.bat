@rem Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
@rem check bat (`check.bat`)
@echo off
setlocal

title net-proxy-client

REM Get path in current script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo %cd%

rem client package
set "JAR_FILE=net-proxy-client.jar"

rem find PID
for /f "tokens=1" %%a in ('..\jre\bin\jcmd ^| findstr /i /c:"%JAR_FILE%"') do set "PID=%%a"

if defined PID (
    set "PID=%PID:"=%
	echo Application is on.
    echo Client Java process stared with PID %PID% for %JAR_FILE%
) else (
    echo Application is off.
    echo Client Java process not found for %JAR_FILE%.
)
endlocal

if not "%1" == "skip" (
    echo Wait for 30 second and auto close current windows!
    timeout /t 30 /nobreak
)