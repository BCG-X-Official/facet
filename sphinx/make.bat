@ECHO OFF
pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set NOTEBOOK_QUICK_START=..\tutorial
set BUILDDIR=build
REM say to apidoc which members to consider
REM set SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,inherited-members


if "%1" == "" goto help
if "%1" == "html" goto html

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:html
REM generate apidoc using docstrings
sphinx-apidoc -e --no-toc -o %SOURCEDIR% -f ./../src/yieldengine
REM run the sphinx build for html
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
REM run sphinx for notebook
REM %SPHINXBUILD% -b %1 -c %SOURCEDIR% %NOTEBOOK_QUICK_START% %BUILDDIR% %SPHINXOPTS%
REM clean up potentially pre-existing files in /docs
del /q /s ..\docs\* >nul
for /d %%i in (..\docs\*) do rd /s /q "%%i"
REM move the last build into /docs
set DIR_HTML=build\html
set DIR_DOCS=..\docs
for %%i in (%DIR_HTML%\*) do move "%%i" %DIR_DOCS%\ >nul
for /d %%i in (%DIR_HTML%\*) do move "%%i" %DIR_DOCS%\ >nul
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
