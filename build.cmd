@echo off
setlocal

rem 1. Set the following for the options you want to build.
set CUDNN=false
set CUDA=true
set DX12=false
set OPENCL=false
set MKL=false
set DNNL=false
set OPENBLAS=false
set EIGEN=false
set TEST=false
set CUTLASS=true

if "%CUDA%"=="true" (
  if not defined CUDA_PATH (
    echo WARNING: CUDA_PATH environment variable not found. Using default value.
  ) else (
    echo CUDA_PATH found in system environment: "%CUDA_PATH%"
  )
)

rem 2. Edit the paths for the build dependencies.
if not defined CUDA_PATH set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set CUDNN_PATH=%CUDA_PATH%
set OPENCL_LIB_PATH=%CUDA_PATH%\lib\x64
set OPENCL_INCLUDE_PATH=%CUDA_PATH%\include
set OPENBLAS_PATH=C:\OpenBLAS
set MKL_PATH=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl
set DNNL_PATH=C:\dnnl_win_1.1.1_cpu_vcomp

rem 3. In most cases you won't need to change anything further down.
if exist build (
  echo.
  echo Build directory exists. Delete it for a clean rebuild?
  echo   Y = delete build\ (weights/models will be copied to weights\ first^)
  echo   N = keep build\ (incremental build; may fail if .proto changed^)
  choice /C YN /M "Delete build directory"
  if errorlevel 2 (
    echo Keeping existing build directory.
  ) else (
    if not exist weights mkdir weights
    for %%f in (build\*.pb.gz build\*.onnx build\*.pt) do copy "%%f" weights\ >nul 2>&1
    rd /s /q build
    echo Deleted build\ - saved any .pb.gz/.onnx/.pt files to weights\
  )
)

set CC=cl
set CXX=cl
set CC_LD=link
set CXX_LD=link

if exist "C:\Program Files\Microsoft Visual Studio\2022" (
  where /q cl
  if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2022
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2019
) else (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2022
)

set BLAS=true
if %MKL%==false if %DNNL%==false if %OPENBLAS%==false if %EIGEN%==false set BLAS=false

if "%CUDA_PATH%"=="%CUDNN_PATH%" (
  set CUDNN_LIB_PATH=%CUDNN_PATH%\lib\x64
  set CUDNN_INCLUDE_PATH=%CUDNN_PATH%\include
) else (
  set CUDNN_LIB_PATH=%CUDA_PATH%\lib\x64,%CUDNN_PATH%\lib\x64
  set CUDNN_INCLUDE_PATH=%CUDA_PATH%\include,%CUDNN_PATH%\include
)

if %CUDNN%==true set PATH=%CUDA_PATH%\bin;%PATH%

meson setup build --backend %backend% --buildtype release -Ddx=%DX12% -Dcudnn=%CUDNN% -Dplain_cuda=%CUDA% ^
-Dopencl=%OPENCL% -Dblas=%BLAS% -Dmkl=%MKL% -Dopenblas=%OPENBLAS% -Ddnnl=%DNNL% -Dgtest=%TEST% ^
-Dcudnn_include="%CUDNN_INCLUDE_PATH%" -Dcudnn_libdirs="%CUDNN_LIB_PATH%" ^
-Dmkl_include="%MKL_PATH%\include" -Dmkl_libdirs="%MKL_PATH%\lib\intel64" -Ddnnl_dir="%DNNL_PATH%" ^
-Dopencl_libdirs="%OPENCL_LIB_PATH%" -Dopencl_include="%OPENCL_INCLUDE_PATH%" ^
-Dopenblas_include="%OPENBLAS_PATH%\include" -Dopenblas_libdirs="%OPENBLAS_PATH%\lib" ^
-Dcutlass="%CUTLASS%" ^
-Donnx=false ^
-Ddefault_library=static

if errorlevel 1 exit /b

pause

cd build

msbuild /m /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=true ^
/p:PreferredToolArchitecture=x64 lc0.sln /filelogger
