{
  pkgs,
  lib,
  stdenv,
  ...
}:
let
  pythonPackages = pkgs.python313Packages;
in
pkgs.mkShell {
  buildInputs = [
    pythonPackages.python
    pythonPackages.venvShellHook
    pkgs.autoPatchelfHook
    pythonPackages.onnxruntime
    pkgs.onnxruntime
    pythonPackages.watchdog
    pythonPackages.python-dotenv
    pythonPackages.openai
    pythonPackages.pymupdf
    pythonPackages.pillow
    pythonPackages.numpy
    pythonPackages.torch
    pythonPackages.torchvision
    pythonPackages.transformers
    pythonPackages.sentencepiece

  ];
  venvDir = "./.venv";
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -U jupyter
    autoPatchelf ./.venv
  '';
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export JAVA_HOME=${pkgs.jdk11.home}
    export PATH="${pkgs.jdk11}/bin:$PATH"
    export LD_LIBRARY_PATH=${
      lib.makeLibraryPath [
        stdenv.cc.cc
      ]
    }:$LD_LIBRARY_PATH
  '';
}
