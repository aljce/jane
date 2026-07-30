{
  description = "jane — a transformer from scratch in Rust with Burn (CUDA)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ (import rust-overlay) ];
        # CUDA is unfree.
        config.allowUnfree = true;
      };

      inherit (pkgs) lib;

      rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

      cuda = pkgs.cudaPackages;

      # Burn's `HuggingfaceDatasetLoader` shells out to a bundled importer.py.
      # Its actual imports are `datasets`, `sqlalchemy` and `pyarrow`; Pillow and
      # soundfile are needed lazily by `datasets` for Image/Audio columns.
      #
      # IMPORTANT: by default the loader builds a venv and pip-installs those
      # packages. That is broken on NixOS — pip's binary wheels (pyarrow above
      # all) ship unpatched ELF interpreters and will not run. Always construct
      # the loader with `.with_use_python_venv(false)` so it uses the ambient
      # `python3` from this shell instead.
      pythonEnv = pkgs.python3.withPackages (ps: [
        ps.datasets
        ps.sqlalchemy
        ps.pyarrow
        ps.pillow
        ps.soundfile
      ]);

      # Burn's CUDA backend (CubeCL) compiles kernels at runtime through NVRTC,
      # so nvrtc + cudart must be resolvable at *runtime*, not just build time.
      # This is also what lets us target sm_120 (Blackwell / RTX 50-series)
      # without waiting on precompiled fatbins.
      cudaLibs = [
        cuda.cuda_cudart
        cuda.cuda_nvrtc
      ];

      # libcuda.so.1 and libnvidia-ml.so.1 come from the *driver*, never from
      # nixpkgs. On WSL2 the driver is bind-mounted at /usr/lib/wsl/lib; on bare
      # NixOS it is /run/opengl-driver/lib. Include both so this shell works in
      # either place. `nvidia-smi` fails outside this shell precisely because
      # NixOS does not put /usr/lib/wsl/lib on the linker path by default.
      driverPaths = [
        "/usr/lib/wsl/lib"
        "/run/opengl-driver/lib"
      ];

      libraryPath = lib.concatStringsSep ":" (
        driverPaths ++ [ (lib.makeLibraryPath (cudaLibs ++ [ pkgs.stdenv.cc.cc.lib pkgs.zlib ])) ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "jane";

        packages = [
          rustToolchain

          # Python side of the data pipeline — see the pythonEnv comment above.
          pythonEnv

          # Native build deps: `tokenizers` builds C (oniguruma), rusqlite may
          # build bundled SQLite, and any HTTP crate wants openssl + pkg-config.
          pkgs.pkg-config
          pkgs.openssl
          pkgs.sqlite
          pkgs.zlib

          # nvcc is not needed for runtime-compiled kernels, but cuda-gdb and
          # the headers are useful when a kernel misbehaves.
          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cuda_nvrtc

          # Data wrangling: fetching the corpus and checking it.
          pkgs.curl
          pkgs.coreutils

          # Quality of life.
          pkgs.cargo-watch
          pkgs.cargo-nextest
          pkgs.tokei
        ];

        # Burn / CubeCL discovery.
        CUDA_PATH = cuda.cuda_nvcc;
        CUDA_ROOT = cuda.cuda_nvcc;
        LD_LIBRARY_PATH = libraryPath;

        # Keep rust-analyzer able to jump into std.
        RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

        # Keep the multi-GB HF cache inside the project (gitignored) rather than
        # in ~/.cache, so `rm -rf data/` genuinely resets the data pipeline.
        HF_HOME = "./data/.hf";

        # `datasets` phones home for metadata on every load; flip to 1 to work
        # entirely from cache once the corpus is downloaded.
        HF_DATASETS_OFFLINE = "0";

        # Full backtraces are worth the noise while writing kernels.
        RUST_BACKTRACE = "1";

        shellHook = ''
          echo "jane — transformer from scratch (Burn 0.20.1)"
          echo "  rustc:  $(rustc --version)"
          echo "  python: $(python3 --version) (datasets $(python3 -c 'import datasets; print(datasets.__version__)' 2>/dev/null || echo '??'))"

          if nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null; then
            :
          else
            echo "  GPU:   NOT VISIBLE — CUDA training will fail."
            echo "         Check that the host NVIDIA driver is installed and that"
            echo "         one of these exists: ${lib.concatStringsSep ", " driverPaths}"
          fi
        '';
      };

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
