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

          # Merged toolkit: needed for the CUDA headers NVRTC includes at
          # runtime (see CUDA_PATH below), not just for nvcc.
          cuda.cudatoolkit
          cuda.cuda_cudart
          cuda.cuda_nvrtc

          # Data wrangling: fetching the corpus and checking it.
          pkgs.curl
          pkgs.coreutils

          # Shared compilation cache. This is what makes parallel agent
          # worktrees affordable: each worktree keeps its own `target/` (cargo
          # takes an exclusive lock on a build dir, so *sharing* one would
          # serialize agents), while sccache means Burn's dependency tree is
          # actually compiled only once across all of them.
          pkgs.sccache

          pkgs.git

          # Quality of life.
          pkgs.cargo-watch
          pkgs.cargo-nextest
          pkgs.tokei
        ];

        # CubeCL generates CUDA C that `#include <cuda_runtime.h>` and compiles it
        # at runtime with NVRTC, passing `--include-path=$CUDA_PATH/include`
        # (cubecl-cuda's `install::include_path`). So CUDA_PATH must point at a
        # tree that actually has `include/cuda_runtime.h` — the split
        # `cuda_nvcc`/`cuda_cudart` outputs do not. `cudatoolkit` is the merged
        # distribution and does.
        CUDA_PATH = cuda.cudatoolkit;
        CUDA_ROOT = cuda.cudatoolkit;
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

        # Route every rustc call through sccache. Set here rather than in
        # .cargo/config.toml so it can never point at an sccache that isn't on
        # PATH. Note `incremental = false` in .cargo/config.toml is a hard
        # requirement for this to hit — sccache cannot cache incremental builds.
        RUSTC_WRAPPER = "sccache";

        SCCACHE_CACHE_SIZE = "20G";

        shellHook = ''
          # Absolute and outside the repo, so every agent worktree shares one
          # cache. Expanded here rather than in a Nix string because flakes
          # evaluate purely — builtins.getEnv "HOME" would yield "".
          export SCCACHE_DIR="''${SCCACHE_DIR:-$HOME/.cache/sccache/jane}"
          mkdir -p "$SCCACHE_DIR"

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
