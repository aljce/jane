{
  description = "loom — MCP server for multi-agent orchestration across git worktrees";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      packages.${system}.default = pkgs.buildNpmPackage {
        pname = "loom";
        version = "0.1.0";
        src = ./.;
        npmDepsHash = "sha256-I9wfB4BkR7MSMmWd6Ub+pIAUtOJmAnsmPavD5+pCgms=";
        buildPhase = ''
          npm run build
        '';
        installPhase = ''
          mkdir -p $out/lib $out/bin
          cp -r dist node_modules $out/lib/
          cp package.json $out/lib/

          cat > $out/bin/loom <<'WRAPPER'
          #!/usr/bin/env bash
          exec node "$(dirname "$(readlink -f "$0")")/../lib/dist/index.js" "$@"
          WRAPPER
          chmod +x $out/bin/loom
        '';
      };

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
