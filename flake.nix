{
  description = "Multi-environment project example";
  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    dev-environments.url = "github:Govcraft/dev-environments";
  };
  outputs =
    inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.dev-environments.flakeModules.rust
        inputs.dev-environments.flakeModules.golang
        inputs.dev-environments.flakeModules.node
        inputs.dev-environments.flakeModules.typst
      ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          ...
        }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
          };
        in
        {

          # Golang Development Environment Options
          # ----------------------------------
          # enable: boolean - Enable/disable the Golang environment
          # goVersion: enum - Go toolchain version ("1.18", "1.19", "1.20", "1.21, "1.22"", "1.23") (default: "1.23")
          # withTools: list of strings - Additional Go tools to include (e.g., "golint", "gopls")
          # extraPackages: list of packages - Additional packages to include
          go-dev = {
            # enable = true;
            # goVersion = "1.23";
            # withTools = [ "gopls" "golint" ];
            # extraPackages = [ ];
          };

          # Rust Development Environment Options
          # ----------------------------------
          # enable: boolean - Enable/disable the Rust environment
          # rustVersion: enum - Rust toolchain ("stable", "beta", "nightly") (default: "stable")
          # withTools: list of strings - Additional Rust tools to include (converted to cargo-*)
          # extraPackages: list of packages - Additional packages to include
          # ide.type: enum - IDE preference ("rust-rover", "vscode", "none") (default: "none")
          rust-dev = {
            enable = true;
            rustVersion = "stable";
            # Example configuration:
            # withTools = [ ];  # Will be prefixed with cargo-
            extraPackages = [
              pkgs.openssl
              pkgs.openssl.dev
              pkgs.pkg-config
            ]; # Add openssl libs, dev libs, and pkg-config
            #ide.type = "vscode";
          };

          # Node.js Development Environment Options
          # -------------------------------------
          # enable: boolean - Enable/disable the Node environment
          # nodeVersion: string - Version of Node.js to use (default: "20")
          # withTools: list of strings - Global tools to include (default: ["typescript" "yarn" "pnpm"])
          # extraPackages: list of packages - Additional packages to include
          # ide.type: enum - IDE preference ("vscode", "webstorm", "none") (default: "none")
          node-dev = {
            # Example configuration:
            enable = true;
            # nodeVersion = "20";
            # withTools = [ "typescript" "yarn" "pnpm" ];
            # extraPackages = [ ];
            ide.type = "vscode";
          };

          typst-dev = {
            # Example configuration:
            # enable = true;
            # withTools = [ "typst-fmt" "typst-lsp" ];
            # extraPackages = [ ];
            # ide.type = "none";
          };
          # Create the combined shell
          devShells.default = pkgs.mkShell {
            buildInputs = nixpkgs.lib.flatten (nixpkgs.lib.attrValues config.env-packages ++ [ ]);
            # Combine existing hooks with new exports for OpenSSL
            shellHook = ''
              ${nixpkgs.lib.concatStringsSep "\n" (nixpkgs.lib.attrValues config.env-hooks)}

              # Export paths for openssl-sys build script
              export OPENSSL_DIR="${pkgs.openssl.out}"
              export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
              export OPENSSL_INCLUDE_DIR="${pkgs.openssl.dev}/include"
              # Ensure pkg-config can find the openssl .pc file
              export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.pkg-config}/lib/pkgconfig:$PKG_CONFIG_PATH"
              # Debug: List contents of OpenSSL lib directory to verify
              echo "OpenSSL lib directory contents:"
              ls -la ${pkgs.openssl.out}/lib || echo "Failed to list OpenSSL lib directory"
              echo ">>> OpenSSL environment variables set by shellHook <<<"
            '';
          };
        };
    };
}
