{
  description = "A Nix-flake-based Node.js development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-darwin";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs {
        inherit system;
      };
    in
      pkgs.mkShell {
        # create an environment with nodejs_18, pnpm, and yarn
        packages = with pkgs; [
          python314
        ];

        shellHook = ''
          echo "`python3 --version`"
        '';
      };
  };
}
