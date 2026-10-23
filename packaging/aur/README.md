# AUR packages

Two packages so Arch / Omarchy users can `yay -S` limbo:

| AUR package      | Builds from                       | Updates                                   |
|------------------|-----------------------------------|-------------------------------------------|
| `limbo-code`     | PyPI sdist of the latest `v*` tag | CI pushes a new PKGBUILD on every release |
| `limbo-code-git` | `main` on GitHub                  | `yay -Syu --devel` rebuilds new commits   |

Both install the `limbo` command and conflict with each other.

The files here are the source of truth; the AUR repos are copies. On a `v*`
tag, `.github/workflows/publish.yml` publishes to PyPI, then the `aur` job
bumps `pkgver`/`sha256sums`, test-builds with `makepkg` in an Arch container,
regenerates `.SRCINFO`, and pushes to `aur.archlinux.org` (only if the
`AUR_SSH_PRIVATE_KEY` secret is set).

## One-time setup (maintainer)

1. Create an account at <https://aur.archlinux.org/register>.
2. Make a dedicated key: `ssh-keygen -t ed25519 -f ~/.ssh/aur -C aur`, paste
   `~/.ssh/aur.pub` into *My Account → SSH Public Key*.
3. Add to `~/.ssh/config`:
   ```
   Host aur.archlinux.org
     IdentityFile ~/.ssh/aur
     User aur
   ```
4. First push of each package (cloning a non-existent name creates it):
   ```sh
   git clone ssh://aur@aur.archlinux.org/limbo-code.git /tmp/aur-limbo-code
   cp packaging/aur/limbo-code/{PKGBUILD,.SRCINFO} /tmp/aur-limbo-code/
   cd /tmp/aur-limbo-code
   makepkg -si                       # build + install locally first
   makepkg --printsrcinfo > .SRCINFO # always regenerate before committing
   git add PKGBUILD .SRCINFO && git commit -m "Initial import" && git push
   ```
   Repeat with `limbo-code-git`.
5. For automatic updates, add the **private** key (`~/.ssh/aur`) as the
   repository secret `AUR_SSH_PRIVATE_KEY`.

## Releasing

Tag and push as usual (`git tag v0.2.3 && git push origin v0.2.3`). PyPI and
the AUR `limbo-code` package update from that tag. For a packaging-only fix
(same upstream version), bump `pkgrel` here and push to the AUR by hand.
