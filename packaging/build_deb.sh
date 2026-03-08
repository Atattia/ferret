#!/usr/bin/env bash
# Build a .deb package from the PyInstaller output.
# Usage: bash packaging/build_deb.sh [version]
# Requires: dpkg-deb (pre-installed on Ubuntu)
set -euo pipefail

VERSION="${1:-1.0.0}"
ARCH="amd64"
PKG="ferret_${VERSION}_${ARCH}"
BUILD_SRC="dist/ferret"
DEB_ROOT="packaging/_deb_build/${PKG}"

if [ ! -d "$BUILD_SRC" ]; then
  echo "Error: PyInstaller output not found at $BUILD_SRC"
  echo "Run PyInstaller first: pyinstaller ferret.spec"
  exit 1
fi

echo "==> Preparing .deb structure..."
rm -rf "packaging/_deb_build"
mkdir -p "${DEB_ROOT}/DEBIAN"
mkdir -p "${DEB_ROOT}/opt/ferret"
mkdir -p "${DEB_ROOT}/usr/share/applications"
mkdir -p "${DEB_ROOT}/usr/local/bin"

echo "==> Copying app files..."
cp -r "${BUILD_SRC}/." "${DEB_ROOT}/opt/ferret/"

echo "==> Writing control file..."
cat > "${DEB_ROOT}/DEBIAN/control" <<EOF
Package: ferret
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Depends: tesseract-ocr, libxcb-xinerama0, libxcb-cursor0, libxcb-icccm4
Maintainer: Mahmoud Yousry
Description: Local semantic search for your files
 Ferret watches your folders for document changes and indexes them using
 a local AI model, enabling fast semantic search via a hotkey-triggered
 search bar. All data stays on your machine.
EOF

echo "==> Writing postinst script..."
cat > "${DEB_ROOT}/DEBIAN/postinst" <<'EOF'
#!/bin/bash
chmod +x /opt/ferret/ferret
EOF
chmod 755 "${DEB_ROOT}/DEBIAN/postinst"

echo "==> Copying icon..."
mkdir -p "${DEB_ROOT}/usr/share/icons/hicolor/256x256/apps"
cp "${BUILD_SRC}/_internal/assets/ferret.png" "${DEB_ROOT}/usr/share/icons/hicolor/256x256/apps/ferret.png"

echo "==> Copying desktop file..."
cp packaging/ferret.desktop "${DEB_ROOT}/usr/share/applications/"

echo "==> Creating launcher..."
cat > "${DEB_ROOT}/usr/local/bin/ferret" <<'EOF'
#!/bin/bash
exec /opt/ferret/ferret "$@"
EOF
chmod +x "${DEB_ROOT}/usr/local/bin/ferret"

echo "==> Building .deb..."
dpkg-deb --build "${DEB_ROOT}" "${PKG}.deb"
echo ""
echo "Done: ${PKG}.deb"
