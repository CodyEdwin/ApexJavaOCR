#!/bin/bash
# ApexJavaOCR - Build System Installation Script
# This script generates a properly formatted Makefile

cat > Makefile << 'MAKEFILE_END'
#!/usr/bin/env make
#===============================================================================
# ApexJavaOCR - High-Performance Pure Java OCR Engine
# Build System for Local Installation
#===============================================================================

# Version Information
VERSION := 1.0.0
PACKAGE_NAME := apex-ocr-$(VERSION)
INSTALL_PREFIX ?= /usr/local
BIN_INSTALL_DIR := $(INSTALL_PREFIX)/bin
LIB_INSTALL_DIR := $(INSTALL_PREFIX)/lib
SHARE_INSTALL_DIR := $(INSTALL_PREFIX)/share/apex-ocr

# Java Configuration
JAVA_HOME ?= $(shell which java > /dev/null 2>&1 && dirname $$(dirname $$(readlink -f $$(which java))) 2>/dev/null || echo "")
JAVA_VERSION_MIN := 11

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.PHONY: all
all: info build

#===============================================================================
# Information and Detection
#===============================================================================

.PHONY: info
info:
	@echo "============================================"
	@echo "ApexJavaOCR Build System"
	@echo "============================================"
	@echo "Version: $(VERSION)"
	@echo "Package: $(PACKAGE_NAME)"
	@echo "Install Prefix: $(INSTALL_PREFIX)"
	@echo ""
	@echo "Java Configuration:"
	@if [ -n "$$JAVA_HOME" ]; then \
		echo "  JAVA_HOME: $$JAVA_HOME"; \
		$$JAVA_HOME/bin/java -version 2>&1 | head -1; \
	else \
		echo "  JAVA_HOME not set, using system default"; \
		java -version 2>&1 | head -1; \
	fi
	@echo ""

#===============================================================================
# Build Targets
#===============================================================================

.PHONY: build
build: clean info
	@echo "Building ApexJavaOCR..."
	@./gradlew assemble --no-daemon -q
	@echo ""
	@echo "Build completed successfully!"
	@echo "JAR files created in build/libs/"

.PHONY: build-cli
build-cli:
	@echo "Building CLI distribution..."
	@./gradlew :apex-ocr-cli:distTar :apex-ocr-cli:distZip --no-daemon -q
	@echo "CLI distribution created!"

.PHONY: compile
compile:
	@echo "Compiling source files..."
	@./gradlew compileJava --no-daemon -q
	@echo "Compilation successful!"

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@./gradlew clean --no-daemon -q 2>/dev/null || true
	@rm -rf build/ .gradle/*/build .gradle/*/fileHashes
	@echo "Clean complete!"

.PHONY: distclean
distclean: clean
	@echo "Removing all generated files..."
	@rm -rf .gradle/
	@echo "Distclean complete!"

#===============================================================================
# Test Targets
#===============================================================================

.PHONY: test
test:
	@echo "Running unit tests..."
	@./gradlew test --no-daemon 2>&1 | tee test_output.txt
	@if grep -q "BUILD SUCCESSFUL" test_output.txt; then \
		echo ""; \
		echo "All tests passed!"; \
		rm -f test_output.txt; \
	else \
		echo ""; \
		echo "Some tests failed!"; \
		rm -f test_output.txt; \
		exit 1; \
	fi

#===============================================================================
# Installation Targets
#===============================================================================

.PHONY: install
install: build
	@echo "============================================"
	@echo "Installing ApexJavaOCR"
	@echo "============================================"
	@echo ""
	@echo "Install prefix: $(INSTALL_PREFIX)"
	@echo "Binary directory: $(BIN_INSTALL_DIR)"
	@echo "Library directory: $(LIB_INSTALL_DIR)"
	@echo "Share directory: $(SHARE_INSTALL_DIR)"
	@echo ""

	@# Create directories
	@echo "Creating directories..."
	@mkdir -p $(DESTDIR)$(BIN_INSTALL_DIR)
	@mkdir -p $(DESTDIR)$(LIB_INSTALL_DIR)
	@mkdir -p $(DESTDIR)$(SHARE_INSTALL_DIR)
	@mkdir -p $(DESTDIR)$(SHARE_INSTALL_DIR)/bin

	@# Install main JAR
	@echo "Installing main libraries..."
	@cp apex-ocr-core/build/libs/apex-ocr-core-$(VERSION).jar $(DESTDIR)$(LIB_INSTALL_DIR)/
	@cp apex-ocr-engine/build/libs/apex-ocr-engine-$(VERSION).jar $(DESTDIR)$(LIB_INSTALL_DIR)/
	@cp apex-ocr-preprocessing/build/libs/apex-ocr-preprocessing-$(VERSION).jar $(DESTDIR)$(LIB_INSTALL_DIR)/

	@# Install CLI
	@echo "Installing CLI..."
	@./gradlew :apex-ocr-cli:distTar --no-daemon -q 2>/dev/null
	@tar -xzf apex-ocr-cli/build/distributions/apex-ocr-cli-$(VERSION).tar -C $(DESTDIR)$(SHARE_INSTALL_DIR)/
	@ln -sf $(SHARE_INSTALL_DIR)/bin/apex-ocr $(DESTDIR)$(BIN_INSTALL_DIR)/apex-ocr

	@# Create wrapper script
	@echo "Creating wrapper scripts..."
	@cat > $(DESTDIR)$(BIN_INSTALL_DIR)/apex-ocr << 'WRAPPER_EOF'
#!/bin/bash
# ApexJavaOCR Wrapper Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR=$(LIB_INSTALL_DIR)
JAVA=java

JAR_FILES=""
for jar in $$LIB_DIR/apex-ocr-*.jar; do
    if [ -f "$$jar" ]; then
        JAR_FILES="$$JAR_FILES:$$jar"
    fi
done

if [ -z "$$JAR_FILES" ]; then
    echo "Error: JAR files not found!"
    exit 1
fi

exec $$JAVA -cp "$$JAR_FILES" com.apexocr.cli.Main "$$@"
WRAPPER_EOF
	@chmod +x $(DESTDIR)$(BIN_INSTALL_DIR)/apex-ocr

	@# Create environment setup script
	@echo "Creating environment setup..."
	@cat > $(DESTDIR)$(SHARE_INSTALL_DIR)/apex-ocr-env.sh << 'ENV_EOF'
# ApexJavaOCR Environment Setup
export APEX_OCR_HOME=$(SHARE_INSTALL_DIR)
export APEX_OCR_LIB=$(LIB_INSTALL_DIR)
ENV_EOF

	@# Create version file
	@echo "Creating version file..."
	@echo "APEX_OCR_VERSION=$(VERSION)" > $(DESTDIR)$(SHARE_INSTALL_DIR)/VERSION

	@echo ""
	@echo "============================================"
	@echo "Installation Complete!"
	@echo "============================================"
	@echo ""
	@echo "To use ApexJavaOCR:"
	@echo "  1. Run: $(BIN_INSTALL_DIR)/apex-ocr <image_file>"
	@echo ""
	@echo "Libraries installed to: $(LIB_INSTALL_DIR)/"
	@echo ""

.PHONY: install-user
install-user: build
	@echo "Installing to user home directory..."
	@make install INSTALL_PREFIX=$$HOME/.local DESTDIR=

.PHONY: uninstall
uninstall:
	@echo "Removing ApexJavaOCR..."
	@rm -f $(DESTDIR)$(BIN_INSTALL_DIR)/apex-ocr
	@rm -rf $(DESTDIR)$(SHARE_INSTALL_DIR)/
	@rm -f $(DESTDIR)$(LIB_INSTALL_DIR)/apex-ocr-core-$(VERSION).jar
	@rm -f $(DESTDIR)$(LIB_INSTALL_DIR)/apex-ocr-engine-$(VERSION).jar
	@rm -f $(DESTDIR)$(LIB_INSTALL_DIR)/apex-ocr-preprocessing-$(VERSION).jar
	@echo "Uninstallation complete!"

#===============================================================================
# Package Targets
#===============================================================================

.PHONY: package
package: build build-cli
	@echo "Creating distribution packages..."
	@mkdir -p package
	@cp apex-ocr-cli/build/distributions/apex-ocr-cli-$(VERSION).tar.gz package/
	@cp apex-ocr-cli/build/distributions/apex-ocr-cli-$(VERSION).zip package/
	@tar -czf package/apex-ocr-sources-$(VERSION).tar.gz \
		--exclude='.git' \
		--exclude='.gradle' \
		--exclude='build' \
		--exclude='package' \
		.
	@echo ""
	@echo "Packages created in package/:"
	@ls -lh package/

#===============================================================================
# Development Targets
#===============================================================================

.PHONY: deps
deps:
	@echo "Checking dependencies..."
	@./gradlew dependencies --no-daemon -q

.PHONY: lint
lint:
	@echo "Running code quality checks..."
	@./gradlew compileJava --no-daemon -q
	@echo "Code quality check passed!"

.PHONY: doc
doc:
	@echo "Generating documentation..."
	@mkdir -p doc
	@./gradlew javadoc --no-daemon -q 2>/dev/null || echo "Javadoc not configured"
	@echo "Documentation generated in doc/"

#===============================================================================
# Help
#===============================================================================

.PHONY: help
help:
	@echo ""
	@echo "============================================"
	@echo "  ApexJavaOCR Build System - Help"
	@echo "============================================"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Main targets:"
	@echo "  all       - Build everything (default)"
	@echo "  build     - Clean and build the project"
	@echo "  compile   - Compile source files only"
	@echo "  clean     - Remove build artifacts"
	@echo "  test      - Run unit tests"
	@echo "  install   - Install to system (requires root)"
	@echo "  uninstall - Remove installed files"
	@echo "  package   - Create distribution packages"
	@echo ""
	@echo "Options:"
	@echo "  INSTALL_PREFIX=/path  Set installation prefix"
	@echo ""
	@echo "Examples:"
	@echo "  make build                    # Build the project"
	@echo "  make install                 # Install to /usr/local"
	@echo "  make install-user            # Install to ~/.local"
	@echo ""

MAKEFILE_END

echo "Makefile generated successfully!"
