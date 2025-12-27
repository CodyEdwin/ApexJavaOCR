#!/bin/bash
#===============================================================================
# ApexJavaOCR Build Script
# Simple interface to the Gradle build system
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}ApexJavaOCR Build System${NC}"
echo "================================"

# Check Java
if ! command -v java &> /dev/null; then
    echo -e "${RED}Error: Java not installed${NC}"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 11 ]; then
    echo -e "${RED}Error: Java 11+ required (found: $JAVA_VERSION)${NC}"
    exit 1
fi

# Parse arguments
CMD="${1:-build}"
shift 2>/dev/null || true

# Show help
case $CMD in
    help|--help|-h)
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  build         - Build project (default)"
        echo "  quick         - Quick build without tests"
        echo "  full          - Full build with tests"
        echo "  clean         - Clean build artifacts"
        echo "  test          - Run tests"
        echo "  install       - Build and prepare install"
        echo "  package       - Create distribution packages"
        echo "  docs          - Generate documentation"
        echo "  info          - Show project info"
        echo "  help          - Show this message"
        echo ""
        echo "Options:"
        echo "  -PinstallPrefix=/path  Set install prefix"
        echo ""
        echo "Examples:"
        echo "  $0 build              # Build JARs"
        echo "  $0 test               # Run tests"
        echo "  $0 full               # Full build with tests"
        echo "  $0 package            # Create packages"
        echo "  $0 -PinstallPrefix=/opt/apex install  # Custom prefix"
        echo ""
        echo "Or use Makefile directly:"
        echo "  make build"
        echo "  make install"
        echo "  make package"
        exit 0
        ;;
esac

# Execute Gradle command
case $CMD in
    build|quick)
        echo -e "${GREEN}Building ApexJavaOCR...${NC}"
        ./gradlew assemble -q --no-daemon
        echo -e "${GREEN}Build successful!${NC}"
        ;;
    full)
        echo -e "${GREEN}Full build with tests...${NC}"
        ./gradlew clean assemble test --no-daemon
        echo -e "${GREEN}Full build complete!${NC}"
        ;;
    clean)
        echo -e "${YELLOW}Cleaning...${NC}"
        ./gradlew clean -q --no-daemon
        rm -rf build/ .gradle/*/build
        echo -e "${GREEN}Clean complete!${NC}"
        ;;
    test)
        echo -e "${GREEN}Running tests...${NC}"
        ./gradlew test --no-daemon
        ;;
    install)
        echo -e "${GREEN}Preparing installation...${NC}"
        ./gradlew clean assemble -q --no-daemon
        echo -e "${GREEN}Build complete!${NC}"
        echo ""
        echo "To install system-wide, run:"
        echo "  make install"
        echo "  or"
        echo "  sudo ./build.sh install"
        ;;
    package)
        echo -e "${GREEN}Creating packages...${NC}"
        ./gradlew :apex-ocr-cli:distTar :apex-ocr-cli:distZip --no-daemon
        echo -e "${GREEN}Packages created!${NC}"
        ;;
    docs|doc)
        echo -e "${GREEN}Generating documentation...${NC}"
        ./gradlew generateDocs --no-daemon
        ;;
    info)
        echo -e "${GREEN}Project information:${NC}"
        echo "Project: com.apexocr:apex-java-ocr"
        echo "Version: 1.0.0"
        echo "Java: $(java -version 2>&1 | head -n1)"
        echo ""
        echo "Subprojects:"
        echo "  - apex-ocr-cli"
        echo "  - apex-ocr-core"
        echo "  - apex-ocr-engine"
        echo "  - apex-ocr-preprocessing"
        ;;
    system)
        echo -e "${GREEN}Build system info:${NC}"
        ./gradlew buildSystem --no-daemon
        ;;
    *)
        echo -e "${RED}Unknown command: $CMD${NC}"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
