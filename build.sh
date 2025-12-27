#!/bin/bash
# ApexJavaOCR Build Script
# This script helps build and test the ApexJavaOCR project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}ApexJavaOCR Build Script${NC}"
echo "================================"

# Check for Java
if ! command -v java &> /dev/null; then
    echo -e "${RED}Error: Java is not installed${NC}"
    echo "Please install Java 21 or higher:"
    echo "  - Ubuntu/Debian: sudo apt-get install openjdk-21-jdk"
    echo "  - macOS: brew install openjdk@21"
    echo "  - Windows: Download from https://adoptium.net/"
    exit 1
fi

# Check Java version
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 21 ]; then
    echo -e "${YELLOW}Warning: Java 21 or higher is recommended (found Java $JAVA_VERSION)${NC}"
fi

# Check for Gradle
if ! command -v gradle &> /dev/null; then
    echo -e "${YELLOW}Gradle not found, using wrapper...${NC}"
    if [ ! -f "./gradlew" ]; then
        echo "Downloading Gradle wrapper..."
        mkdir -p gradle/wrapper
        cat > gradle/wrapper/gradle-wrapper.properties << 'EOF'
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-8.5-bin.zip
networkTimeout=10000
validateDistributionUrl=true
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
EOF
        # Download gradle wrapper jar (simplified approach)
        echo "Please download Gradle 8.5+ from https://gradle.org/install/"
        exit 1
    fi
    chmod +x gradlew
    GRADLE_CMD="./gradlew"
else
    GRADLE_CMD="gradle"
fi

# Parse command line arguments
BUILD_TYPE="build"
RUN_TESTS=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        build)
            BUILD_TYPE="build"
            shift
            ;;
        test)
            RUN_TESTS=true
            shift
            ;;
        clean)
            BUILD_TYPE="clean"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute build
case $BUILD_TYPE in
    clean)
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        $GRADLE_CMD clean
        ;;
    build)
        echo -e "${GREEN}Building project...${NC}"
        if [ "$VERBOSE" = true ]; then
            $GRADLE_CMD build -x test --info
        else
            $GRADLE_CMD build -x test
        fi
        echo -e "${GREEN}Build successful!${NC}"
        ;;
esac

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo -e "${GREEN}Running tests...${NC}"
    $GRADLE_CMD test
    echo -e "${GREEN}Tests completed!${NC}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
