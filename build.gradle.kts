// ApexJavaOCR - High-Performance Pure Java OCR Engine
// Root Build Configuration

allprojects {
    group = "com.apexocr"
    version = "1.0.0"

    repositories {
        mavenCentral()
    }
}

//===============================================================================
// Build System Tasks
//===============================================================================

tasks.register("buildSystem") {
    description = "Generate Makefile and build scripts"
    group = "build setup"
    doLast {
        println("Build system files ready:")
        println("  - Makefile (run 'make' for full build system)")
        println("  - build.sh (simple Gradle-based script)")
        println("  - generate-makefile.sh (regenerates Makefile)")
    }
}

tasks.register("installSystem") {
    description = "Install ApexJavaOCR to system"
    group = "installation"
    dependsOn("assemble")
    doLast {
        println("Installation prepared. Run 'make install' to install system-wide")
    }
}

tasks.register("quickBuild") {
    description = "Quick build without tests"
    group = "build"
    dependsOn("assemble")
    doLast {
        println("Quick build complete. JAR files in build/libs/")
    }
}

tasks.register("fullBuild") {
    description = "Complete build with tests"
    group = "build"
    dependsOn("clean", "assemble", "test")
    doLast {
        println("Full build complete!")
    }
}

tasks.register("generateDocs") {
    description = "Generate project documentation"
    group = "documentation"
    doLast {
        mkdir("doc")
        file("doc/README.md").writeText("""
            # ApexJavaOCR Documentation
            
            ## Quick Start
            
            ### Building
            ```bash
            ./gradlew assemble     # Build JARs
            ./gradlew test         # Run tests
            make build            # Full build with Makefile
            ```
            
            ### Installing
            ```bash
            make install          # Install to /usr/local (requires root)
            make install-user     # Install to ~/.local
            ```
            
            ### Using ApexJavaOCR
            ```bash
            java -cp "libs/*" com.apexocr.cli.Main image.png
            ```
            
            ## Project Structure
            
            - `apex-ocr-core/` - Core tensor operations and neural network layers
            - `apex-ocr-engine/` - Main OCR engine implementation
            - `apex-ocr-preprocessing/` - Image preprocessing utilities
            - `apex-ocr-cli/` - Command-line interface
            
            ## Requirements
            
            - Java 11 or higher
            - Gradle 8.5+
            
            ## For Developers
            
            - Run tests: `./gradlew test`
            - Clean build: `./gradlew clean`
            - Create packages: `./gradlew distTar distZip`
            
        """.trimIndent())
        println("Documentation generated in doc/")
    }
}
